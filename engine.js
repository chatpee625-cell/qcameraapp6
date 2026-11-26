/**
 * @file engine.js
 * @description TensorFlow.js による実学習・実推論エンジン (v4)
 *
 * 仕組み:
 *  - MobileNet v1 (0.25, 224) を特徴抽出器として凍結使用 (Backbone凍結)
 *  - 中間層 conv_pw_11_relu の出力 = 14x14マス × 128次元 の特徴マップ
 *  - 各マスが「対象の内側か外側か」を小さな2層ネット (Head) で学習
 *  - 推論時は「対象らしいマス」の塊を1つの枠にまとめて出力
 */

(function() {
    const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
    const FEATURE_LAYER = 'conv_pw_11_relu';
    const INPUT = 224;
    const GRID = 14;
    const FEAT = 128;
    const HIDDEN = 32;
    const LEARNING_RATE = 0.01;   // 小さなHeadなので高め
    const BATCH = 256;
    const HEAD_KEY = 'yolo_head_weights_v4';

    /**
     * FeatureExtractor: 凍結Backbone
     */
    class FeatureExtractor {
        constructor() {
            this.model = null;
            this.ready = false;
        }

        async load() {
            if (this.ready) return;
            if (!window.tf) throw new Error('TensorFlow.js not loaded');
            const base = await tf.loadLayersModel(MODEL_URL);
            const layer = base.getLayer(FEATURE_LAYER);
            this.model = tf.model({ inputs: base.inputs, outputs: layer.output });
            // ウォームアップ (初回推論の遅延をここで吸収)
            tf.tidy(() => { this.model.predict(tf.zeros([1, INPUT, INPUT, 3])); });
            head.ensure();
            this.ready = true;
        }

        /**
         * @param {HTMLCanvasElement} source 224x224 のキャンバス
         * @returns {tf.Tensor2D} [196, 128] (呼び出し側でdispose)
         */
        extract(source, opts = {}) {
            const { flip = false, brightness = 1 } = opts;
            return tf.tidy(() => {
                let img = tf.browser.fromPixels(source).toFloat();
                if (img.shape[0] !== INPUT || img.shape[1] !== INPUT) {
                    img = tf.image.resizeBilinear(img, [INPUT, INPUT]);
                }
                if (flip) img = tf.reverse(img, 1);
                if (brightness !== 1) img = img.mul(brightness).clipByValue(0, 255);
                const x = img.div(127.5).sub(1).expandDims(0);
                return this.model.predict(x).reshape([GRID * GRID, FEAT]);
            });
        }
    }

    /**
     * DetectionHead: 学習対象 (マスごとの対象確率を出す)
     */
    class DetectionHead {
        constructor() {
            this.trained = false;
            this.vars = null; // tf読み込み後に ensure() で生成
        }

        ensure() {
            if (!this.vars) this._init();
        }

        _init() {
            if (this.vars) this.vars.forEach(v => v.dispose());
            this.w1 = tf.variable(tf.randomNormal([FEAT, HIDDEN], 0, Math.sqrt(2 / FEAT)));
            this.b1 = tf.variable(tf.zeros([HIDDEN]));
            this.w2 = tf.variable(tf.randomNormal([HIDDEN, 1], 0, Math.sqrt(1 / HIDDEN)));
            this.b2 = tf.variable(tf.fill([1], -2)); // 初期は「対象ではない」寄り
            this.vars = [this.w1, this.b1, this.w2, this.b2];
        }

        reset() {
            this._init();
            this.trained = false;
        }

        logits(x) {
            return tf.tidy(() => x.matMul(this.w1).add(this.b1).relu().matMul(this.w2).add(this.b2));
        }

        /** 学習結果をJSONに (ファイル書き出し・localStorage共通) */
        toJSON() {
            return {
                app: 'yolo-camera',
                version: 4,
                grid: GRID, feat: FEAT, hidden: HIDDEN,
                savedAt: new Date().toISOString(),
                vars: this.vars.map(v => ({ shape: v.shape, values: Array.from(v.dataSync()) }))
            };
        }

        /** JSONから学習結果を復元。形が合わなければ例外 */
        fromJSON(obj) {
            const vars = Array.isArray(obj) ? obj : (obj && obj.vars); // 旧形式(配列)にも対応
            if (!vars || vars.length !== this.vars.length) throw new Error('学習データの形式が違います');
            vars.forEach((d, i) => {
                if (JSON.stringify(d.shape) !== JSON.stringify(this.vars[i].shape)) {
                    throw new Error('学習データの形式が違います');
                }
            });
            vars.forEach((d, i) => {
                const t = tf.tensor(d.values, d.shape);
                this.vars[i].assign(t);
                t.dispose();
            });
            this.trained = true;
        }

        save() {
            try {
                localStorage.setItem(HEAD_KEY, JSON.stringify(this.toJSON()));
            } catch (e) {
                console.warn('[DetectionHead] save failed', e);
            }
        }

        load() {
            try {
                const raw = localStorage.getItem(HEAD_KEY);
                if (!raw) return false;
                this.fromJSON(JSON.parse(raw));
                return true;
            } catch (e) {
                console.warn('[DetectionHead] load failed', e);
                return false;
            }
        }

        clearSaved() {
            try { localStorage.removeItem(HEAD_KEY); } catch (e) { /* noop */ }
        }
    }

    const extractor = new FeatureExtractor();
    const head = new DetectionHead();

    /**
     * BBox → 14x14マスの正解ラベル
     * y: 1=対象, 0=背景 / m: 1=学習に使う, 0=境界なので無視
     */
    function makeLabels(boxes, flip) {
        const y = new Float32Array(GRID * GRID);
        const m = new Float32Array(GRID * GRID).fill(1);
        // 背景だけのサンプル: 全マス「対象ではない」
        if (!boxes || boxes.length === 0) return { y, m };

        const half = 0.5 / GRID;
        const bs = boxes.map(bb => flip ? { x: 1 - bb.x - bb.w, y: bb.y, w: bb.w, h: bb.h } : bb);
        for (let r = 0; r < GRID; r++) {
            for (let c = 0; c < GRID; c++) {
                const cx = (c + 0.5) / GRID;
                const cy = (r + 0.5) / GRID;
                const i = r * GRID + c;
                let inside = false, nearEdge = false;
                for (const b of bs) {
                    if (cx >= b.x && cx <= b.x + b.w && cy >= b.y && cy <= b.y + b.h) inside = true;
                    else if (cx >= b.x - half && cx <= b.x + b.w + half &&
                             cy >= b.y - half && cy <= b.y + b.h + half) nearEdge = true;
                }
                if (inside) { y[i] = 1; m[i] = 1; }
                else if (nearEdge) { m[i] = 0; } // 境界は学習に使わない
            }
        }
        // 小さすぎる枠でも中心マスは必ず正解にする
        for (const b of bs) {
            const c = Math.min(GRID - 1, Math.max(0, Math.floor((b.x + b.w / 2) * GRID)));
            const r = Math.min(GRID - 1, Math.max(0, Math.floor((b.y + b.h / 2) * GRID)));
            y[r * GRID + c] = 1; m[r * GRID + c] = 1;
        }
        return { y, m };
    }

    /**
     * DataManager: 学習サンプル管理
     */
    class DataManager {
        constructor() {
            this.dataset = []; // { image: 224x224 canvas, label, boxes: [{x,y,w,h}] (空=背景のみ) }
        }
        addSample(image, label, boxes) {
            this.dataset.push({
                id: Date.now() + Math.random().toString(36).substr(2, 5),
                image,
                label,
                boxes: (boxes || []).map(b => ({ ...b }))
            });
        }
        clear() { this.dataset = []; }
        counts() {
            const obj = this.dataset.filter(d => d.boxes.length > 0).length;
            return { obj, bg: this.dataset.length - obj };
        }
        getDataset() { return this.dataset; }
        size() { return this.dataset.length; }
    }

    /**
     * TrainingEngine: Headの実学習 (データ拡張 / 検証分割 / Early Stopping)
     */
    class TrainingEngine {
        constructor() {
            this.config = AppUtils.StorageManager.getConfig();
            this.validationSplit = 0.2;
            this.earlyStoppingPatience = 5;
            this.bestValLoss = Infinity;
            this.patienceCounter = 0;
            this.isTraining = false;
        }

        async _buildTensors(samples, augment) {
            const feats = [], ys = [], ms = [];
            for (const s of samples) {
                const variants = augment
                    ? [
                        { flip: false, brightness: 1 },
                        { flip: true, brightness: 1 },
                        { flip: false, brightness: AppUtils.AugmentationEngine.getBrightnessFactor(this.config.augmentationStrength) },
                        { flip: true, brightness: AppUtils.AugmentationEngine.getBrightnessFactor(this.config.augmentationStrength) }
                      ]
                    : [{ flip: false, brightness: 1 }];
                for (const v of variants) {
                    feats.push(extractor.extract(s.image, v));
                    const { y, m } = makeLabels(s.boxes, v.flip);
                    ys.push(y); ms.push(m);
                    await tf.nextFrame(); // UIを止めない
                }
            }
            const X = tf.concat(feats);
            feats.forEach(t => t.dispose());
            const n = ys.length * GRID * GRID;
            const yAll = new Float32Array(n), mAll = new Float32Array(n);
            ys.forEach((y, i) => { yAll.set(y, i * GRID * GRID); mAll.set(ms[i], i * GRID * GRID); });
            return { X, yArr: yAll, mArr: mAll, n };
        }

        _weights(yArr, mArr, posWeight) {
            const w = new Float32Array(yArr.length);
            for (let i = 0; i < w.length; i++) w[i] = mArr[i] * (yArr[i] > 0.5 ? posWeight : 1);
            return w;
        }

        _lossValue(X, Y, W) {
            let v = 0;
            tf.tidy(() => { v = tf.losses.sigmoidCrossEntropy(Y, head.logits(X), W).dataSync()[0]; });
            return v;
        }

        async runTrainingLoop(dataset, onEpochComplete) {
            if (!extractor.ready) return { success: false, error: 'モデルが読み込まれていません' };
            if (!dataset || dataset.length < 2) {
                return { success: false, error: '学習には最低2枚以上のデータが必要です' };
            }
            if (!dataset.some(d => d.boxes.length > 0)) {
                return { success: false, error: '対象を囲んだ写真が1枚もありません' };
            }

            this.isTraining = true;
            this.patienceCounter = 0;
            this.bestValLoss = Infinity;
            const trainLossHistory = [], valLossHistory = [];
            const disposables = [];
            let best = null;
            let optimizer = null;

            try {
                const { trainSet, valSet } = this._splitDataset(dataset);
                const tr = await this._buildTensors(trainSet, true);
                const va = await this._buildTensors(valSet, false);

                // 正例が少ないので重みでバランスを取る
                let pos = 0, neg = 0;
                for (let i = 0; i < tr.n; i++) {
                    if (tr.mArr[i]) { if (tr.yArr[i] > 0.5) pos++; else neg++; }
                }
                const posWeight = Math.min(5, Math.max(1, neg / Math.max(1, pos))); // 強すぎると「迷ったら対象」になる

                const Xtr = tr.X, Ytr = tf.tensor2d(tr.yArr, [tr.n, 1]), Wtr = tf.tensor2d(this._weights(tr.yArr, tr.mArr, posWeight), [tr.n, 1]);
                const Xva = va.X, Yva = tf.tensor2d(va.yArr, [va.n, 1]), Wva = tf.tensor2d(this._weights(va.yArr, va.mArr, posWeight), [va.n, 1]);
                disposables.push(Xtr, Ytr, Wtr, Xva, Yva, Wva);

                head.reset();
                optimizer = tf.train.adam(LEARNING_RATE);
                const indices = Array.from({ length: tr.n }, (_, i) => i);

                for (let epoch = 1; epoch <= this.config.epochs; epoch++) {
                    if (!this.isTraining) break;

                    // シャッフルしてミニバッチ学習
                    for (let i = indices.length - 1; i > 0; i--) {
                        const j = Math.floor(Math.random() * (i + 1));
                        [indices[i], indices[j]] = [indices[j], indices[i]];
                    }
                    for (let s = 0; s < indices.length; s += BATCH) {
                        const batchIdx = indices.slice(s, s + BATCH);
                        tf.tidy(() => {
                            const idx = tf.tensor1d(batchIdx, 'int32');
                            const xb = tf.gather(Xtr, idx), yb = tf.gather(Ytr, idx), wb = tf.gather(Wtr, idx);
                            optimizer.minimize(() => tf.losses.sigmoidCrossEntropy(yb, head.logits(xb), wb), false, head.vars);
                        });
                    }

                    const trainLoss = this._lossValue(Xtr, Ytr, Wtr);
                    const valLoss = this._lossValue(Xva, Yva, Wva);
                    trainLossHistory.push(trainLoss);
                    valLossHistory.push(valLoss);

                    let isEarlyStopped = false;
                    if (valLoss < this.bestValLoss - 0.001) {
                        this.bestValLoss = valLoss;
                        this.patienceCounter = 0;
                        if (best) best.forEach(t => t.dispose());
                        best = head.vars.map(v => tf.keep(v.clone()));
                        this._saveCheckpoint(epoch);
                    } else {
                        this.patienceCounter++;
                        if (this.patienceCounter >= this.earlyStoppingPatience) {
                            console.warn(`[TrainingEngine] Early Stopping at Epoch ${epoch}`);
                            isEarlyStopped = true;
                        }
                    }

                    if (onEpochComplete) {
                        onEpochComplete({
                            epoch,
                            maxEpochs: this.config.epochs,
                            trainLoss,
                            valLoss,
                            trainHistory: trainLossHistory,
                            valHistory: valLossHistory,
                            progress: epoch / this.config.epochs,
                            isEarlyStopped
                        });
                    }
                    if (isEarlyStopped) break;

                    // サーマルスロットリング抑制
                    await new Promise(resolve => setTimeout(resolve, 80));
                }

                // 最良の重みに戻す
                if (best) best.forEach((t, i) => head.vars[i].assign(t));
                head.trained = true;
                head.save();
            } catch (e) {
                console.error('[TrainingEngine] failed', e);
                return { success: false, error: e.message };
            } finally {
                this.isTraining = false;
                if (best) best.forEach(t => t.dispose());
                disposables.forEach(t => t.dispose());
                if (optimizer) optimizer.dispose();
                this._releaseMemory();
            }

            return {
                success: true,
                finalValLoss: this.bestValLoss,
                trainHistory: trainLossHistory,
                valHistory: valLossHistory
            };
        }

        _splitDataset(data) {
            const shuffled = [...data].sort(() => Math.random() - 0.5);
            const valCount = Math.max(1, Math.floor(shuffled.length * this.validationSplit));
            const trainCount = shuffled.length - valCount;
            return { trainSet: shuffled.slice(0, trainCount), valSet: shuffled.slice(trainCount) };
        }

        _saveCheckpoint(epoch) {
            AppUtils.StorageManager.saveTrainingState({
                bestLoss: this.bestValLoss,
                savedAtEpoch: epoch,
                timestamp: Date.now()
            });
        }

        _releaseMemory() {
            console.log('[TrainingEngine] tensors in memory:', tf.memory().numTensors);
        }

        stop() { this.isTraining = false; }
    }

    /**
     * InferenceEngine: 実推論 (マスの塊 → 枠)
     */
    class InferenceEngine {
        constructor() {
            this.isActive = false;
            this.confidenceThreshold = 0.6;
            this.maxBoxes = 5;
            this.targetLabel = 'Target';
            this.tracks = []; // { id, box, conf, hits, miss }
            this.nextId = 1;  // 枠ごとの番号 (測定値をなめらかにするため、同じ物には同じ番号)
            this.lastTime = 0;

            // 表示の落ち着かせ方 (検出の判定そのものには影響しない)
            this.smoothTime = 0.25;  // 秒。枠が新しい位置に追いつく速さ (大きいほどゆっくり)
            this.confirmHits = 2;    // 何フレーム続けて見えたら表示するか (一瞬の誤検出を出さない)
            this.holdMisses = 4;     // 何フレーム見失うまで表示を続けるか (点滅を防ぐ)
            this.edgeLow = 0.3;      // 枠の端を半マス単位で調整するときの下限確率
            this.deadZone = 0.02;    // 画面の2%未満のずれは無視して枠を止めておく (小刻みな震えを防ぐ)
        }

        setTargetLabel(label) { this.targetLabel = label; }
        isTrained() { return head.trained; }

        /**
         * @param {HTMLCanvasElement} source 224x224 のフレーム
         */
        async predict(source) {
            if (!this.isActive || !extractor.ready || !head.trained || !source) return [];

            const probT = tf.tidy(() => {
                const f = extractor.extract(source);
                return tf.sigmoid(head.logits(f)).reshape([GRID * GRID]);
            });
            const prob = await probT.data();
            probT.dispose();

            const now = performance.now();
            const dt = this.lastTime ? Math.min(0.5, (now - this.lastTime) / 1000) : 0;
            this.lastTime = now;

            this._updateTracks(this._findBlobs(prob), dt);
            return this.tracks
                .filter(t => t.hits >= this.confirmHits)
                .map(t => ({ id: t.id, box: { ...t.box }, label: this.targetLabel, conf: t.conf }));
        }

        /** 前フレームの枠と重なるものは同じ物として滑らかに動かす */
        _updateTracks(blobs, dt) {
            const iou = AppUtils.CollisionDetector.calculateIoU;
            // フレームの間隔に関係なく同じ速さで追いつく係数
            const a = dt > 0 ? 1 - Math.exp(-dt / this.smoothTime) : 1;
            const used = new Set();
            for (const t of this.tracks) {
                let best = -1, bestIoU = 0.2;
                blobs.forEach((b, i) => {
                    if (used.has(i)) return;
                    const v = iou(t.box, b);
                    if (v > bestIoU) { bestIoU = v; best = i; }
                });
                if (best >= 0) {
                    const b = blobs[best];
                    used.add(best);
                    ['x', 'y', 'w', 'h'].forEach(k => {
                        const d = b[k] - t.box[k];
                        if (Math.abs(d) >= this.deadZone) t.box[k] += d * a;
                    });
                    t.conf += (b.conf - t.conf) * a;
                    t.hits++;
                    t.miss = 0;
                } else {
                    t.miss++;
                }
            }
            // 確定前に見失ったものはすぐ捨てる / 確定済みはしばらく表示を続ける
            this.tracks = this.tracks.filter(t =>
                t.hits >= this.confirmHits ? t.miss <= this.holdMisses : t.miss === 0);
            blobs.forEach((b, i) => {
                if (!used.has(i) && this.tracks.length < this.maxBoxes) {
                    this.tracks.push({ id: this.nextId++, box: { x: b.x, y: b.y, w: b.w, h: b.h }, conf: b.conf, hits: 1, miss: 0 });
                }
            });
        }

        /**
         * 塊の端を半マス単位で調整する量 (マス単位)。
         * 端のマスが「ぎりぎり対象」なら内側へ、外隣のマスが「対象に近い」なら外側へ。
         * しきい値をまたいでマスが増減しても、枠の端が1マス分いきなり飛ばないようにする。
         */
        _edgeOffset(pIn, pOut) {
            const th = this.confidenceThreshold, lo = this.edgeLow;
            const clamp01 = (v) => Math.max(0, Math.min(1, v));
            const shrink = 0.5 * (1 - clamp01((pIn - th) / (1 - th)));
            const grow = 0.5 * clamp01((pOut - lo) / (th - lo));
            return grow - shrink;
        }

        /** しきい値以上のマスの塊をすべて枠にする (ノイズ的な1マスは除外) */
        _findBlobs(prob) {
            const th = this.confidenceThreshold;
            const seen = new Uint8Array(GRID * GRID);
            const blobs = [];
            const P = (r, c) => (r < 0 || c < 0 || r >= GRID || c >= GRID) ? 0 : prob[r * GRID + c];
            for (let i = 0; i < GRID * GRID; i++) {
                if (seen[i] || prob[i] < th) continue;
                const stack = [i];
                const members = [];
                seen[i] = 1;
                let minR = GRID, maxR = -1, minC = GRID, maxC = -1, score = 0, maxP = 0;
                while (stack.length) {
                    const k = stack.pop();
                    members.push(k);
                    const r = Math.floor(k / GRID), c = k % GRID;
                    minR = Math.min(minR, r); maxR = Math.max(maxR, r);
                    minC = Math.min(minC, c); maxC = Math.max(maxC, c);
                    score += prob[k]; maxP = Math.max(maxP, prob[k]);
                    [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]].forEach(([rr, cc]) => {
                        if (rr < 0 || cc < 0 || rr >= GRID || cc >= GRID) return;
                        const n = rr * GRID + cc;
                        if (!seen[n] && prob[n] >= th) { seen[n] = 1; stack.push(n); }
                    });
                }
                if (members.length < 2 && maxP < 0.85) continue;

                // 4辺それぞれについて、端のマスとその外隣のマスの確率 (最大値) を調べる
                const e = { L: [0, 0], R: [0, 0], T: [0, 0], B: [0, 0] };
                for (const k of members) {
                    const r = Math.floor(k / GRID), c = k % GRID, p = prob[k];
                    if (c === minC) { e.L[0] = Math.max(e.L[0], p); e.L[1] = Math.max(e.L[1], P(r, c - 1)); }
                    if (c === maxC) { e.R[0] = Math.max(e.R[0], p); e.R[1] = Math.max(e.R[1], P(r, c + 1)); }
                    if (r === minR) { e.T[0] = Math.max(e.T[0], p); e.T[1] = Math.max(e.T[1], P(r - 1, c)); }
                    if (r === maxR) { e.B[0] = Math.max(e.B[0], p); e.B[1] = Math.max(e.B[1], P(r + 1, c)); }
                }
                const left = Math.max(0, minC - this._edgeOffset(...e.L));
                const right = Math.min(GRID, maxC + 1 + this._edgeOffset(...e.R));
                const top = Math.max(0, minR - this._edgeOffset(...e.T));
                const bottom = Math.min(GRID, maxR + 1 + this._edgeOffset(...e.B));

                blobs.push({
                    x: left / GRID, y: top / GRID,
                    w: Math.max(0.5, right - left) / GRID, h: Math.max(0.5, bottom - top) / GRID,
                    conf: maxP, score
                });
            }
            return blobs.sort((a, b) => b.score - a.score).slice(0, this.maxBoxes);
        }

        setMode(active) {
            this.isActive = active;
            if (!active) { this.tracks = []; this.lastTime = 0; }
        }
    }

    window.AppEngine = {
        FeatureExtractor: extractor,
        Head: head,
        DataManager: new DataManager(),
        TrainingEngine: new TrainingEngine(),
        InferenceEngine: new InferenceEngine(),
        INPUT_SIZE: INPUT
    };
})();
