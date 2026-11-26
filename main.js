/**
 * @file main.js
 * @description 全体オーケストレーション、イベント接続およびUI状態更新
 *   - 画面に見えている範囲をそのまま 224x224 に切り出して学習/推論 (座標ズレ防止)
 *   - 枠を描き始めた瞬間に映像を一時停止し、そのフレームを学習データにする
 *   - 何枚でも撮影可能 (枠を描く→保存 を繰り返す)
 */

(function() {
    const SOUND_KEY = 'app_sound_enabled';

    const Orchestrator = {
        state: {
            mode: 'READY', // READY, DETECT, TRAIN
            isDrawing: false,
            currentBoxes: [],   // 確定した枠 (1枚の写真に複数OK)
            drawingBox: null,   // 指で描いている途中の枠
            startPos: null,
            pendingFrame: null  // 枠を描き始めたときに止めたフレーム
        },

        dom: {},

        async init() {
            try {
                console.log("[Orchestrator] Starting initialization...");

                if (!window.tf) {
                    this.cacheDOM();
                    this.updateStatus('CRIT', 'TF読込失敗（通信?）');
                    return;
                }
                if (!window.AppUtils || !window.DetectionUXHelper || !window.AppEngine) {
                    throw new Error("Required modules (Utils/UX/Engine) not found.");
                }

                this.cacheDOM();
                this.visual = DetectionUXHelper.Visual('detection-canvas');
                this.sfx = DetectionUXHelper.Sfx;
                this.loadSoundSetting();

                const size = AppEngine.INPUT_SIZE;
                this.inferCanvas = document.createElement('canvas');
                this.inferCanvas.width = size;
                this.inferCanvas.height = size;

                this.resizeCanvas();
                this.setupEventListeners();

                await this.initCamera();
                this.startResourceMonitor();
                window.addEventListener('resize', () => this.resizeCanvas());

                // モデル読み込み (初回は数MBダウンロード)
                this.updateStatus('WARN', 'モデル読込中…');
                await AppEngine.FeatureExtractor.load();
                const restored = AppEngine.Head.load();

                this.updateStatus('OK', restored ? '準備OK（学習済み）' : '準備OK');
                console.log("[Orchestrator] System initialized successfully.");

            } catch (error) {
                console.error("[Orchestrator] Initialization failed:", error);
                if (this.dom.statusText) {
                    this.updateStatus('CRIT', '初期化エラー');
                    this.dom.statusText.title = error.message;
                }
            }
        },

        cacheDOM() {
            const $ = (id) => document.getElementById(id);
            this.dom = {
                viewport: $('viewport-container'),
                video: $('camera-stream'),
                canvas: $('detection-canvas'),
                flash: $('capture-flash'),
                statusDot: $('status-dot'),
                statusText: $('status-text'),
                memUsage: $('memory-usage'),
                btnMenu: $('btn-menu'),
                menu: $('menu-panel'),
                btnMenuSave: $('btn-menu-save'),
                btnMenuLoad: $('btn-menu-load'),
                btnMenuSound: $('btn-menu-sound'),
                fileLoad: $('file-load'),
                trainInfo: $('train-info'),
                countObj: $('count-obj'),
                countBg: $('count-bg'),
                trainHint: $('train-hint'),
                btnClear: $('btn-clear-data'),
                trainPanel: $('training-dashboard'),
                progressBar: $('training-progress-bar'),
                trainLoss: $('val-train-loss'),
                valLoss: $('val-val-loss'),
                epochVal: $('val-epoch'),
                lossChart: $('loss-chart-canvas'),
                toast: $('toast'),
                freezeChip: $('freeze-chip'),
                btnDetect: $('btn-mode-detect'),
                btnTrain: $('btn-mode-train'),
                trainActions: $('train-actions'),
                btnSub: $('btn-train-sub'),
                btnMain: $('btn-train-main')
            };
        },

        resizeCanvas() {
            if (this.dom.canvas && this.dom.viewport) {
                // 高解像度画面では画素数を増やし、線をくっきり描く
                const rect = this.dom.viewport.getBoundingClientRect();
                const dpr = window.devicePixelRatio || 1;
                this.dom.canvas.width = Math.round(rect.width * dpr);
                this.dom.canvas.height = Math.round(rect.height * dpr);
                if (this.state.mode === 'TRAIN') this.renderCurrentBBox();
            }
        },

        async initCamera() {
            try {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    this.updateStatus('CRIT', 'カメラ非対応');
                    return;
                }
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: 'environment',
                        width: { ideal: 640 },
                        height: { ideal: 640 }
                    },
                    audio: false
                });
                this.dom.video.srcObject = stream;
            } catch (e) {
                console.error("[Orchestrator] Camera access error:", e);
                this.updateStatus('CRIT', 'カメラエラー');
            }
        },

        /**
         * 画面に見えている範囲 (object-fit: cover の切り抜き) を 224x224 に描画
         */
        grabFrame(dst) {
            const v = this.dom.video;
            const vw = v.videoWidth, vh = v.videoHeight;
            if (!vw || !vh) return false;
            const cw = this.dom.canvas.clientWidth, ch = this.dom.canvas.clientHeight;
            const scale = Math.max(cw / vw, ch / vh);
            const sw = cw / scale, sh = ch / scale;
            const sx = (vw - sw) / 2, sy = (vh - sh) / 2;
            dst.getContext('2d').drawImage(v, sx, sy, sw, sh, 0, 0, dst.width, dst.height);
            return true;
        },

        setupEventListeners() {
            this.dom.btnDetect.onclick = () => this.setMode('DETECT');
            this.dom.btnTrain.onclick = () => this.setMode('TRAIN');
            this.dom.btnSub.onclick = () => this.handleSubAction();
            this.dom.btnMain.onclick = () => this.handleMainAction();
            this.dom.btnClear.onclick = () => this.clearDataset();

            // メニュー
            this.dom.btnMenu.onclick = (e) => { e.stopPropagation(); this.toggleMenu(); };
            this.dom.menu.onclick = (e) => e.stopPropagation();
            document.addEventListener('click', () => this.toggleMenu(false));
            this.dom.btnMenuSave.onclick = () => { this.toggleMenu(false); this.exportModel(); };
            this.dom.btnMenuLoad.onclick = () => { this.toggleMenu(false); this.dom.fileLoad.click(); };
            this.dom.btnMenuSound.onclick = () => this.setSound(!this.sfx.enabled);
            this.dom.fileLoad.onchange = (e) => this.importModel(e.target.files[0]);

            const canvas = this.dom.canvas;
            canvas.addEventListener('mousedown', (e) => this.startAnno(e));
            canvas.addEventListener('mousemove', (e) => this.drawAnno(e));
            canvas.addEventListener('mouseup', () => this.endAnno());

            canvas.addEventListener('touchstart', (e) => {
                if (e.touches.length > 0) this.startAnno(e.touches[0]);
            }, { passive: false });
            canvas.addEventListener('touchmove', (e) => {
                if (e.touches.length > 0) this.drawAnno(e.touches[0]);
                e.preventDefault();
            }, { passive: false });
            canvas.addEventListener('touchend', () => this.endAnno());
        },

        // ---------- メニュー・設定 ----------

        toggleMenu(open) {
            const show = open === undefined ? this.dom.menu.hidden : open;
            this.dom.menu.hidden = !show;
            this.dom.btnMenu.setAttribute('aria-expanded', String(show));
        },

        loadSoundSetting() {
            let on = false; // 初期状態は音なし
            try { on = localStorage.getItem(SOUND_KEY) === '1'; } catch (e) { /* noop */ }
            this.setSound(on, false);
        },

        setSound(on, save = true) {
            this.sfx.enabled = on;
            this.dom.btnMenuSound.setAttribute('aria-pressed', String(on));
            this.dom.btnMenuSound.querySelector('.toggle-state').textContent = on ? 'オン' : 'オフ';
            if (save) {
                try { localStorage.setItem(SOUND_KEY, on ? '1' : '0'); } catch (e) { /* noop */ }
            }
        },

        // ---------- 画面の状態 ----------

        setMode(mode) {
            if (AppEngine.TrainingEngine.isTraining) return;
            this.state.mode = mode;
            this.dom.btnDetect.className = mode === 'DETECT' ? 'btn btn-primary' : 'btn btn-secondary';
            this.dom.btnTrain.className = mode === 'TRAIN' ? 'btn btn-primary' : 'btn btn-secondary';
            this.visual.resetOverfitVisual();
            this.state.currentBoxes = [];
            this.state.drawingBox = null;
            this.state.pendingFrame = null;
            this.resumeVideo();
            this.clearCanvas();

            const isTrain = mode === 'TRAIN';
            this.dom.trainInfo.hidden = !isTrain;
            this.dom.trainActions.hidden = !isTrain;
            this.dom.trainPanel.style.display = 'none';

            if (mode === 'DETECT') {
                if (!AppEngine.InferenceEngine.isTrained()) {
                    this.updateStatus('WARN', '未学習（先に学習を）');
                } else {
                    this.updateStatus('OK', '検出中');
                }
                AppEngine.InferenceEngine.setMode(true);
                this.runInferenceLoop();
            } else if (isTrain) {
                this.updateStatus('OK', '学習モード');
                AppEngine.InferenceEngine.setMode(false);
            }
            this.updateTrainUI();
        },

        /** 枠を描いている途中 (映像が止まっている) か */
        isFrozen() {
            return this.state.mode === 'TRAIN' && !!this.state.pendingFrame;
        },

        canTrain() {
            const c = AppEngine.DataManager.counts();
            return c.obj >= 1 && c.obj + c.bg >= 2;
        },

        /** 学習モードの表示 (枚数・案内・ボタン・停止中の表示) をまとめて更新 */
        updateTrainUI() {
            const frozen = this.isFrozen();
            this.dom.viewport.classList.toggle('is-frozen', frozen);
            this.dom.freezeChip.hidden = !frozen;
            if (this.state.mode !== 'TRAIN') return;

            const c = AppEngine.DataManager.counts();
            const nBox = this.state.currentBoxes.length;
            const training = AppEngine.TrainingEngine.isTraining;
            this.dom.countObj.textContent = c.obj;
            this.dom.countBg.textContent = c.bg;
            this.dom.btnClear.disabled = training || c.obj + c.bg === 0;

            let hint;
            if (training) {
                hint = '学習中です。しばらくお待ちください';
            } else if (frozen) {
                this.dom.freezeChip.textContent = nBox > 0 ? `静止中 — 枠 ${nBox}個` : '静止中 — 対象を指で囲んでください';
                hint = nBox > 0
                    ? '他にも対象があれば続けて囲めます。終わったら「この写真を保存」'
                    : '対象を指でなぞって四角く囲んでください';
            } else if (c.obj === 0) {
                hint = '対象を指でなぞって囲むと、映像が止まります';
            } else if (!this.canTrain()) {
                hint = '角度や距離を変えて、もう1枚以上撮ってください';
            } else {
                hint = '撮り足すか、「学習開始」を押してください';
            }
            this.dom.trainHint.textContent = hint;

            if (frozen) {
                this.dom.btnSub.textContent = 'やり直す';
                this.dom.btnMain.textContent = nBox > 0 ? `この写真を保存（枠${nBox}）` : 'この写真を保存';
                this.dom.btnSub.disabled = false;
                this.dom.btnMain.disabled = nBox === 0;
            } else {
                this.dom.btnSub.textContent = '背景を撮影';
                const n = c.obj + c.bg;
                this.dom.btnMain.textContent = training ? '学習中…' : (n > 0 ? `学習開始（${n}枚）` : '学習開始');
                this.dom.btnSub.disabled = training;
                this.dom.btnMain.disabled = training || !this.canTrain();
            }
        },

        updateStatus(type, text) {
            if (this.dom.statusDot) this.dom.statusDot.className = `status-dot ${type.toLowerCase()}`;
            if (this.dom.statusText) this.dom.statusText.innerText = text;
        },

        /** 画面下部に短いお知らせを出す */
        showToast(text) {
            const t = this.dom.toast;
            if (!t) return;
            t.textContent = text;
            t.classList.add('is-visible');
            clearTimeout(this._toastTimer);
            this._toastTimer = setTimeout(() => t.classList.remove('is-visible'), 2200);
        },

        flashCapture() {
            const f = this.dom.flash;
            f.classList.remove('is-active');
            void f.offsetWidth; // アニメーションを最初から再生
            f.classList.add('is-active');
        },

        // ---------- 学習モードの操作 ----------

        handleSubAction() {
            if (this.state.mode !== 'TRAIN') return;
            if (this.isFrozen()) this.cancelFrame();
            else this.captureBackground();
        },

        handleMainAction() {
            if (this.state.mode !== 'TRAIN') return;
            if (this.isFrozen()) this.captureSample();
            else if (this.canTrain()) this.startTraining();
        },

        /** 止めた写真と描いた枠を捨てて、映像を再開する */
        cancelFrame() {
            this.state.pendingFrame = null;
            this.state.currentBoxes = [];
            this.state.drawingBox = null;
            this.clearCanvas();
            this.resumeVideo();
            this.updateTrainUI();
        },

        clearDataset() {
            const n = AppEngine.DataManager.size();
            if (n === 0) return;
            if (!window.confirm(`撮影した ${n}枚 をすべて消去します。よろしいですか？\n（学習済みの結果は消えません）`)) return;
            AppEngine.DataManager.clear();
            this.cancelFrame();
            this.showToast('撮影データを消去しました');
        },

        pauseVideo() {
            try { this.dom.video.pause(); } catch (e) { /* noop */ }
        },

        resumeVideo() {
            const p = this.dom.video.play();
            if (p && p.catch) p.catch(() => {});
        },

        captureSample() {
            if (this.state.currentBoxes.length === 0 || !this.state.pendingFrame) return;
            this.sfx.playAnnotate();
            AppEngine.DataManager.addSample(this.state.pendingFrame, "Target", this.state.currentBoxes);
            this.flashCapture();
            this.showToast(`保存しました（対象 ${AppEngine.DataManager.counts().obj}枚目）`);
            this.cancelFrame(); // 次の1枚へ
        },

        /**
         * 対象が写っていない「背景だけ」の写真を保存 (誤検出を減らす)
         */
        captureBackground() {
            if (this.state.mode !== 'TRAIN' || AppEngine.TrainingEngine.isTraining) return;
            const frame = document.createElement('canvas');
            frame.width = frame.height = AppEngine.INPUT_SIZE;
            if (!this.grabFrame(frame)) return;
            this.sfx.playAnnotate();
            AppEngine.DataManager.addSample(frame, "Background", []);
            this.flashCapture();
            this.showToast(`背景を保存しました（${AppEngine.DataManager.counts().bg}枚目）`);
            this.updateTrainUI();
        },

        /** 学習結果をファイルに書き出す (URLが変わっても読み込めば復元できる) */
        exportModel() {
            if (!AppEngine.InferenceEngine.isTrained()) {
                this.showToast('保存する学習結果がありません');
                return;
            }
            const json = JSON.stringify(AppEngine.Head.toJSON());
            const blob = new Blob([json], { type: 'application/json' });
            const d = new Date();
            const pad = (v) => String(v).padStart(2, '0');
            const name = `yolo-camera_${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}.json`;
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = name;
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(a.href), 1000);
            this.showToast('学習結果をファイルに保存しました');
        },

        /** 書き出したファイルから学習結果を読み込む */
        async importModel(file) {
            if (!file) return;
            try {
                if (!AppEngine.FeatureExtractor.ready) throw new Error('モデル読込中です');
                const obj = JSON.parse(await file.text());
                AppEngine.Head.fromJSON(obj);
                AppEngine.Head.save(); // このURLでも次回から自動で使う
                this.sfx.playLearnSuccess();
                this.showToast('学習結果を読み込みました');
                if (this.state.mode === 'DETECT') this.updateStatus('OK', '検出中');
            } catch (e) {
                console.error('[Orchestrator] import failed', e);
                this.updateStatus('CRIT', `読込エラー：${e.message}`);
            } finally {
                this.dom.fileLoad.value = '';
            }
        },

        async startTraining() {
            this.dom.trainPanel.style.display = 'grid';
            this.dom.progressBar.style.width = '0%';
            this.sfx.playLearnStart();
            this.updateStatus('WARN', '学習中…');
            this.dom.btnDetect.disabled = true;
            this.dom.btnTrain.disabled = true;

            const dataset = AppEngine.DataManager.getDataset();
            const run = AppEngine.TrainingEngine.runTrainingLoop(dataset, (data) => {
                this.updateTrainingUI(data);
            });
            this.updateTrainUI(); // isTraining=true になった状態でボタンを無効化
            const result = await run;

            this.dom.btnDetect.disabled = false;
            this.dom.btnTrain.disabled = false;
            this.visual.resetOverfitVisual();
            this.dom.trainPanel.style.display = 'none';

            if (result.success) {
                this.sfx.playLearnSuccess();
                this.setMode('DETECT');
                this.showToast('学習が完了しました。検出モードに切り替えました');
            } else {
                this.updateStatus('CRIT', `学習エラー：${result.error || ''}`);
                this.dom.statusText.title = result.error || '';
                this.updateTrainUI();
            }
        },

        updateTrainingUI(data) {
            this.dom.progressBar.style.width = `${data.progress * 100}%`;
            this.dom.trainLoss.innerText = AppUtils.Formatter.formatLoss(data.trainLoss);
            this.dom.valLoss.innerText = AppUtils.Formatter.formatLoss(data.valLoss);
            this.dom.epochVal.innerText = `${data.epoch}/${data.maxEpochs}`;

            if (data.valLoss > data.trainLoss * 1.4 && data.epoch > 3) {
                this.visual.applyOverfitPulse();
                this.sfx.playOverfitWarning();
            } else {
                this.visual.resetOverfitVisual();
            }

            this.visual.drawLossGraph(data.trainHistory, data.valHistory, this.dom.lossChart);
        },

        // ---------- 検出モード ----------

        async runInferenceLoop() {
            if (this.state.mode !== 'DETECT') return;

            let results = [];
            try {
                if (this.grabFrame(this.inferCanvas)) {
                    results = await AppEngine.InferenceEngine.predict(this.inferCanvas);
                }
            } catch (e) {
                console.error("[Orchestrator] inference error", e);
            }
            if (this.state.mode !== 'DETECT') return;

            this.clearCanvas();
            const layout = { canvasW: this.dom.canvas.width, canvasH: this.dom.canvas.height };

            results.forEach(res => {
                const screenBox = AppUtils.CoordinateTransformer.denormalize(res.box, layout);
                this.visual.drawBBox(screenBox, res.label, res.conf);
            });

            requestAnimationFrame(() => this.runInferenceLoop());
        },

        // ---------- 枠を描く操作 ----------

        startAnno(e) {
            // メニューが開いているときのタップは「メニューを閉じる」だけにする
            if (!this.dom.menu.hidden) { this.toggleMenu(false); return; }
            if (this.state.mode !== 'TRAIN' || AppEngine.TrainingEngine.isTraining) return;
            // 描き始めた瞬間のフレームを保存し、映像を止める
            if (!this.state.pendingFrame) {
                const f = document.createElement('canvas');
                f.width = f.height = AppEngine.INPUT_SIZE;
                if (!this.grabFrame(f)) return;
                this.state.pendingFrame = f;
                this.pauseVideo();
                this.updateTrainUI();
            }
            const rect = this.dom.canvas.getBoundingClientRect();
            this.state.startPos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            this.state.drawingBox = null;
            this.state.isDrawing = true;
        },

        drawAnno(e) {
            if (!this.state.isDrawing || this.state.mode !== 'TRAIN') return;
            const rect = this.dom.canvas.getBoundingClientRect();
            const curX = e.clientX - rect.left;
            const curY = e.clientY - rect.top;

            const x = Math.min(this.state.startPos.x, curX);
            const y = Math.min(this.state.startPos.y, curY);
            const w = Math.abs(this.state.startPos.x - curX);
            const h = Math.abs(this.state.startPos.y - curY);

            this.state.drawingBox = {
                x: AppUtils.CoordinateTransformer.clamp(x / rect.width),
                y: AppUtils.CoordinateTransformer.clamp(y / rect.height),
                w: AppUtils.CoordinateTransformer.clamp(w / rect.width),
                h: AppUtils.CoordinateTransformer.clamp(h / rect.height)
            };

            this.renderCurrentBBox();
        },

        endAnno() {
            if (!this.state.isDrawing) return;
            this.state.isDrawing = false;
            const b = this.state.drawingBox;
            this.state.drawingBox = null;
            if (b && b.w >= 0.03 && b.h >= 0.03) {
                this.state.currentBoxes.push(b);
            } else if (this.state.currentBoxes.length > 0) {
                // 短くタップ = 描いた枠を全部消す
                this.state.currentBoxes = [];
            }
            this.renderCurrentBBox();
            this.updateTrainUI();
        },

        renderCurrentBBox() {
            this.clearCanvas();
            const boxes = [...this.state.currentBoxes];
            if (this.state.drawingBox) boxes.push(this.state.drawingBox);
            if (boxes.length === 0) return;

            const w = this.dom.canvas.width;
            const h = this.dom.canvas.height;
            const ctx = this.dom.canvas.getContext('2d');
            const s = this.dom.canvas.clientWidth ? w / this.dom.canvas.clientWidth : 1;
            ctx.save();
            ctx.strokeStyle = DetectionUXHelper.Colors.PRIMARY;
            ctx.lineWidth = 2 * s;
            ctx.fillStyle = 'rgba(76, 155, 232, 0.12)';
            boxes.forEach(bbox => {
                ctx.fillRect(bbox.x * w, bbox.y * h, bbox.w * w, bbox.h * h);
                ctx.strokeRect(bbox.x * w, bbox.y * h, bbox.w * w, bbox.h * h);
            });
            ctx.restore();
        },

        clearCanvas() {
            if (!this.dom.canvas) return;
            const ctx = this.dom.canvas.getContext('2d');
            ctx.clearRect(0, 0, this.dom.canvas.width, this.dom.canvas.height);
        },

        startResourceMonitor() {
            setInterval(() => {
                if (!this.dom.memUsage) return;
                let text = 'RAM: --';
                if (window.performance && window.performance.memory) {
                    const mem = Math.round(window.performance.memory.usedJSHeapSize / (1024 * 1024));
                    text = `RAM: ${mem}MB`;
                }
                if (window.tf) {
                    const gpu = Math.round(tf.memory().numBytes / (1024 * 1024));
                    text += ` / TF: ${gpu}MB`;
                }
                this.dom.memUsage.innerText = text;
            }, 2000);
        }
    };

    window.addEventListener('DOMContentLoaded', () => {
        Orchestrator.init();
    });
})();
