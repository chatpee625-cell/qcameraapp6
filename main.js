/**
 * @file main.js
 * @description 全体オーケストレーション、イベント接続およびUI状態更新
 * v4: TensorFlow.js による実学習・実推論に対応
 *   - 画面に見えている範囲をそのまま 224x224 に切り出して学習/推論 (座標ズレ防止)
 *   - 枠を描き始めた瞬間に映像を一時停止し、そのフレームを学習データにする
 *   - 何枚でも撮影可能 (枠を描く→CAPTURE を繰り返す)
 */

(function() {
    const Orchestrator = {
        state: {
            mode: 'READY', // READY, DETECT, TRAIN
            isDrawing: false,
            currentBoxes: [],   // 確定した枠 (1枚の写真に複数OK)
            drawingBox: null,   // 指で描いている途中の枠
            startPos: null,
            datasetCount: 0,
            pendingFrame: null,
            lastDetectCount: 0
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
            this.dom = {
                viewport: document.getElementById('viewport-container'),
                video: document.getElementById('camera-stream'),
                canvas: document.getElementById('detection-canvas'),
                statusDot: document.getElementById('status-dot'),
                statusText: document.getElementById('status-text'),
                memUsage: document.getElementById('memory-usage'),
                trainPanel: document.getElementById('training-dashboard'),
                progressBar: document.getElementById('training-progress-bar'),
                trainLoss: document.getElementById('val-train-loss'),
                valLoss: document.getElementById('val-val-loss'),
                epochVal: document.getElementById('val-epoch'),
                lossChart: document.getElementById('loss-chart-canvas'),
                btnDetect: document.getElementById('btn-mode-detect'),
                btnTrain: document.getElementById('btn-mode-train'),
                btnAction: document.getElementById('btn-action-main'),
                annoTools: document.getElementById('annotation-tools'),
                btnClear: document.getElementById('btn-tool-clear'),
                btnBg: document.getElementById('btn-tool-bbox'),
                btnSave: document.getElementById('btn-tool-save'),
                btnLoad: document.getElementById('btn-tool-load'),
                fileLoad: document.getElementById('file-load')
            };
        },

        resizeCanvas() {
            if (this.dom.canvas && this.dom.viewport) {
                const rect = this.dom.viewport.getBoundingClientRect();
                this.dom.canvas.width = rect.width;
                this.dom.canvas.height = rect.height;
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
            this.dom.btnAction.onclick = () => this.handleMainAction();
            this.dom.btnClear.onclick = () => this.clearDataset();
            this.dom.btnBg.onclick = () => this.captureBackground();
            this.dom.btnSave.onclick = () => this.exportModel();
            this.dom.btnLoad.onclick = () => this.dom.fileLoad.click();
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

        setMode(mode) {
            if (AppEngine.TrainingEngine.isTraining) return;
            this.state.mode = mode;
            this.dom.btnDetect.className = mode === 'DETECT' ? 'btn btn-primary' : 'btn btn-secondary';
            this.dom.btnTrain.className = mode === 'TRAIN' ? 'btn btn-primary' : 'btn btn-secondary';
            this.dom.btnAction.onclick = () => this.handleMainAction();
            this.visual.resetOverfitVisual();
            this.resumeVideo();

            if (mode === 'DETECT') {
                this.dom.trainPanel.style.display = 'none';
                this.dom.annoTools.style.display = 'none';
                this.dom.btnAction.innerText = 'ピン';
                if (!AppEngine.InferenceEngine.isTrained()) {
                    this.updateStatus('WARN', '未学習（先に学習を）');
                } else {
                    this.updateStatus('OK', '検出中');
                }
                AppEngine.InferenceEngine.setMode(true);
                this.state.lastDetectCount = 0;
                this.runInferenceLoop();
            } else if (mode === 'TRAIN') {
                this.updateStatus('WARN', '対象を指で囲む');
                this.dom.annoTools.style.display = 'flex';
                AppEngine.InferenceEngine.setMode(false);
                this.state.currentBoxes = [];
                this.state.pendingFrame = null;
                this.clearCanvas();
                this.updateActionLabel();
            }
        },

        updateActionLabel() {
            if (this.state.mode !== 'TRAIN') return;
            const n = this.state.datasetCount;
            if (this.state.currentBoxes.length > 0) {
                this.dom.btnAction.innerText = `撮影 (${n}) 枠${this.state.currentBoxes.length}`;
            } else if (n >= 2) {
                this.dom.btnAction.innerText = `学習開始 (${n}枚)`;
            } else {
                this.dom.btnAction.innerText = `撮影 (${n})`;
            }
        },

        updateStatus(type, text) {
            if (this.dom.statusDot) this.dom.statusDot.className = `status-dot ${type.toLowerCase()}`;
            if (this.dom.statusText) this.dom.statusText.innerText = text;
        },

        handleMainAction() {
            if (this.state.mode === 'TRAIN') {
                if (this.state.currentBoxes.length > 0) {
                    this.captureSample();
                } else if (this.state.datasetCount >= 2) {
                    this.startTraining();
                }
            } else if (this.state.mode === 'DETECT') {
                this.sfx.playDetectionPing();
            } else {
                this.setMode('TRAIN');
            }
        },

        clearDataset() {
            AppEngine.DataManager.clear();
            this.state.datasetCount = 0;
            this.state.currentBoxes = [];
            this.state.pendingFrame = null;
            this.clearCanvas();
            this.resumeVideo();
            this.sfx.playAnnotate();
            this.updateActionLabel();
        },

        pauseVideo() {
            try { this.dom.video.pause(); } catch (e) { /* noop */ }
        },

        resumeVideo() {
            const p = this.dom.video.play();
            if (p && p.catch) p.catch(() => {});
        },

        captureSample() {
            if (this.state.currentBoxes.length === 0) return;
            let frame = this.state.pendingFrame;
            if (!frame) {
                frame = document.createElement('canvas');
                frame.width = frame.height = AppEngine.INPUT_SIZE;
                if (!this.grabFrame(frame)) return;
            }

            this.sfx.playAnnotate();
            AppEngine.DataManager.addSample(frame, "Target", this.state.currentBoxes);
            this.state.datasetCount = AppEngine.DataManager.size();

            // 次の1枚へ
            this.state.pendingFrame = null;
            this.state.currentBoxes = [];
            this.clearCanvas();
            this.resumeVideo();
            this.showCounts();
            this.updateActionLabel();
        },

        /**
         * 対象が写っていない「背景だけ」の写真を保存 (誤検出を減らす)
         */
        captureBackground() {
            if (this.state.mode !== 'TRAIN' || AppEngine.TrainingEngine.isTraining) return;
            this.resumeVideo();
            const frame = document.createElement('canvas');
            frame.width = frame.height = AppEngine.INPUT_SIZE;
            if (!this.grabFrame(frame)) return;
            this.sfx.playAnnotate();
            AppEngine.DataManager.addSample(frame, "Background", []);
            this.state.datasetCount = AppEngine.DataManager.size();
            this.state.pendingFrame = null;
            this.state.currentBoxes = [];
            this.clearCanvas();
            this.showCounts();
            this.updateActionLabel();
        },

        /** 学習結果をファイルに書き出す (URLが変わっても読み込めば復元できる) */
        exportModel() {
            if (!AppEngine.InferenceEngine.isTrained()) {
                this.updateStatus('WARN', '保存する学習結果がありません');
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
            this.updateStatus('OK', '学習結果を保存しました');
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
                this.updateStatus('OK', '学習結果を読み込みました');
            } catch (e) {
                console.error('[Orchestrator] import failed', e);
                this.updateStatus('CRIT', `読込エラー：${e.message}`);
            } finally {
                this.dom.fileLoad.value = '';
            }
        },

        showCounts() {
            const c = AppEngine.DataManager.counts();
            this.updateStatus('WARN', `対象 ${c.obj}枚 / 背景 ${c.bg}枚`);
        },

        async startTraining() {
            this.dom.trainPanel.style.display = 'grid';
            this.sfx.playLearnStart();
            this.updateStatus('WARN', '学習中…');
            this.dom.btnAction.disabled = true;
            this.dom.btnDetect.disabled = true;
            this.dom.btnTrain.disabled = true;

            const dataset = AppEngine.DataManager.getDataset();
            const result = await AppEngine.TrainingEngine.runTrainingLoop(dataset, (data) => {
                this.updateTrainingUI(data);
            });

            this.dom.btnAction.disabled = false;
            this.dom.btnDetect.disabled = false;
            this.dom.btnTrain.disabled = false;
            this.visual.resetOverfitVisual();

            if (result.success) {
                this.updateStatus('OK', '学習完了');
                this.sfx.playLearnSuccess();
                this.dom.btnAction.innerText = '検出する';
                this.dom.btnAction.onclick = () => {
                    this.dom.btnAction.onclick = () => this.handleMainAction();
                    this.setMode('DETECT');
                };
            } else {
                this.updateStatus('CRIT', `学習エラー：${result.error || ''}`);
                this.dom.statusText.title = result.error || '';
                this.updateActionLabel();
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
                // 新たに見つけた瞬間だけパーティクル演出
                if (results.length > this.state.lastDetectCount) {
                    this.visual.spawnDetectionParticles(screenBox.x, screenBox.y, screenBox.w, screenBox.h);
                }
            });
            this.state.lastDetectCount = results.length;

            this.visual.updateParticles();
            requestAnimationFrame(() => this.runInferenceLoop());
        },

        startAnno(e) {
            if (this.state.mode !== 'TRAIN' || AppEngine.TrainingEngine.isTraining) return;
            // 描き始めた瞬間のフレームを保存し、映像を止める
            if (!this.state.pendingFrame) {
                const f = document.createElement('canvas');
                f.width = f.height = AppEngine.INPUT_SIZE;
                if (this.grabFrame(f)) this.state.pendingFrame = f;
                this.pauseVideo();
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
            if (!b || b.w < 0.03 || b.h < 0.03) {
                // 短くタップ = 描いた枠を全部消す
                this.state.currentBoxes = [];
            } else {
                this.state.currentBoxes.push(b);
            }
            this.renderCurrentBBox();
            this.updateActionLabel();
        },

        renderCurrentBBox() {
            this.clearCanvas();
            const boxes = [...this.state.currentBoxes];
            if (this.state.drawingBox) boxes.push(this.state.drawingBox);
            if (boxes.length === 0) return;

            const w = this.dom.canvas.width;
            const h = this.dom.canvas.height;
            const ctx = this.dom.canvas.getContext('2d');
            ctx.save();
            ctx.strokeStyle = '#00ffcc';
            ctx.lineWidth = 2;
            ctx.fillStyle = 'rgba(0, 255, 204, 0.15)';
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
