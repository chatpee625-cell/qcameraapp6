/**
 * @file ux-helper.js
 * @description Web Audio APIによる効果音生成、検出枠・Lossグラフの描画
 */

(function() {
    const COLORS = {
        PRIMARY: '#4c9be8',   // 検出枠・学習Loss
        WARNING: '#d9a23a',   // 検証Loss・過学習の兆候
        SUCCESS: '#3fbf7f',
        OK: '#3fbf7f',        // 測定: 許容値未満
        NG: '#e5534b',        // 測定: 許容値以上
        MUTED: '#969ca5',     // 測定できなかった枠
        NEUTRAL: '#e4e6e9',
        LABEL_BG: 'rgba(14, 16, 19, 0.85)'
    };

    class SfxEngine {
        constructor() {
            this.ctx = null;
            this.enabled = false; // 効果音のオン/オフ (メニューで切替。初期はオフ)
        }

        _init() {
            if (!this.ctx) {
                const AudioCtx = window.AudioContext || window.webkitAudioContext;
                if (AudioCtx) {
                    this.ctx = new AudioCtx();
                }
            }
            if (this.ctx && this.ctx.state === 'suspended') {
                this.ctx.resume();
            }
        }

        _playTone(freq, type, duration, volume, slideFreq = null) {
            if (!this.enabled) return;
            try {
                this._init();
                if (!this.ctx) return;

                const osc = this.ctx.createOscillator();
                const gain = this.ctx.createGain();
                osc.type = type;
                osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
                if (slideFreq) {
                    osc.frequency.exponentialRampToValueAtTime(slideFreq, this.ctx.currentTime + duration);
                }
                gain.gain.setValueAtTime(volume, this.ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);
                osc.connect(gain);
                gain.connect(this.ctx.destination);
                osc.start();
                osc.stop(this.ctx.currentTime + duration);
            } catch (e) {
                console.warn("[SfxEngine] Play error", e);
            }
        }

        playAnnotate() { this._playTone(880, 'sine', 0.08, 0.15); }
        playLearnStart() { this._playTone(220, 'triangle', 0.4, 0.2, 660); }
        playLearnSuccess() {
            this._playTone(523.25, 'sine', 0.15, 0.2);
            setTimeout(() => this._playTone(659.25, 'sine', 0.15, 0.2), 100);
            setTimeout(() => this._playTone(783.99, 'sine', 0.25, 0.2), 200);
        }
        playOverfitWarning() {
            this._playTone(150, 'sawtooth', 0.3, 0.25);
            this._playTone(140, 'sawtooth', 0.3, 0.25);
        }
    }

    class VisualEngine {
        constructor(canvasId) {
            this.canvas = document.getElementById(canvasId);
            this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        }

        /** キャンバスの画素数 / 表示上の大きさ (高解像度画面で線を細くくっきり描くため) */
        _scale() {
            if (!this.canvas || !this.canvas.clientWidth) return 1;
            return this.canvas.width / this.canvas.clientWidth;
        }

        /**
         * 検出枠: 細い実線 + 控えめなラベル
         * @param box キャンバスの画素単位の {x, y, w, h}
         * @param opts { color, text, labelBg, labelFg } 測定モードで色と文字を変えるとき
         */
        drawBBox(box, label, confidence, opts = {}) {
            if (!this.ctx) return;
            const ctx = this.ctx;
            const s = this._scale();
            const { x, y, w, h } = box;
            const color = opts.color || COLORS.PRIMARY;

            ctx.save();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2 * s;
            ctx.lineJoin = 'miter';
            ctx.strokeRect(x, y, w, h);

            // ラベル (枠の左上、画面外に出る場合は枠の内側)
            const text = opts.text || `${label}  ${Math.round(confidence * 100)}%`;
            const fontPx = 12 * s;
            ctx.font = `500 ${fontPx}px ui-monospace, Menlo, Consolas, monospace`;
            const padX = 6 * s;
            const boxH = 20 * s;
            const textW = ctx.measureText(text).width;
            const ly = y - boxH >= 0 ? y - boxH : y;

            ctx.fillStyle = opts.labelBg || COLORS.LABEL_BG;
            ctx.fillRect(x - ctx.lineWidth / 2, ly, textW + padX * 2, boxH);
            ctx.fillStyle = opts.labelFg || COLORS.NEUTRAL;
            ctx.textBaseline = 'middle';
            ctx.fillText(text, x - ctx.lineWidth / 2 + padX, ly + boxH / 2);
            ctx.restore();
        }

        /** 測った範囲 (向きに合わせた長方形) を細い線で描く。points はキャンバスの画素単位 */
        drawPolygon(points, color) {
            if (!this.ctx || !points || points.length < 3) return;
            const ctx = this.ctx, s = this._scale();
            ctx.save();
            ctx.strokeStyle = color || COLORS.NEUTRAL;
            ctx.globalAlpha = 0.85;
            ctx.lineWidth = 1 * s;
            ctx.setLineDash([4 * s, 3 * s]);
            ctx.beginPath();
            points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
            ctx.closePath();
            ctx.stroke();
            ctx.restore();
        }

        /** 十円玉の印 (円 + 「10円」のラベル)。キャンバスの画素単位 */
        drawCoinMarker(cx, cy, r) {
            if (!this.ctx) return;
            const ctx = this.ctx, s = this._scale();
            ctx.save();
            ctx.strokeStyle = COLORS.NEUTRAL;
            ctx.lineWidth = 1.5 * s;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.stroke();
            // 中心の十字
            const t = 5 * s;
            ctx.beginPath();
            ctx.moveTo(cx - t, cy); ctx.lineTo(cx + t, cy);
            ctx.moveTo(cx, cy - t); ctx.lineTo(cx, cy + t);
            ctx.stroke();

            const text = '10円 基準';
            ctx.font = `500 ${11 * s}px ui-monospace, Menlo, Consolas, monospace`;
            const tw = ctx.measureText(text).width, bh = 18 * s, pad = 5 * s;
            const lx = cx - tw / 2 - pad, ly = Math.max(0, cy - r - bh - 4 * s);
            ctx.fillStyle = COLORS.LABEL_BG;
            ctx.fillRect(lx, ly, tw + pad * 2, bh);
            ctx.fillStyle = COLORS.NEUTRAL;
            ctx.textBaseline = 'middle';
            ctx.fillText(text, lx + pad, ly + bh / 2);
            ctx.restore();
        }

        drawLossGraph(trainHistory, valHistory, canvasElement) {
            if (!canvasElement) return;
            // 表示サイズに合わせて画素数を調整 (ぼやけ防止)
            const dpr = window.devicePixelRatio || 1;
            const cw = Math.round(canvasElement.clientWidth * dpr);
            const ch = Math.round(canvasElement.clientHeight * dpr);
            if (cw && ch && (canvasElement.width !== cw || canvasElement.height !== ch)) {
                canvasElement.width = cw;
                canvasElement.height = ch;
            }
            const ctx = canvasElement.getContext('2d');
            const w = canvasElement.width;
            const h = canvasElement.height;
            ctx.clearRect(0, 0, w, h);

            // 目盛り線 (横3本)
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
            ctx.lineWidth = 1;
            for (let i = 1; i <= 3; i++) {
                const gy = Math.round((h * i) / 4) + 0.5;
                ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
            }

            if (!trainHistory || trainHistory.length < 2) return;

            const allVals = [...trainHistory, ...(valHistory || [])];
            const maxLoss = Math.max(...allVals, 0.1);
            const pad = 4 * dpr;

            const drawPath = (data, color) => {
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5 * dpr;
                ctx.lineJoin = 'round';
                const step = w / (data.length - 1);
                data.forEach((val, i) => {
                    const lx = i * step;
                    const ly = h - (val / maxLoss) * (h - pad * 2) - pad;
                    if (i === 0) ctx.moveTo(lx, ly);
                    else ctx.lineTo(lx, ly);
                });
                ctx.stroke();
            };

            drawPath(trainHistory, COLORS.PRIMARY);
            if (valHistory && valHistory.length > 0) {
                drawPath(valHistory, COLORS.WARNING);
            }
        }

        /** 過学習の兆候: 学習パネルの枠線と検証Lossの色で知らせる (画面全体は光らせない) */
        applyOverfitPulse() {
            const panel = document.getElementById('training-dashboard');
            if (panel) panel.classList.add('is-overfit');
        }

        resetOverfitVisual() {
            const panel = document.getElementById('training-dashboard');
            if (panel) panel.classList.remove('is-overfit');
        }
    }

    window.DetectionUXHelper = {
        Sfx: new SfxEngine(),
        Visual: (id) => new VisualEngine(id),
        Colors: COLORS
    };
})();