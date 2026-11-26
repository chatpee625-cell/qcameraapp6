/**
 * @file js/ux-helper.js
 * @description Web Audio APIによる効果音生成、Canvasパーティクル・Lossグラフ描画演出
 */

(function() {
    const COLORS = {
        PRIMARY: '#00ffcc',
        WARNING: '#ff3366',
        SUCCESS: '#33ff77',
        NEUTRAL: '#ffffff',
        OVERFIT: '#ffaa00'
    };

    class SfxEngine {
        constructor() {
            this.ctx = null;
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
        playDetectionPing() { this._playTone(1200, 'sine', 0.05, 0.1); }
    }

    class VisualEngine {
        constructor(canvasId) {
            this.canvas = document.getElementById(canvasId);
            this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
            this.particles = [];
            this.isOverfitting = false;
        }

        drawBBox(box, label, confidence) {
            if (!this.ctx) return;
            const { x, y, w, h } = box;
            const ctx = this.ctx;
            const strokeColor = this.isOverfitting ? COLORS.OVERFIT : COLORS.PRIMARY;

            ctx.save();
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 3;
            if (this.isOverfitting) {
                ctx.setLineDash([6, 6]);
            }

            // コーナー強調表示
            const len = Math.min(20, w / 4, h / 4);
            ctx.beginPath();
            ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
            ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
            ctx.moveTo(x, y + h - len); ctx.lineTo(x, y + h); ctx.lineTo(x + len, y + h);
            ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len);
            ctx.stroke();

            // 枠線
            ctx.globalAlpha = 0.4;
            ctx.strokeRect(x, y, w, h);
            ctx.globalAlpha = 1.0;

            // ラベル背景・テキスト
            const text = `${label} ${(confidence * 100).toFixed(0)}%`;
            ctx.font = 'bold 13px sans-serif';
            const textWidth = ctx.measureText(text).width;
            
            ctx.fillStyle = strokeColor;
            ctx.fillRect(x, Math.max(0, y - 22), textWidth + 10, 22);

            ctx.fillStyle = '#000000';
            ctx.fillText(text, x + 5, Math.max(15, y - 6));
            ctx.restore();
        }

        spawnDetectionParticles(x, y, w, h) {
            for (let i = 0; i < 8; i++) {
                this.particles.push({
                    x: x + Math.random() * w,
                    y: y + Math.random() * h,
                    vx: (Math.random() - 0.5) * 3,
                    vy: (Math.random() - 0.5) * 3,
                    life: 1.0,
                    color: COLORS.PRIMARY
                });
            }
        }

        drawLossGraph(trainHistory, valHistory, canvasElement) {
            if (!canvasElement) return;
            const ctx = canvasElement.getContext('2d');
            const w = canvasElement.width;
            const h = canvasElement.height;
            ctx.clearRect(0, 0, w, h);

            if (!trainHistory || trainHistory.length < 2) return;

            const allVals = [...trainHistory, ...(valHistory || [])];
            const maxLoss = Math.max(...allVals, 0.1);

            const drawPath = (data, color) => {
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                const step = w / (data.length - 1);
                data.forEach((val, i) => {
                    const lx = i * step;
                    const ly = h - (val / maxLoss) * (h - 10) - 5;
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

        applyOverfitPulse() {
            this.isOverfitting = true;
            const overlay = document.getElementById('ui-overlay');
            if (overlay) {
                overlay.style.transition = 'background 0.3s';
                overlay.style.background = `rgba(255, 170, 0, 0.15)`;
            }
        }

        resetOverfitVisual() {
            this.isOverfitting = false;
            const overlay = document.getElementById('ui-overlay');
            if (overlay) {
                overlay.style.background = 'none';
            }
        }

        updateParticles() {
            if (!this.ctx || this.particles.length === 0) return;
            this.ctx.save();
            for (let i = this.particles.length - 1; i >= 0; i--) {
                const p = this.particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.life -= 0.03;
                if (p.life <= 0) {
                    this.particles.splice(i, 1);
                    continue;
                }
                this.ctx.globalAlpha = p.life;
                this.ctx.fillStyle = p.color;
                this.ctx.fillRect(p.x, p.y, 3, 3);
            }
            this.ctx.restore();
        }
    }

    window.DetectionUXHelper = {
        Sfx: new SfxEngine(),
        Visual: (id) => new VisualEngine(id),
        Colors: COLORS
    };
})();