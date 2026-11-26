/**
 * @file js/utils.js
 * @description 座標変換、データ拡張、永続化、IoU計算基盤ユーティリティ
 */

(function() {
    /**
     * StorageManager: localStorageの安全なアクセスラッパー
     */
    const StorageManager = (() => {
        const _safeSet = (key, value) => {
            try {
                const serialized = JSON.stringify(value);
                localStorage.setItem(key, serialized);
                return true;
            } catch (e) {
                console.error(`[StorageManager] Save Error: ${key}`, e);
                return false;
            }
        };

        const _safeGet = (key, defaultValue) => {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (e) {
                console.error(`[StorageManager] Load Error: ${key}`, e);
                return defaultValue;
            }
        };

        return {
            saveConfig: (config) => _safeSet('yolo_hyperparams', config),
            getConfig: () => _safeGet('yolo_hyperparams', {
                learningRate: 0.001,
                epochs: 30,
                batchSize: 1,
                augmentationStrength: 0.4
            }),
            saveTrainingState: (state) => _safeSet('yolo_training_state', state),
            getTrainingState: () => _safeGet('yolo_training_state', { epoch: 0, bestLoss: Infinity, history: [] }),
            clearAll: () => {
                try {
                    localStorage.removeItem('yolo_hyperparams');
                    localStorage.removeItem('yolo_training_state');
                } catch(e) {
                    console.error("[StorageManager] Clear Error", e);
                }
            }
        };
    })();

    /**
     * CoordinateTransformer: BBox座標変換・正規化ユーティリティ
     */
    const CoordinateTransformer = (() => {
        const calculateLetterbox = (srcW, srcH, targetW, targetH) => {
            const scale = Math.min(targetW / srcW, targetH / srcH);
            const newW = srcW * scale;
            const newH = srcH * scale;
            const padLeft = (targetW - newW) / 2;
            const padTop = (targetH - newH) / 2;
            return { scale, padLeft, padTop, newW, newH };
        };

        const denormalize = (bbox, layout) => {
            const { canvasW, canvasH, padLeft = 0, padTop = 0 } = layout;
            const usableW = canvasW - (2 * padLeft);
            const usableH = canvasH - (2 * padTop);
            return {
                x: (bbox.x * usableW) + padLeft,
                y: (bbox.y * usableH) + padTop,
                w: bbox.w * usableW,
                h: bbox.h * usableH
            };
        };

        const clamp = (val, min = 0, max = 1) => Math.max(min, Math.min(max, val));

        return { calculateLetterbox, denormalize, clamp };
    })();

    /**
     * AugmentationEngine: 過学習防止のためのデータ拡張ジェネレータ
     */
    const AugmentationEngine = (() => {
        const getFlipTransform = () => Math.random() > 0.5;
        const getBrightnessFactor = (strength) => {
            const delta = (Math.random() * 2 - 1) * strength;
            return 1.0 + delta;
        };
        const getRotationAngle = (maxAngle) => (Math.random() * 2 - 1) * maxAngle;
        
        return { getFlipTransform, getBrightnessFactor, getRotationAngle };
    })();

    /**
     * CollisionDetector: BBox交差評価（IoU）
     */
    const CollisionDetector = (() => {
        const calculateIoU = (boxA, boxB) => {
            const xA = Math.max(boxA.x, boxB.x);
            const yA = Math.max(boxA.y, boxB.y);
            const xB = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
            const yB = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);
            const interW = Math.max(0, xB - xA);
            const interH = Math.max(0, yB - yA);
            const interArea = interW * interH;
            const areaA = boxA.w * boxA.h;
            const areaB = boxB.w * boxB.h;
            const unionArea = areaA + areaB - interArea;
            return unionArea <= 0 ? 0 : interArea / unionArea;
        };
        return { calculateIoU };
    })();

    /**
     * Formatter: テキスト表示フォーマット用
     */
    const Formatter = (() => {
        const formatLoss = (loss) => typeof loss === 'number' ? loss.toFixed(4) : '0.0000';
        const formatPercent = (val) => `${(val * 100).toFixed(1)}%`;
        const formatTime = (ms) => {
            const seconds = Math.floor(ms / 1000);
            const remaining = Math.floor((ms % 1000) / 100);
            return `${seconds}.${remaining}s`;
        };
        return { formatLoss, formatPercent, formatTime };
    })();

    window.AppUtils = {
        StorageManager,
        CoordinateTransformer,
        AugmentationEngine,
        CollisionDetector,
        Formatter
    };
})();