/**
 * @file measure.js
 * @description 測定モード: 十円玉を基準に、検出した対象の寸法を測る (window.AppMeasure)
 *
 * 仕組み (AIは使わない、昔ながらの画像処理):
 *  1. 画面に見えている範囲を2つの解像度で取り込む
 *     - 低解像度 (幅 最大360px): 十円玉を探す (軽くするため)
 *     - 高解像度 (幅 最大960px): 十円玉の外周と対象の輪郭を細かく測る
 *  2. 十円玉: 輪郭 (エッジ) から円を探し、銅色かどうかで確認。直径23.5mm を基準にする
 *  3. 対象: AIの枠を一回り広げ、背景の色と比べて物体の画素を取り出す
 *     (影は「背景と同じ色味のまま暗いだけ」なので除く)。内側の穴を埋め、
 *     傾きに合わせた最小の長方形で長さ・幅を測る。丸い物は直径を測る
 *  影除去・最小の長方形・円の当てはめの考え方は VISIONINSPECT (手動測定) と同じ
 */

(function() {
    const COIN_MM = 23.5;        // 十円玉の直径 (mm)
    const LO_MAX_W = 360;        // 十円玉を探す画像の最大幅 (px)
    const HI_MAX_W = 960;        // 寸法を測る画像の最大幅 (px)
    const ROI_MAX_PX = 360000;   // 対象を調べる範囲の最大画素数 (超えたら縮小して測る)
    const EDGE_MIN = 60;         // エッジとみなす最小の強さ

    // ----- 影除去の調整値 (VISIONINSPECT と同じ) -----
    const SHADOW_CHROMA = 0.20;  // 背景と比べた色味のずれがこれを超えたら物体
    const SHADOW_DARK = 0.30;    // 背景に対する明るさがこれ未満なら物体 (黒い物)
    const SHADOW_BRIGHT = 1.30;  // 背景よりこれ以上明るければ物体 (照り返しなど)
    const CIRCLE_INLIER = 0.85;  // 円とみなす条件: 輪郭の点のうち円に乗っている割合

    const options = { shadow: true }; // 影を除くか (メニューで切替)

    // 低解像度 (十円玉探し)
    const work = document.createElement('canvas');
    const wctx = work.getContext('2d', { willReadFrequently: true });
    let W = 0, H = 0;
    let rgba = null, gray = null, gx = null, gy = null, mag = null, acc = null;
    // 高解像度 (寸法)
    const hiCanvas = document.createElement('canvas');
    const hctx = hiCanvas.getContext('2d', { willReadFrequently: true });
    let HW = 0, HH = 0;
    // 大きすぎる範囲を縮小して測るための作業用
    const roiCanvas = document.createElement('canvas');
    const rctx = roiCanvas.getContext('2d', { willReadFrequently: true });
    let captured = false;

    // ---------- 小さな道具 ----------

    function median(arr) {
        if (!arr.length) return 0;
        const s = Array.from(arr).sort((a, b) => a - b);
        const m = s.length >> 1;
        return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    }

    function quantile(sorted, q) {
        if (!sorted.length) return 0;
        const pos = (sorted.length - 1) * q;
        const i = Math.floor(pos), f = pos - i;
        return i + 1 < sorted.length ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i];
    }

    function bilinear(arr, w, h, x, y) {
        const x0 = Math.floor(x), y0 = Math.floor(y);
        if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1) return 0;
        const fx = x - x0, fy = y - y0, i = y0 * w + x0;
        return arr[i] * (1 - fx) * (1 - fy) + arr[i + 1] * fx * (1 - fy) +
               arr[i + w] * (1 - fx) * fy + arr[i + w + 1] * fx * fy;
    }

    /** Sobel フィルタで輪郭の強さと向きを求める */
    function sobel(g, w, h, outX, outY, outM) {
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const i = y * w + x;
                const a = g[i - w - 1], b = g[i - w], c = g[i - w + 1];
                const d = g[i - 1], f = g[i + 1];
                const p = g[i + w - 1], q = g[i + w], k = g[i + w + 1];
                const sx = (c + 2 * f + k) - (a + 2 * d + p);
                const sy = (p + 2 * q + k) - (a + 2 * b + c);
                outX[i] = sx; outY[i] = sy;
                outM[i] = Math.sqrt(sx * sx + sy * sy);
            }
        }
    }

    function toGray(data, n, out) {
        for (let i = 0, j = 0; i < n; i++, j += 4) out[i] = 0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2];
    }

    /** 3元連立方程式 (ガウスの消去法) */
    function solve3(A, b) {
        const M = [[...A[0], b[0]], [...A[1], b[1]], [...A[2], b[2]]];
        for (let c = 0; c < 3; c++) {
            let p = c;
            for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
            if (Math.abs(M[p][c]) < 1e-9) return null;
            [M[c], M[p]] = [M[p], M[c]];
            for (let r = 0; r < 3; r++) {
                if (r === c) continue;
                const f = M[r][c] / M[c][c];
                for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k];
            }
        }
        return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
    }

    // ---------- 1. 取り込み ----------

    /**
     * @param source video または canvas
     * @param crop   画面に見えている範囲 { sx, sy, sw, sh } (source の画素単位)
     */
    function capture(source, crop) {
        // 高解像度: 映像をそのまま (幅 最大 HI_MAX_W) 写す。画素の読み出しは測る範囲だけ後で行う
        const sh = Math.min(1, HI_MAX_W / crop.sw);
        const hw = Math.max(16, Math.round(crop.sw * sh)), hh = Math.max(16, Math.round(crop.sh * sh));
        if (hw !== HW || hh !== HH) { HW = hw; HH = hh; hiCanvas.width = HW; hiCanvas.height = HH; }
        hctx.drawImage(source, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, HW, HH);

        // 低解像度: 十円玉探し用 (高解像度の画像を縮小)
        const s = Math.min(1, LO_MAX_W / HW);
        const w = Math.max(16, Math.round(HW * s)), h = Math.max(16, Math.round(HH * s));
        if (w !== W || h !== H) {
            W = w; H = h;
            work.width = W; work.height = H;
            gray = new Float32Array(W * H);
            gx = new Float32Array(W * H);
            gy = new Float32Array(W * H);
            mag = new Float32Array(W * H);
            acc = new Float32Array((W >> 1) * (H >> 1));
        }
        wctx.drawImage(hiCanvas, 0, 0, HW, HH, 0, 0, W, H);
        rgba = wctx.getImageData(0, 0, W, H).data;
        toGray(rgba, W * H, gray);
        sobel(gray, W, H, gx, gy, mag);
        captured = true;
    }

    /** 画面の中でエッジとみなす強さ (上位8%、最低 EDGE_MIN) */
    function edgeThreshold() {
        const hist = new Uint32Array(512);
        for (let i = 0; i < mag.length; i++) hist[Math.min(511, mag[i] >> 2)]++;
        const target = mag.length * 0.92;
        let sum = 0;
        for (let b = 0; b < 512; b++) {
            sum += hist[b];
            if (sum >= target) return Math.max(EDGE_MIN, b * 4);
        }
        return EDGE_MIN;
    }

    // ---------- 2. 十円玉を探す ----------

    function findCoin() {
        const T = edgeThreshold();
        const hw = W >> 1, hh = H >> 1;
        const rMin = Math.max(6, Math.round(W * 0.03));
        const rMax = Math.round(Math.min(W, H) * 0.35);
        acc.fill(0);

        // (a) 輪郭の向きに沿って「円の中心」に投票する (ハフ変換)
        for (let y = 2; y < H - 2; y += 2) {
            for (let x = 2; x < W - 2; x += 2) {
                const i = y * W + x, m = mag[i];
                if (m < T) continue;
                const ux = gx[i] / m, uy = gy[i] / m;
                for (let r = rMin; r <= rMax; r += 2) {
                    let cx = (x + r * ux) * 0.5, cy = (y + r * uy) * 0.5;
                    if (cx >= 0 && cy >= 0 && cx < hw && cy < hh) acc[(cy | 0) * hw + (cx | 0)]++;
                    cx = (x - r * ux) * 0.5; cy = (y - r * uy) * 0.5;
                    if (cx >= 0 && cy >= 0 && cx < hw && cy < hh) acc[(cy | 0) * hw + (cx | 0)]++;
                }
            }
        }

        // (b) 投票の多い点 (候補) を最大8個
        const cands = [];
        for (let y = 1; y < hh - 1; y++) {
            for (let x = 1; x < hw - 1; x++) {
                const i = y * hw + x;
                const v = acc[i - hw - 1] + acc[i - hw] + acc[i - hw + 1] + acc[i - 1] + acc[i] +
                          acc[i + 1] + acc[i + hw - 1] + acc[i + hw] + acc[i + hw + 1];
                if (v < 12) continue;
                let isMax = true;
                for (let dy = -1; dy <= 1 && isMax; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if ((dx || dy) && acc[i + dy * hw + dx] > acc[i]) { isMax = false; break; }
                    }
                }
                if (isMax) cands.push({ x: x * 2 + 1, y: y * 2 + 1, v });
            }
        }
        cands.sort((a, b) => b.v - a.v);
        const picked = [];
        for (const c of cands) {
            if (picked.every(p => Math.hypot(p.x - c.x, p.y - c.y) > rMin)) picked.push(c);
            if (picked.length >= 8) break;
        }

        // (c) 候補ごとに半径・円周の揃い具合・色を確かめる
        let best = null;
        for (const c of picked) {
            // 半径の候補を外側から順に確かめ、条件を満たす一番外側を十円玉の外周とする
            // (模様の内側の輪や、縁の内側の段を外周と間違えないため)
            for (const r of estimateRadii(c.x, c.y, rMin, rMax, T)) {
                const cov = coverage(c.x, c.y, r, T);
                if (cov < 0.6 || !isCopper(c.x, c.y, r)) continue;
                if (!best || cov > best.cov) best = { x: c.x, y: c.y, r, cov };
                break;
            }
        }
        if (!best) return null;
        const lo = refineCircle(mag, W, H, best.x, best.y, best.r, T);
        if (!lo) return null;
        // 高解像度で外周を測り直す (うまくいかなければ低解像度の結果を拡大して使う)
        const K = HW / W;
        return refineCoinHi(lo, K) || { x: lo.x * K, y: lo.y * K, diameterPx: lo.diameterPx * K, r: lo.r * K };
    }

    /** 高解像度の画像で、十円玉のまわりだけ輪郭を計算し、直径を細かく測る */
    function refineCoinHi(c, K) {
        const cx = c.x * K, cy = c.y * K, R = c.r * K;
        const pad = R * 1.3 + 8;
        const x0 = Math.max(0, Math.floor(cx - pad)), y0 = Math.max(0, Math.floor(cy - pad));
        const x1 = Math.min(HW, Math.ceil(cx + pad)), y1 = Math.min(HH, Math.ceil(cy + pad));
        const w = x1 - x0, h = y1 - y0;
        if (w < 16 || h < 16) return null;
        const data = hctx.getImageData(x0, y0, w, h).data;
        const n = w * h, g = new Float32Array(n), ex = new Float32Array(n), ey = new Float32Array(n), em = new Float32Array(n);
        toGray(data, n, g);
        sobel(g, w, h, ex, ey, em);
        const sorted = Float32Array.from(em).sort();
        const T = Math.max(EDGE_MIN, quantile(sorted, 0.9));
        const r = refineCircle(em, w, h, cx - x0, cy - y0, R, T);
        if (!r) return null;
        // 低解像度の結果と大きく違うときは信用しない
        if (Math.abs(r.diameterPx - c.diameterPx * K) > Math.max(4, c.diameterPx * K * 0.08)) return null;
        return { x: r.x + x0, y: r.y + y0, diameterPx: r.diameterPx, r: r.r };
    }

    /**
     * 中心から見て放射方向を向いた輪郭が多い半径の候補 (外側から順)
     * 傾いた十円玉 (楕円) では、短い半径と長い半径の両方に山ができる
     */
    function estimateRadii(cx, cy, rMin, rMax, T) {
        const hist = new Float32Array(rMax + 3);
        const x0 = Math.max(1, Math.floor(cx - rMax - 2)), x1 = Math.min(W - 2, Math.ceil(cx + rMax + 2));
        const y0 = Math.max(1, Math.floor(cy - rMax - 2)), y1 = Math.min(H - 2, Math.ceil(cy + rMax + 2));
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
                const i = y * W + x, m = mag[i];
                if (m < T) continue;
                const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy);
                if (d < rMin - 1 || d > rMax + 1) continue;
                if (Math.abs(gx[i] * dx + gy[i] * dy) / (m * d) < 0.9) continue;
                hist[Math.round(d)]++;
            }
        }
        let bestScore = 0;
        const score = new Float32Array(hist.length);
        for (let r = rMin; r <= rMax; r++) {
            score[r] = (hist[r - 1] + hist[r] + hist[r + 1]) / (2 * Math.PI * r);
            bestScore = Math.max(bestScore, score[r]);
        }
        if (bestScore < 0.3) return [];
        const out = [];
        for (let r = rMax; r >= rMin; r--) {
            const v = score[r];
            if (v < bestScore * 0.35) continue;
            if (v >= (score[r - 1] || 0) && v >= (score[r + 1] || 0) && v >= (score[r - 2] || 0) && v >= (score[r + 2] || 0)) out.push(r);
            if (out.length >= 6) break;
        }
        return out;
    }

    /**
     * 円周のうち、輪郭が実際にある割合 (72分割)。
     * 傾けて写すと楕円になるので、半径の 0.75倍〜1倍 の範囲で輪郭を探す。
     */
    function coverage(cx, cy, r, T) {
        let hit = 0;
        const dMin = Math.floor(-0.25 * r - 1);
        for (let k = 0; k < 72; k++) {
            const th = (2 * Math.PI * k) / 72, c = Math.cos(th), s = Math.sin(th);
            for (let dr = 2; dr >= dMin; dr--) {
                const x = Math.round(cx + c * (r + dr)), y = Math.round(cy + s * (r + dr));
                if (x < 1 || y < 1 || x >= W - 1 || y >= H - 1) continue;
                const i = y * W + x, m = mag[i];
                if (m >= T && Math.abs(gx[i] * c + gy[i] * s) / m > 0.8) { hit++; break; }
            }
        }
        return hit / 72;
    }

    /** 円の内側が銅色 (赤〜橙で、ある程度色が濃い) か */
    function isCopper(cx, cy, r) {
        let R = 0, G = 0, B = 0, n = 0;
        const step = Math.max(1, Math.round(r / 10));
        for (let y = Math.round(cy - r); y <= cy + r; y += step) {
            for (let x = Math.round(cx - r); x <= cx + r; x += step) {
                if (x < 0 || y < 0 || x >= W || y >= H) continue;
                const d = Math.hypot(x - cx, y - cy);
                if (d < r * 0.2 || d > r * 0.75) continue;
                const j = (y * W + x) * 4;
                R += rgba[j]; G += rgba[j + 1]; B += rgba[j + 2]; n++;
            }
        }
        if (!n) return false;
        R /= n; G /= n; B /= n;
        const mx = Math.max(R, G, B), mn = Math.min(R, G, B);
        if (mx < 40) return false;
        const sat = (mx - mn) / mx;
        if (sat < 0.15) return false;           // 灰色 (百円玉・ステンレスなど) は除外
        let hue;
        if (mx === R) hue = (60 * (G - B)) / (mx - mn);
        else if (mx === G) hue = 120 + (60 * (B - R)) / (mx - mn);
        else hue = 240 + (60 * (R - G)) / (mx - mn);
        if (hue < 0) hue += 360;
        return hue <= 50 || hue >= 345;
    }

    /**
     * 輪郭の位置を画素より細かく求め、直径を決める。
     * 傾けて写すと楕円になるので、最も長い方向の直径 (= 本来の直径) を使う。
     */
    function refineCircle(magArr, w, h, cx, cy, r, T) {
        const N = 90, rho = new Array(N).fill(null);
        // 楕円 (短い方向は半径の0.75倍程度まで) も入るように探す範囲を取る
        const step = 0.5, lo = Math.max(2, r * 0.73 - 2), hiR = r + Math.max(5, r * 0.06);
        const n = Math.ceil((hiR - lo) / step) + 1;
        const vals = new Float32Array(n);
        for (let k = 0; k < N; k++) {
            const th = (2 * Math.PI * k) / N, c = Math.cos(th), s = Math.sin(th);
            let vmax = 0;
            for (let j = 0; j < n; j++) {
                vals[j] = bilinear(magArr, w, h, cx + c * (lo + j * step), cy + s * (lo + j * step));
                vmax = Math.max(vmax, vals[j]);
            }
            if (vmax < T * 0.6) continue;
            let pick = -1; // 強い山のうち一番外側 (十円玉の外周)
            for (let j = 1; j < n - 1; j++) {
                if (vals[j] >= vals[j - 1] && vals[j] >= vals[j + 1] && vals[j] >= 0.6 * vmax) pick = j;
            }
            if (pick < 0) continue;
            const a = vals[pick - 1], b = vals[pick], d = vals[pick + 1], den = a - 2 * b + d;
            const off = den !== 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (a - d)) / den)) : 0;
            rho[k] = lo + (pick + off) * step;
        }
        const valid = rho.filter(v => v !== null);
        if (valid.length < N * 0.5) return null;
        // 外れ値 (影や模様を拾った点) を除く: 前後の角度の値と大きく違う点
        const local = rho.map((v, k) => {
            if (v === null) return null;
            const near = [];
            for (let d = -3; d <= 3; d++) { const u = rho[(k + d + N) % N]; if (u !== null) near.push(u); }
            return median(near);
        });
        for (let k = 0; k < N; k++) if (rho[k] !== null && Math.abs(rho[k] - local[k]) > 2.5) rho[k] = null;

        // 向かい合う2点の距離 d(θ) = D0 + A cos2θ + B sin2θ を当てはめる
        const half = N / 2;
        const ATA = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], ATb = [0, 0, 0];
        let mx = 0, my = 0, pairs = 0;
        for (let k = 0; k < half; k++) {
            if (rho[k] === null || rho[k + half] === null) continue;
            const th = (2 * Math.PI * k) / N;
            const row = [1, Math.cos(2 * th), Math.sin(2 * th)];
            const dd = rho[k] + rho[k + half];
            for (let p = 0; p < 3; p++) {
                ATb[p] += row[p] * dd;
                for (let q = 0; q < 3; q++) ATA[p][q] += row[p] * row[q];
            }
            const off = (rho[k] - rho[k + half]) / 2;
            mx += Math.cos(th) * off; my += Math.sin(th) * off;
            pairs++;
        }
        if (pairs < 15) return null;
        const sol = solve3(ATA, ATb);
        if (!sol) return null;
        const amp = Math.hypot(sol[1], sol[2]);
        const major = sol[0] + amp, minor = sol[0] - amp;
        if (!(major > 0) || minor / major < 0.7) return null; // 極端に潰れた円は十円玉ではない
        return { x: cx + mx / pairs, y: cy + my / pairs, diameterPx: major, minorPx: minor, r: major / 2 };
    }

    // ---------- 3. 対象を測る ----------

    /** 枠を一回り広げて測る。うまくいかない・はみ出しているときは、さらに広げて測り直す */
    function measureBox(box, coin) {
        let res = segment(box, 0.25, coin);
        if (!res || res.touches) res = segment(box, 0.6, coin) || res;
        return res ? { ...res, truncated: res.touches } : null;
    }

    /**
     * 白黒(0/1)の絵を上下左右に1画素太らせる(dilate)／細らせる(erode)
     * 範囲の外は「隣の画素と同じ」とみなす (ふちで物体が削れて、はみ出しを見落とさないように)
     */
    function morph(src, w, h, dilate) {
        const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
        for (let y = 0; y < h; y++) {
            const row = y * w;
            for (let x = 0; x < w; x++) {
                const c = src[row + x];
                const l = x > 0 ? src[row + x - 1] : c, r = x < w - 1 ? src[row + x + 1] : c;
                tmp[row + x] = dilate ? (l | c | r) : (l & c & r);
            }
        }
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const i = y * w + x, c = tmp[i];
                const u = y > 0 ? tmp[i - w] : c, d = y < h - 1 ? tmp[i + w] : c;
                out[i] = dilate ? (u | c | d) : (u & c & d);
            }
        }
        return out;
    }

    /**
     * 範囲の中から物体の画素を取り出す
     * shadow=true: 背景と同じ色味のまま暗いだけの画素 (影・木目などの濃淡) は背景とみなす
     * shadow=false: 背景の色との差が、ふちの色のばらつきより十分大きい画素を物体とみなす
     */
    function classify(data, w, h, inCoin, shadow) {
        const N = w * h;
        // 背景の色: 範囲のふち (外周の帯) の、明るい側の半分の中央値 (帯に影がかかっていても背景を取れる)
        const band = Math.max(3, Math.round(Math.min(w, h) * 0.03));
        const ring = [];
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (x >= band && x < w - band && y >= band && y < h - band) continue;
                const i = y * w + x;
                if (inCoin(x, y)) continue;
                const j = i * 4;
                ring.push(j);
            }
        }
        if (ring.length < 30) return null;
        const lum = (j) => data[j] * 0.299 + data[j + 1] * 0.587 + data[j + 2] * 0.114;
        ring.sort((a, b) => lum(a) - lum(b));
        const upper = ring.slice(ring.length >> 1);
        const med = (k) => {
            const v = upper.map(j => data[j + k]).sort((a, b) => a - b);
            return Math.max(1, v[v.length >> 1]);
        };
        const bgR = med(0), bgG = med(1), bgB = med(2);

        const obj = new Uint8Array(N);
        if (shadow) {
            for (let i = 0, j = 0; i < N; i++, j += 4) {
                const r = data[j] / bgR, g = data[j + 1] / bgG, b = data[j + 2] / bgB;
                const mean = (r + g + b) / 3;
                if (mean < SHADOW_DARK || mean > SHADOW_BRIGHT) { obj[i] = 1; continue; }
                const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
                if (mx - mn > SHADOW_CHROMA * mean) {
                    // 暗い背景ではノイズで色味がぶれるので、明るさの段階でも8以上ずれていることを確かめる
                    const dev = Math.max(Math.abs(data[j] - mean * bgR), Math.abs(data[j + 1] - mean * bgG), Math.abs(data[j + 2] - mean * bgB));
                    if (dev > 8) obj[i] = 1;
                }
            }
        } else {
            const dist = (j) => Math.abs(data[j] - bgR) + Math.abs(data[j + 1] - bgG) + Math.abs(data[j + 2] - bgB);
            const bd = Float32Array.from(ring.map(dist)).sort();
            const thr = Math.max(30, 1.4 * quantile(bd, 0.9) + 10);
            for (let i = 0, j = 0; i < N; i++, j += 4) if (dist(j) > thr) obj[i] = 1;
        }
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (inCoin(x, y)) obj[y * w + x] = 0;
        // ふち (本来は背景) のうち物体と判定された割合と、一番外側の1周で物体が何か所に分かれて現れるか。
        // 木目・織り目のような模様は「あちこちに細切れに」現れ、隣の紙や物のはみ出しは「1〜2か所にまとまって」現れる
        let ringObj = 0;
        for (const j of ring) if (obj[j >> 2]) ringObj++;
        const perim = [];
        for (let x = 0; x < w; x++) perim.push(x);
        for (let y = 1; y < h; y++) perim.push(y * w + w - 1);
        for (let x = w - 2; x >= 0; x--) perim.push((h - 1) * w + x);
        for (let y = h - 2; y > 0; y--) perim.push(y * w);
        let runs = 0, len = 0;
        for (let k = 0; k <= perim.length; k++) {
            const on = k < perim.length && obj[perim[k]];
            if (on) len++;
            else { if (len >= 3) runs++; len = 0; }
        }
        return { obj, ringFrac: ringObj / ring.length, ringRuns: runs };
    }

    function segment(box, e, coin) {
        // 調べる範囲 (高解像度の画素)
        const bx0 = box.x * HW, by0 = box.y * HH, bx1 = (box.x + box.w) * HW, by1 = (box.y + box.h) * HH;
        const ex = (bx1 - bx0) * e, ey = (by1 - by0) * e;
        const x0 = Math.max(0, Math.floor(bx0 - ex)), y0 = Math.max(0, Math.floor(by0 - ey));
        const x1 = Math.min(HW, Math.ceil(bx1 + ex)), y1 = Math.min(HH, Math.ceil(by1 + ey));
        const rw = x1 - x0, rh = y1 - y0;
        if (rw < 8 || rh < 8) return null;

        // 範囲が大きすぎるときは縮小して測る (f = 縮小率)
        const f = rw * rh > ROI_MAX_PX ? Math.sqrt(ROI_MAX_PX / (rw * rh)) : 1;
        const w = Math.max(8, Math.round(rw * f)), h = Math.max(8, Math.round(rh * f));
        const fx = w / rw, fy = h / rh;
        let data;
        if (f === 1) {
            data = hctx.getImageData(x0, y0, rw, rh).data;
        } else {
            roiCanvas.width = w; roiCanvas.height = h;
            rctx.drawImage(hiCanvas, x0, y0, rw, rh, 0, 0, w, h);
            data = rctx.getImageData(0, 0, w, h).data;
        }
        const N = w * h;
        // 範囲の中での座標 (u, v) ⇔ 高解像度の座標
        const toHiX = (u) => x0 + (u + 0.5) / fx, toHiY = (v) => y0 + (v + 0.5) / fy;
        const ccx = coin ? (coin.x - x0) * fx : 0, ccy = coin ? (coin.y - y0) * fy : 0;
        const cr2 = coin ? Math.pow((coin.r + 3) * fx, 2) : -1;
        const inCoin = (u, v) => coin ? (u - ccx) ** 2 + (v - ccy) ** 2 < cr2 : false;

        // 1. 物体の画素を取り出す
        //    影除去で分けられない (木目のように色味まで変わる模様で、ふちの多くが物体と判定される)
        //    ときや、物体が見つからないときは、色の差だけで分け直す
        let res = null;
        if (options.shadow) {
            const c = classify(data, w, h, inCoin, true);
            if (c && c.ringFrac <= 0.35 && c.ringRuns <= 4) res = pickObject(c.obj, w, h, box, x0, y0, fx, fy);
        }
        if (!res) {
            const c = classify(data, w, h, inCoin, false);
            res = c ? pickObject(c.obj, w, h, box, x0, y0, fx, fy) : null;
        }
        if (!res) return null;

        // 2. 傾きに合わせた最小の長方形
        const rect = minAreaRect(res.pts);
        const k = (fx + fy) / 2;
        let lengthPx = (Math.max(rect.w, rect.h) + 1) / k;
        let widthPx = (Math.min(rect.w, rect.h) + 1) / k;
        let corners = rect.corners.map(p => ({ x: toHiX(p.x) / HW, y: toHiY(p.y) / HH }));

        // 3. 外周がほぼ円なら、円の直径を寸法にする
        let circle = null;
        const fc = fitCircle(res.pts);
        const aspect = Math.min(rect.w, rect.h) / Math.max(1, Math.max(rect.w, rect.h));
        if (fc && fc.r >= 6 && fc.inlier >= CIRCLE_INLIER && fc.rms <= Math.max(0.8, fc.r * 0.02) && aspect >= 0.85) {
            const d = (fc.r + 0.5) * 2 / k; // 外周の点は物体の内側の画素の中心なので半画素足す
            lengthPx = widthPx = d;
            circle = { x: toHiX(fc.x) / HW, y: toHiY(fc.y) / HH, r: (fc.r + 0.5) / k / HW };
        }
        return { lengthPx, widthPx, corners, circle, touches: res.touches };
    }

    /**
     * 物体の画素から、枠の中にある塊を選び、内側の穴を埋めて外周の点を返す
     * @returns { pts: [x0,y0,x1,y1,...] (範囲の中の座標), touches } | null
     */
    function pickObject(obj0, w, h, box, x0, y0, fx, fy) {
        const N = w * h;
        // 小さな欠け・すき間を埋める (1画素太らせてから細らせる)
        const obj = morph(morph(obj0, w, h, true), w, h, false);

        // 枠の範囲 (この範囲の中の座標)
        const bu0 = (box.x * HW - x0) * fx, bu1 = ((box.x + box.w) * HW - x0) * fx;
        const bv0 = (box.y * HH - y0) * fy, bv1 = ((box.y + box.h) * HH - y0) * fy;

        // つながった塊に分ける
        const label = new Int32Array(N), stack = new Int32Array(N);
        const comps = [null];
        for (let s0 = 0; s0 < N; s0++) {
            if (!obj[s0] || label[s0]) continue;
            const id = comps.length;
            let sp = 0, size = 0, inside = 0, touches = false;
            stack[sp++] = s0; label[s0] = id;
            while (sp > 0) {
                const p = stack[--sp];
                const u = p % w, v = (p / w) | 0;
                size++;
                if (u >= bu0 && u < bu1 && v >= bv0 && v < bv1) inside++;
                if (u === 0 || v === 0 || u === w - 1 || v === h - 1) touches = true;
                if (u > 0 && obj[p - 1] && !label[p - 1]) { label[p - 1] = id; stack[sp++] = p - 1; }
                if (u < w - 1 && obj[p + 1] && !label[p + 1]) { label[p + 1] = id; stack[sp++] = p + 1; }
                if (v > 0 && obj[p - w] && !label[p - w]) { label[p - w] = id; stack[sp++] = p - w; }
                if (v < h - 1 && obj[p + w] && !label[p + w]) { label[p + w] = id; stack[sp++] = p + w; }
            }
            comps.push({ size, inside, touches });
        }
        // 4分の1以上が枠の中にある塊を候補にし、一番大きい塊の2%未満のゴミは捨てる
        let largest = 0;
        for (let k = 1; k < comps.length; k++) {
            const c = comps[k];
            if (c.inside >= 0.25 * c.size && c.size > largest) largest = c.size;
        }
        if (!largest) return null;
        const keepMin = Math.max(12, largest * 0.02);
        const keep = new Uint8Array(comps.length);
        let total = 0, touches = false;
        for (let k = 1; k < comps.length; k++) {
            const c = comps[k];
            if (c.inside >= 0.25 * c.size && c.size >= keepMin) { keep[k] = 1; total += c.size; touches = touches || c.touches; }
        }
        // 物体がほとんど無い、または範囲のほぼ全部が物体 (背景の見立てが外れた) ときは諦める
        if (total < 20 || total > N * 0.9) return null;

        // 内側の穴 (文字・照り返しなど) を埋める: 範囲のふちから背景をたどり、届かない所は物体とする
        const outside = new Uint8Array(N);
        let sp = 0;
        const isObj = (p) => keep[label[p]] === 1;
        const push = (p) => { if (!isObj(p) && !outside[p]) { outside[p] = 1; stack[sp++] = p; } };
        for (let u = 0; u < w; u++) { push(u); push(N - w + u); }
        for (let v = 0; v < h; v++) { push(v * w); push(v * w + w - 1); }
        while (sp > 0) {
            const p = stack[--sp];
            const u = p % w;
            if (u > 0) push(p - 1);
            if (u < w - 1) push(p + 1);
            if (p >= w) push(p - w);
            if (p < N - w) push(p + w);
        }
        // 外周の点
        const pts = [];
        for (let v = 0, i = 0; v < h; v++) {
            for (let u = 0; u < w; u++, i++) {
                if (outside[i]) continue;
                if (u === 0 || u === w - 1 || v === 0 || v === h - 1 ||
                    outside[i - 1] || outside[i + 1] || outside[i - w] || outside[i + w]) pts.push(u, v);
            }
        }
        if (pts.length < 8) return null;
        return { pts, touches };
    }

    /** 点の集まりを囲む、面積が最小の傾いた長方形 (flat = [x0,y0,x1,y1,...]) */
    function minAreaRect(flat) {
        const P = [];
        for (let i = 0; i < flat.length; i += 2) P.push([flat[i], flat[i + 1]]);
        P.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
        const lower = [], upper = [];
        for (const p of P) {
            while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
            lower.push(p);
        }
        for (let i = P.length - 1; i >= 0; i--) {
            const p = P[i];
            while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
            upper.push(p);
        }
        lower.pop(); upper.pop();
        const hull = lower.concat(upper);
        if (hull.length < 3) {
            const a = hull[0], b = hull[hull.length - 1];
            const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
            return { w: d, h: 0, corners: [{ x: a[0], y: a[1] }, { x: b[0], y: b[1] }, { x: b[0], y: b[1] }, { x: a[0], y: a[1] }] };
        }
        let best = null;
        for (let i = 0; i < hull.length; i++) {
            const a = hull[i], b = hull[(i + 1) % hull.length];
            const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
            if (len === 0) continue;
            const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len, vx = -uy, vy = ux;
            let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
            for (const p of hull) {
                const u = p[0] * ux + p[1] * uy, v = p[0] * vx + p[1] * vy;
                if (u < minU) minU = u; if (u > maxU) maxU = u;
                if (v < minV) minV = v; if (v > maxV) maxV = v;
            }
            const area = (maxU - minU) * (maxV - minV);
            if (!best || area < best.area) best = { area, ux, uy, vx, vy, minU, maxU, minV, maxV };
        }
        const pt = (u, v) => ({ x: u * best.ux + v * best.vx, y: u * best.uy + v * best.vy });
        return {
            w: best.maxU - best.minU, h: best.maxV - best.minV,
            corners: [pt(best.minU, best.minV), pt(best.maxU, best.minV), pt(best.maxU, best.maxV), pt(best.minU, best.maxV)]
        };
    }

    /** 点の集まりに円を当てはめる。外れた点を除いてもう一度当てはめる */
    function fitCircle(flat) {
        const solve = (use) => {
            let n = 0, mx = 0, my = 0;
            for (let i = 0; i < flat.length; i += 2) { if (use && !use[i >> 1]) continue; mx += flat[i]; my += flat[i + 1]; n++; }
            if (n < 8) return null;
            mx /= n; my /= n;
            let Suu = 0, Suv = 0, Svv = 0, Suuu = 0, Svvv = 0, Suvv = 0, Svuu = 0;
            for (let i = 0; i < flat.length; i += 2) {
                if (use && !use[i >> 1]) continue;
                const u = flat[i] - mx, v = flat[i + 1] - my;
                Suu += u * u; Suv += u * v; Svv += v * v;
                Suuu += u * u * u; Svvv += v * v * v; Suvv += u * v * v; Svuu += v * u * u;
            }
            const det = Suu * Svv - Suv * Suv;
            if (Math.abs(det) < 1e-9) return null;
            const b1 = (Suuu + Suvv) / 2, b2 = (Svvv + Svuu) / 2;
            const uc = (b1 * Svv - b2 * Suv) / det, vc = (b2 * Suu - b1 * Suv) / det;
            return { x: uc + mx, y: vc + my, r: Math.sqrt(uc * uc + vc * vc + (Suu + Svv) / n) };
        };
        const total = flat.length / 2;
        let c = solve(null);
        if (!c) return null;
        let use = null, kept = total;
        for (let pass = 0; pass < 2; pass++) {
            const lim = Math.max(1.5, c.r * 0.04);
            const next = new Uint8Array(total);
            let k = 0;
            for (let i = 0; i < flat.length; i += 2) {
                if (Math.abs(Math.hypot(flat[i] - c.x, flat[i + 1] - c.y) - c.r) <= lim) { next[i >> 1] = 1; k++; }
            }
            if (k < total * 0.6) break;
            const c2 = solve(next);
            if (!c2) break;
            c = c2; use = next; kept = k;
        }
        let ss = 0, cnt = 0;
        for (let i = 0; i < flat.length; i += 2) {
            if (use && !use[i >> 1]) continue;
            const d = Math.hypot(flat[i] - c.x, flat[i + 1] - c.y) - c.r;
            ss += d * d; cnt++;
        }
        return { x: c.x, y: c.y, r: c.r, rms: Math.sqrt(ss / Math.max(1, cnt)), inlier: kept / total };
    }

    // ---------- 公開する関数 ----------

    window.AppMeasure = {
        COIN_MM,

        /** 解析用の画像を取り込む (推論に使うフレームと同じ瞬間に呼ぶ) */
        capture,

        /** 設定: { shadow: 影を除くか } */
        setOptions(o) { Object.assign(options, o); },

        /**
         * 取り込んだ画像で十円玉を探し、各枠の寸法を測る
         * @param boxes 0〜1 の座標の枠のリスト
         * @returns { coin: {x, y, r, diameterPx} (0〜1の座標, r は幅に対する割合) | null,
         *            items: [{ lengthPx, widthPx, corners, circle, truncated } | null] }
         *          lengthPx などは高解像度の画素単位 (十円玉の diameterPx と同じ単位)
         *          circle は丸い物と判定したとき { x, y, r } (0〜1の座標)、それ以外は null
         */
        analyze(boxes) {
            if (!captured) return { coin: null, items: [] };
            const coin = findCoin();
            const items = (boxes || []).map(b => measureBox(b, coin));
            return {
                coin: coin ? { x: coin.x / HW, y: coin.y / HH, r: coin.r / HW, diameterPx: coin.diameterPx } : null,
                items
            };
        }
    };
})();
