/**
 * @file measure.js
 * @description 測定モード: 十円玉を基準に、検出した対象の寸法を測る (window.AppMeasure)
 *
 * 仕組み (AIは使わない、昔ながらの画像処理):
 *  1. 画面に見えている範囲を解析用の画像 (幅 最大360px) に取り込む
 *  2. 十円玉: 輪郭 (エッジ) から円を探し、銅色かどうかで確認。直径23.5mm を基準にする
 *  3. 対象: AIの枠を一回り広げ、その中で背景と色が違う塊を拾い、
 *           塊の向きに合わせた長方形で長さ・幅を測る
 */

(function() {
    const COIN_MM = 23.5;      // 十円玉の直径 (mm)
    const MAX_W = 360;         // 解析画像の最大幅 (px)
    const EDGE_MIN = 60;       // エッジとみなす最小の強さ

    const work = document.createElement('canvas');
    const wctx = work.getContext('2d', { willReadFrequently: true });
    let W = 0, H = 0;
    let rgba = null, gray = null, gx = null, gy = null, mag = null, acc = null;

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

    function bilinear(arr, x, y) {
        const x0 = Math.floor(x), y0 = Math.floor(y);
        if (x0 < 0 || y0 < 0 || x0 >= W - 1 || y0 >= H - 1) return 0;
        const fx = x - x0, fy = y - y0, i = y0 * W + x0;
        return arr[i] * (1 - fx) * (1 - fy) + arr[i + 1] * fx * (1 - fy) +
               arr[i + W] * (1 - fx) * fy + arr[i + W + 1] * fx * fy;
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
        const s = Math.min(1, MAX_W / crop.sw);
        const w = Math.max(16, Math.round(crop.sw * s));
        const h = Math.max(16, Math.round(crop.sh * s));
        if (w !== W || h !== H) {
            W = w; H = h;
            work.width = W; work.height = H;
            gray = new Float32Array(W * H);
            gx = new Float32Array(W * H);
            gy = new Float32Array(W * H);
            mag = new Float32Array(W * H);
            acc = new Float32Array((W >> 1) * (H >> 1));
        }
        wctx.drawImage(source, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, W, H);
        rgba = wctx.getImageData(0, 0, W, H).data;
        for (let i = 0, j = 0; i < W * H; i++, j += 4) {
            gray[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
        }
        // Sobel フィルタで輪郭の強さと向きを求める
        for (let y = 1; y < H - 1; y++) {
            for (let x = 1; x < W - 1; x++) {
                const i = y * W + x;
                const a = gray[i - W - 1], b = gray[i - W], c = gray[i - W + 1];
                const d = gray[i - 1], f = gray[i + 1];
                const g = gray[i + W - 1], hh = gray[i + W], k = gray[i + W + 1];
                const sx = (c + 2 * f + k) - (a + 2 * d + g);
                const sy = (g + 2 * hh + k) - (a + 2 * b + c);
                gx[i] = sx; gy[i] = sy;
                mag[i] = Math.sqrt(sx * sx + sy * sy);
            }
        }
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
            const r = estimateRadius(c.x, c.y, rMin, rMax, T);
            if (!r) continue;
            const cov = coverage(c.x, c.y, r, T);
            if (cov < 0.6) continue;
            if (!isCopper(c.x, c.y, r)) continue;
            if (!best || cov > best.cov) best = { x: c.x, y: c.y, r, cov };
        }
        if (!best) return null;
        return refineCoin(best.x, best.y, best.r, T, best.cov);
    }

    /** 中心から見て放射方向を向いた輪郭が最も多い半径 (同じくらいなら外側を選ぶ) */
    function estimateRadius(cx, cy, rMin, rMax, T) {
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
        let bestScore = 0, bestR = 0;
        const score = new Float32Array(hist.length);
        for (let r = rMin; r <= rMax; r++) {
            score[r] = (hist[r - 1] + hist[r] + hist[r + 1]) / (2 * Math.PI * r);
            if (score[r] > bestScore) { bestScore = score[r]; bestR = r; }
        }
        if (bestScore < 0.3) return 0;
        // (1) 十円玉の模様の内側の輪より外周を選ぶため、強い山のうち一番外側
        let rA = bestR;
        for (let r = rMax; r >= rMin; r--) if (score[r] >= bestScore * 0.6) { rA = r; break; }
        // (2) 傾いた十円玉 (楕円) では短い半径と長い半径の両方に山ができる。
        //     長い半径 (= 本来の直径) を返すため、少し外側まで見る
        const rOut = Math.min(rMax, Math.floor(rA / 0.75));
        for (let r = rOut; r > rA; r--) if (score[r] >= score[rA] * 0.5) return r;
        return rA;
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
    function refineCoin(cx, cy, r, T, cov) {
        const N = 90, rho = new Array(N).fill(null);
        // 楕円 (短い方向は半径の0.75倍程度まで) も入るように探す範囲を取る
        const step = 0.5, lo = Math.max(2, r * 0.73 - 2), n = Math.ceil((r + 5 - lo) / step) + 1;
        const vals = new Float32Array(n);
        for (let k = 0; k < N; k++) {
            const th = (2 * Math.PI * k) / N, c = Math.cos(th), s = Math.sin(th);
            let vmax = 0;
            for (let j = 0; j < n; j++) {
                vals[j] = bilinear(mag, cx + c * (lo + j * step), cy + s * (lo + j * step));
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
        return {
            x: cx + mx / pairs, y: cy + my / pairs,
            diameterPx: major, minorPx: minor,
            r: major / 2, coverage: cov
        };
    }

    // ---------- 3. 対象を測る ----------

    /** 枠を一回り広げて測る。うまくいかない・はみ出しているときは、さらに広げて測り直す */
    function measureBox(box, coin) {
        let res = segment(box, 0.25, coin);
        if (!res || res.touches) res = segment(box, 0.6, coin) || res;
        return res ? { ...res, truncated: res.touches } : null;
    }

    function segment(box, e, coin) {
        const bx0 = box.x * W, by0 = box.y * H, bx1 = (box.x + box.w) * W, by1 = (box.y + box.h) * H;
        const ex = (bx1 - bx0) * e, ey = (by1 - by0) * e;
        const x0 = Math.max(0, Math.floor(bx0 - ex)), y0 = Math.max(0, Math.floor(by0 - ey));
        const x1 = Math.min(W - 1, Math.ceil(bx1 + ex)), y1 = Math.min(H - 1, Math.ceil(by1 + ey));
        const rw = x1 - x0 + 1, rh = y1 - y0 + 1;
        if (rw < 8 || rh < 8) return null;

        // 背景の色 = 調べる範囲のふち (外周の細い帯) の色の中央値
        const band = Math.max(2, Math.round(Math.min(rw, rh) * 0.04));
        const hR = new Uint32Array(256), hG = new Uint32Array(256), hB = new Uint32Array(256);
        const borderIdx = [];
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
                if (x - x0 >= band && x1 - x >= band && y - y0 >= band && y1 - y >= band) continue;
                const j = (y * W + x) * 4;
                hR[rgba[j]]++; hG[rgba[j + 1]]++; hB[rgba[j + 2]]++;
                borderIdx.push(j);
            }
        }
        const histMedian = (h, total) => {
            let s = 0;
            for (let v = 0; v < 256; v++) { s += h[v]; if (s * 2 >= total) return v; }
            return 255;
        };
        const nb = borderIdx.length;
        const mR = histMedian(hR, nb), mG = histMedian(hG, nb), mB = histMedian(hB, nb);
        const bd = new Float32Array(nb);
        for (let k = 0; k < nb; k++) {
            const j = borderIdx[k];
            bd[k] = Math.abs(rgba[j] - mR) + Math.abs(rgba[j + 1] - mG) + Math.abs(rgba[j + 2] - mB);
        }
        bd.sort();
        // 背景の模様 (木目・筋など) のばらつきより、はっきり色が違うものだけを対象とする
        const thr = Math.max(30, 1.4 * quantile(bd, 0.9) + 10);

        // 背景と色が違う画素 (十円玉の部分は除く)
        const mask = new Uint8Array(rw * rh);
        const coinR2 = coin ? Math.pow(coin.r + 3, 2) : -1;
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
                if (coin && (x - coin.x) ** 2 + (y - coin.y) ** 2 < coinR2) continue;
                const j = (y * W + x) * 4;
                const dist = Math.abs(rgba[j] - mR) + Math.abs(rgba[j + 1] - mG) + Math.abs(rgba[j + 2] - mB);
                if (dist > thr) mask[(y - y0) * rw + (x - x0)] = 1;
            }
        }

        // つながった塊に分け、枠の中にある塊を集める
        const label = new Int32Array(rw * rh);
        const comps = [];
        const stack = [];
        for (let i = 0; i < mask.length; i++) {
            if (!mask[i] || label[i]) continue;
            const id = comps.length + 1;
            let size = 0, inside = 0, touches = false;
            label[i] = id; stack.push(i);
            while (stack.length) {
                const k = stack.pop();
                const lx = k % rw, ly = (k / rw) | 0, X = lx + x0, Y = ly + y0;
                size++;
                if (X >= bx0 && X < bx1 && Y >= by0 && Y < by1) inside++;
                if (lx === 0 || ly === 0 || lx === rw - 1 || ly === rh - 1) touches = true;
                if (lx > 0 && mask[k - 1] && !label[k - 1]) { label[k - 1] = id; stack.push(k - 1); }
                if (lx < rw - 1 && mask[k + 1] && !label[k + 1]) { label[k + 1] = id; stack.push(k + 1); }
                if (ly > 0 && mask[k - rw] && !label[k - rw]) { label[k - rw] = id; stack.push(k - rw); }
                if (ly < rh - 1 && mask[k + rw] && !label[k + rw]) { label[k + rw] = id; stack.push(k + rw); }
            }
            comps.push({ size, inside, touches });
        }
        const minSize = Math.max(6, 0.004 * (bx1 - bx0) * (by1 - by0));
        const use = new Uint8Array(comps.length + 1);
        let total = 0, touches = false;
        comps.forEach((c, k) => {
            if (c.size >= minSize && c.inside >= 0.25 * c.size) {
                use[k + 1] = 1; total += c.size; touches = touches || c.touches;
            }
        });
        if (total === 0) return null;

        // 塊の向き (主成分) を求め、その向きの長方形で長さ・幅を測る
        const px = new Float32Array(total), py = new Float32Array(total);
        let n = 0, sx = 0, sy = 0;
        for (let i = 0; i < label.length; i++) {
            if (!use[label[i]]) continue;
            px[n] = (i % rw) + x0 + 0.5; py[n] = ((i / rw) | 0) + y0 + 0.5;
            sx += px[n]; sy += py[n]; n++;
        }
        const mx = sx / n, my = sy / n;
        let sxx = 0, syy = 0, sxy = 0;
        for (let k = 0; k < n; k++) {
            const dx = px[k] - mx, dy = py[k] - my;
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
        const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
        const ux = Math.cos(th), uy = Math.sin(th);
        const pu = new Float32Array(n), pv = new Float32Array(n);
        for (let k = 0; k < n; k++) {
            const dx = px[k] - mx, dy = py[k] - my;
            pu[k] = dx * ux + dy * uy;
            pv[k] = -dx * uy + dy * ux;
        }
        pu.sort(); pv.sort();
        const uLo = quantile(pu, 0.005) - 0.5, uHi = quantile(pu, 0.995) + 0.5;
        const vLo = quantile(pv, 0.005) - 0.5, vHi = quantile(pv, 0.995) + 0.5;
        const corner = (u, v) => ({ x: (mx + u * ux - v * uy) / W, y: (my + u * uy + v * ux) / H });

        return {
            lengthPx: uHi - uLo,
            widthPx: vHi - vLo,
            corners: [corner(uLo, vLo), corner(uHi, vLo), corner(uHi, vHi), corner(uLo, vHi)],
            touches
        };
    }

    // ---------- 公開する関数 ----------

    window.AppMeasure = {
        COIN_MM,

        /** 解析用の画像を取り込む (推論に使うフレームと同じ瞬間に呼ぶ) */
        capture,

        /**
         * 取り込んだ画像で十円玉を探し、各枠の寸法を測る
         * @param boxes 0〜1 の座標の枠のリスト
         * @returns { coin: {x, y, r, diameterPx} (0〜1の座標, r は幅に対する割合) | null,
         *            items: [{ lengthPx, widthPx, corners, truncated } | null] }
         *          lengthPx などは解析画像の画素単位 (十円玉の diameterPx と同じ単位)
         */
        analyze(boxes) {
            if (!rgba) return { coin: null, items: [] };
            const coin = findCoin();
            const items = (boxes || []).map(b => measureBox(b, coin));
            return {
                coin: coin ? { x: coin.x / W, y: coin.y / H, r: coin.r / W, diameterPx: coin.diameterPx } : null,
                items
            };
        }
    };
})();
