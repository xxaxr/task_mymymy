/**
 * Nyx 数据集加载与统计模块
 * ---------------------------------------------------------------
 * Nyx 宇宙学模拟每个时间步输出 128x128x128 的 float32 密度场
 * （小端字节序，列优先存储：z 变化最快，其次 y，最后 x）。
 *
 * 注意：该数据集的数值本身已经是 log10(密度) 量级（约 7.7 ~ 14.5），
 * 因此"密度对数直方图"直接对数值分箱，不再重复取对数。
 */

const NYX_GRID_SIZE = 128;
const NYX_VOXEL_COUNT = NYX_GRID_SIZE * NYX_GRID_SIZE * NYX_GRID_SIZE; // 2097152

const NYX_HIST_MIN = 7.5;      // log10 密度统计下界
const NYX_HIST_MAX = 14.5;     // log10 密度统计上界
const NYX_HIST_BINS = 140;     // 直方图分箱数

/**
 * 解析单个时间步的密度数据。
 * 输入: ArrayBuffer（128^3 个 float32，小端，列优先 z,y,x）
 * 输出: Float32Array，索引顺序 (x, y, z)，即 x*N*N + y*N + z
 *       （便于 texImage3D 上传为 3D 纹理）
 */
function parseNyxDensity(buffer) {
    const dv = new DataView(buffer);
    const out = new Float32Array(NYX_VOXEL_COUNT);
    const N = NYX_GRID_SIZE;
    for (let x = 0; x < N; x++) {
        const xBase = x * N * N;
        for (let y = 0; y < N; y++) {
            const rawBase = y * N + xBase;
            const outBase = xBase + y * N;
            for (let z = 0; z < N; z++) {
                out[outBase + z] = dv.getFloat32((rawBase + z) * 4, true);
            }
        }
    }
    return out;
}

/** 基础统计量（min / max / mean / std） */
function computeDensityStats(arr) {
    let min = Infinity, max = -Infinity, sum = 0, sumSq = 0;
    const n = arr.length;
    for (let i = 0; i < n; i++) {
        const v = arr[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sumSq += v * v;
    }
    const mean = sum / n;
    const variance = Math.max(0, sumSq / n - mean * mean);
    return { min, max, mean, std: Math.sqrt(variance), count: n };
}

/** 由直方图累积分布估计分位数 */
function quantileFromHist(hist, domainMin, domainMax, q) {
    let total = 0;
    for (let i = 0; i < hist.length; i++) total += hist[i];
    const target = q * total;
    const binW = (domainMax - domainMin) / hist.length;
    let cum = 0;
    for (let i = 0; i < hist.length; i++) {
        if (cum + hist[i] >= target) {
            const frac = hist[i] > 0 ? (target - cum) / hist[i] : 0;
            return domainMin + (i + frac) * binW;
        }
        cum += hist[i];
    }
    return domainMax;
}

/**
 * 对单个时间步做完整统计摘要：
 * 统计量 + log10 密度直方图 + 常用分位数
 */
function summarizeDensity(data, domainMin = NYX_HIST_MIN, domainMax = NYX_HIST_MAX, numBins = NYX_HIST_BINS) {
    const n = data.length;
    let min = Infinity, max = -Infinity, sum = 0, sumSq = 0;
    const hist = new Uint32Array(numBins);
    const binW = (domainMax - domainMin) / numBins;
    const invBinW = 1 / binW;

    for (let i = 0; i < n; i++) {
        const v = data[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sumSq += v * v;
        let b = Math.floor((v - domainMin) * invBinW);
        if (b < 0) b = 0;
        if (b >= numBins) b = numBins - 1;
        hist[b]++;
    }
    const mean = sum / n;
    const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));

    return {
        min, max, mean, std,
        hist, domainMin, domainMax, numBins,
        p1: quantileFromHist(hist, domainMin, domainMax, 0.01),
        p5: quantileFromHist(hist, domainMin, domainMax, 0.05),
        p50: quantileFromHist(hist, domainMin, domainMax, 0.50),
        p95: quantileFromHist(hist, domainMin, domainMax, 0.95),
        p99: quantileFromHist(hist, domainMin, domainMax, 0.99)
    };
}

/** 统计直方图中落在 [a, b] 内的体素占比 */
function histogramFraction(summary, a, b) {
    const { hist, domainMin, domainMax, numBins } = summary;
    const binW = (domainMax - domainMin) / numBins;
    let total = 0, inside = 0;
    for (let i = 0; i < numBins; i++) {
        const c = hist[i];
        total += c;
        const lo = domainMin + i * binW;
        const hi = lo + binW;
        if (hi > a && lo < b) inside += c;
    }
    return total > 0 ? inside / total : 0;
}

/* ------------------------------------------------------------------ */
/* float32 -> float16（用于 WebGL2 R16F 3D 纹理）                      */
/* ------------------------------------------------------------------ */
const _f32 = new Float32Array(1);
const _i32 = new Int32Array(_f32.buffer);

function floatToHalf(val) {
    _f32[0] = val;
    const x = _i32[0];
    const sign = (x >> 16) & 0x8000;
    let exp = (x >> 23) & 0xff;
    let mant = x & 0x7fffff;
    if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
    let e = exp - 127 + 15;
    if (e >= 0x1f) return sign | 0x7c00;
    if (e <= 0) {
        if (e < -10) return sign;
        mant |= 0x800000;
        const shift = 14 - e;
        return sign | (mant >> shift);
    }
    return sign | (e << 10) | (mant >> 13);
}

/** 将密度数据归一化到 [0,1] 并转为 half float（R16F） */
function volumeToHalf(data, texMin, texMax) {
    const n = data.length;
    const out = new Uint16Array(n);
    const inv = 1 / Math.max(1e-6, texMax - texMin);
    for (let i = 0; i < n; i++) {
        let t = (data[i] - texMin) * inv;
        if (t < 0) t = 0; else if (t > 1) t = 1;
        out[i] = floatToHalf(t);
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* 数据集：加载全部时间步摘要，按需读取体数据                          */
/* ------------------------------------------------------------------ */

/**
 * 从文件列表创建数据集。
 * 只保留每个时间步的统计摘要（体数据按需重读，避免 100 步 × 8MB 常驻内存）。
 * files: File[]（名称形如 0000.dat ... 0099.dat）
 */
async function loadNyxDataset(files, onProgress) {
    const fileArr = Array.from(files)
        .filter(f => /\.dat$/i.test(f.name))
        .sort((a, b) => a.name.localeCompare(b.name));
    if (fileArr.length === 0) throw new Error('未找到 .dat 数据文件');

    const summaries = [];
    for (let i = 0; i < fileArr.length; i++) {
        const buf = await fileArr[i].arrayBuffer();
        const data = parseNyxDensity(buf);
        summaries.push({ name: fileArr[i].name, ...summarizeDensity(data) });
        if (onProgress) onProgress((i + 1) / fileArr.length, fileArr[i].name);
        // 让出主线程，保证进度条刷新
        await new Promise(r => setTimeout(r, 0));
    }

    const min = Math.min(...summaries.map(s => s.min));
    const max = Math.max(...summaries.map(s => s.max));

    // 最近使用的时间步体数据缓存（避免反复读取文件）
    const cache = new Map();
    const CACHE_LIMIT = 3;

    return {
        files: fileArr,
        names: fileArr.map(f => f.name),
        summaries,
        min,
        max,
        async getData(index) {
            if (cache.has(index)) {
                const v = cache.get(index);
                cache.delete(index);
                cache.set(index, v); // LRU 触碰
                return v;
            }
            const buf = await fileArr[index].arrayBuffer();
            const data = parseNyxDensity(buf);
            cache.set(index, data);
            if (cache.size > CACHE_LIMIT) {
                cache.delete(cache.keys().next().value);
            }
            return data;
        }
    };
}
