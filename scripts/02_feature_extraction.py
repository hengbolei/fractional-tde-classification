# 02_feature_extraction.py
import pandas as pd
import numpy as np
from scipy.special import binom
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import os
import time
import warnings

warnings.filterwarnings('ignore')


def gl_fractional_derivative(y, alpha=0.5, dt=1.0, n_terms=80):
    """Grünwald-Letnikov 分数阶导数"""
    N = len(y)
    if N < 10:
        return np.zeros(N, dtype=np.float32)

    k = np.arange(n_terms)
    coeffs = (-1.0) ** k * binom(alpha, k)
    coeffs = np.array(coeffs, dtype=np.float32)

    y = np.array(y, dtype=np.float32)
    deriv = np.zeros(N, dtype=np.float32)

    for i in range(N):
        k_max = min(i + 1, n_terms)
        deriv[i] = np.sum(coeffs[:k_max] * y[i - np.arange(k_max)]) / (dt ** alpha)

    return deriv


def extract_unified_stats(series, t=None):
    """统一的 10 个统计量（常规和分数阶共用）"""
    if len(series) < 3 or np.all(np.isnan(series)):
        return [np.nan] * 10

    s = np.array(series, dtype=np.float32)
    stats = [
        float(np.mean(s)),
        float(np.std(s)),
        float(np.max(s)),
        float(np.min(s)),
        float(np.max(s) - np.min(s)),  # amplitude
        float(skew(s, nan_policy='omit')),
        float(kurtosis(s, nan_policy='omit')),
        float(np.median(s)),
        float(np.median(np.abs(s - np.median(s)))),  # mad
        float(np.argmax(s)) if len(s) > 0 else np.nan  # rise_time
    ]
    return stats


def main():
    PROCESSED_DIR = "./processed"
    INPUT_FILE = os.path.join(PROCESSED_DIR, "data_processed.parquet")
    OUTPUT_CONV = os.path.join(PROCESSED_DIR, "features_conventional.parquet")
    OUTPUT_FRAC = os.path.join(PROCESSED_DIR, "features_fractional.parquet")

    ALPHAS = [0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
    MIN_POINTS = 10
    N_TERMS = 80

    start_time = time.time()

    print("加载预处理后的数据...")
    df = pd.read_parquet(INPUT_FILE, engine='pyarrow')
    print(f"数据形状: {df.shape}")

    # 按 object_id + filter 分组
    grouped = df.groupby(['object_id', 'filter'])
    print(f"共有 {len(grouped)} 个 (object_id, filter) 组合")

    conv_features = []
    frac_features = []

    for (obj_id, filt), group in tqdm(grouped, desc="特征提取"):
        group = group.sort_values('mjd')
        y = group['flux_interp'].values
        t = group['mjd'].values

        if len(y) < MIN_POINTS:
            continue

        dt = float(max(0.1, group['mjd'].diff().median()))

        # ==================== 常规特征 ====================
        conv_stats = extract_unified_stats(y, t)
        conv_row = {'object_id': obj_id, 'filter': filt}
        cols = ['mean', 'std', 'max', 'min', 'amplitude', 'skew', 'kurt', 'median', 'mad', 'rise_time']
        conv_row.update({f"{col}_flux": conv_stats[i] for i, col in enumerate(cols)})
        conv_features.append(conv_row)

        # ==================== 分数阶特征 ====================
        frac_row = {'object_id': obj_id}
        for alpha in ALPHAS:
            if alpha == 0.0:
                deriv = y.astype(np.float32)
            else:
                deriv = gl_fractional_derivative(y, alpha=alpha, dt=dt, n_terms=N_TERMS)

            frac_stats = extract_unified_stats(deriv, t)
            prefix = f"alpha_{alpha:.1f}_{filt}"
            frac_row.update({f"{prefix}_{col}": frac_stats[i] for i, col in enumerate(cols)})

        frac_features.append(frac_row)

    # ==================== 保存常规特征（宽表） ====================
    conv_df = pd.DataFrame(conv_features)
    conv_pivot = conv_df.pivot_table(
        index='object_id',
        columns='filter',
        values=[col for col in conv_df.columns if col not in ['object_id', 'filter']],
        aggfunc='first'
    )
    conv_pivot.columns = [f"{col[0]}_{col[1]}" for col in conv_pivot.columns]
    conv_pivot = conv_pivot.reset_index()
    conv_pivot.to_parquet(OUTPUT_CONV, engine='pyarrow', index=False)
    print(f"常规特征保存完成！形状: {conv_pivot.shape}")

    # ==================== 保存分数阶特征（宽表） ====================
    frac_df = pd.DataFrame(frac_features)
    frac_pivot = frac_df.set_index('object_id')
    frac_pivot = frac_pivot.reset_index()
    frac_pivot.to_parquet(OUTPUT_FRAC, engine='pyarrow', index=False)
    print(f"分数阶特征保存完成！形状: {frac_pivot.shape}")

    elapsed = (time.time() - start_time) / 60
    print(f"\n✅ 特征提取全部完成！总耗时: {elapsed:.1f} 分钟")
    print(f"常规特征文件: {OUTPUT_CONV}")
    print(f"分数阶特征文件: {OUTPUT_FRAC}")


if __name__ == "__main__":
    main()