# 00_preprocess.py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import glob
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def preprocess_lightcurve(group, dt=1.0):
    group = group.sort_values('mjd').copy()
    t = group['mjd'].values
    flux = group['flux'].values

    if len(t) < 10:  # 统一最小观测点数
        return None

    # 1. 基线扣除
    n_baseline = max(3, int(0.2 * len(t)))
    weights = 1.0 / group['flux_err'].values[:n_baseline]
    baseline = np.average(flux[:n_baseline], weights=weights)
    flux = flux - baseline

    # 2. 均匀插值
    t_uniform = np.arange(t.min(), t.max() + dt, dt)
    interp_func = interp1d(t, flux, kind='linear', fill_value='extrapolate', bounds_error=False)
    flux_interp = interp_func(t_uniform)

    result = pd.DataFrame({
        'mjd': t_uniform,
        'flux_interp': flux_interp,
    })
    result['filter'] = group['filter'].iloc[0]
    result['object_id'] = group['object_id'].iloc[0]

    return result


def main():
    PROCESSED_DIR = "./processed"
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    DATA_DIR = "./mallorn-dataset"  # ← 修改为你的数据集路径

    print("1. 加载标签文件...")
    train_log = pd.read_csv(os.path.join(DATA_DIR, "train_log.csv"))
    print(f"标签数量: {len(train_log)}")

    print("2. 合并 lightcurves 文件...")
    lc_files = glob.glob(os.path.join(DATA_DIR, "split_*", "train_full_lightcurves.csv"))
    train_lc = pd.concat([pd.read_csv(f) for f in tqdm(lc_files)], ignore_index=True)

    # 列名清洗
    train_lc.columns = train_lc.columns.str.strip()
    train_lc = train_lc.rename(columns={
        'Time (MJD)': 'mjd', 'Time (MJD )': 'mjd',
        'Flux': 'flux', 'Flux_err': 'flux_err', 'Filter': 'filter'
    })

    df = train_lc.merge(train_log, on='object_id', how='left')
    print(f"合并后总数据量: {df.shape}")

    print("3. 开始预处理（基线扣除 + 均匀插值）...")
    grouped = df.groupby(['object_id', 'filter'])
    processed_list = []

    for key in tqdm(grouped.groups.keys(), desc="预处理"):
        group = grouped.get_group(key)
        result = preprocess_lightcurve(group, dt=1.0)
        if result is not None:
            processed_list.append(result)

    df_processed = pd.concat(processed_list, ignore_index=True)
    print(f"预处理完成！最终形状: {df_processed.shape}")

    # 保存
    df_processed.to_parquet(os.path.join(PROCESSED_DIR, "data_processed.parquet"), engine='pyarrow', index=False)
    train_log.to_parquet(os.path.join(PROCESSED_DIR, "labels.parquet"), engine='pyarrow', index=False)

    print("✅ 预处理完成！文件已保存：")
    print("   → processed/data_processed.parquet")
    print("   → processed/labels.parquet")


if __name__ == "__main__":
    main()