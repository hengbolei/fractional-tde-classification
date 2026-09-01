# 01_eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    PROCESSED_DIR = "./processed"
    PLOT_DIR = os.path.join(PROCESSED_DIR, "eda_plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("加载预处理后的数据和标签...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "data_processed.parquet"))
    labels = pd.read_parquet(os.path.join(PROCESSED_DIR, "labels.parquet"))

    # 正确计算 TDE 对象数量（按 object_id）
    label_map = labels.set_index('object_id')['target'].to_dict()
    object_labels = pd.DataFrame({
        'object_id': df['object_id'].unique()
    })
    object_labels['target'] = object_labels['object_id'].map(label_map)
    object_labels['is_tde'] = object_labels['target'] == 1

    print(f"总对象数: {len(object_labels)}")
    print(f"TDE 对象数: {object_labels['is_tde'].sum()}")
    print(f"非TDE 对象数: {(~object_labels['is_tde']).sum()}")

    # 为 df 添加 is_tde 标签（用于绘图）
    df['is_tde'] = df['object_id'].map(object_labels.set_index('object_id')['is_tde'])

    print(f"\n总观测记录数: {len(df):,}")
    print(f"波段分布:\n{df['filter'].value_counts()}")

    # ==================== 统计对比 ====================
    print("\n=== TDE vs 非TDE 统计对比 ===")
    stats = df.groupby('is_tde').agg({
        'flux_interp': ['mean', 'std', 'max', 'min'],
        'mjd': ['count', 'nunique']
    }).round(4)
    print(stats)

    # ==================== 可视化 ====================

    # 图1：每个波段观测点数量分布 (TDE vs 非TDE)
    obs_count = df.groupby(['object_id', 'filter', 'is_tde']).size().reset_index(name='n_obs')
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='filter', y='n_obs', hue='is_tde', data=obs_count)
    plt.title('不同波段观测点数量分布 (TDE vs 非TDE)')
    plt.ylabel('观测点数量 (n_obs)')
    plt.savefig(os.path.join(PLOT_DIR, "01_obs_count_per_filter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: 01_obs_count_per_filter.png")

    # 图2：r 波段流量分布对比
    plt.figure(figsize=(10, 6))
    sample_r = df[df['filter'] == 'r'].sample(frac=0.05, random_state=42)  # 进一步采样加速
    sns.histplot(data=sample_r, x='flux_interp', hue='is_tde', bins=80, kde=True, alpha=0.6)
    plt.title('r 波段流量分布对比 (TDE vs 非TDE)')
    plt.xlabel('Flux (baseline subtracted)')
    plt.savefig(os.path.join(PLOT_DIR, "02_flux_distribution_r.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: 02_flux_distribution_r.png")

    # 图3：典型光变曲线示例 (r 波段)
    plt.figure(figsize=(15, 8))
    tde_sample = object_labels[object_labels['is_tde']].sample(3, random_state=42)['object_id']
    non_tde_sample = object_labels[~object_labels['is_tde']].sample(3, random_state=42)['object_id']

    colors = ['#d62728', '#1f77b4', '#2ca02c']
    for i, oid in enumerate(tde_sample):
        sub = df[(df['object_id'] == oid) & (df['filter'] == 'r')]
        plt.plot(sub['mjd'], sub['flux_interp'], label=f'TDE {str(oid)[:8]}...',
                 color=colors[i], linewidth=2.2)

    for i, oid in enumerate(non_tde_sample):
        sub = df[(df['object_id'] == oid) & (df['filter'] == 'r')]
        plt.plot(sub['mjd'], sub['flux_interp'], label=f'Non-TDE {str(oid)[:8]}...',
                 color=colors[i], linestyle='--', linewidth=1.8)

    plt.title('典型 TDE 与非TDE 对象的光变曲线示例 (r 波段)')
    plt.xlabel('MJD')
    plt.ylabel('Flux (baseline subtracted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "03_example_lightcurves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: 03_example_lightcurves.png")

    # 图4：不同波段平均流量对比
    mean_flux = df.groupby(['filter', 'is_tde'])['flux_interp'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='filter', y='flux_interp', hue='is_tde', data=mean_flux)
    plt.title('不同波段平均流量对比 (TDE vs 非TDE)')
    plt.ylabel('Mean Flux')
    plt.savefig(os.path.join(PLOT_DIR, "04_mean_flux_per_filter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: 04_mean_flux_per_filter.png")

    print(f"\n✅ EDA 完成！所有图表已保存至: {PLOT_DIR}")

if __name__ == "__main__":
    main()