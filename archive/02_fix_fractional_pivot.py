# 02_fix_fractional_pivot.py
import pandas as pd
import os

PROCESSED_DIR = "./processed"

print("正在修复分数阶特征为宽表...")

frac = pd.read_parquet(os.path.join(PROCESSED_DIR, "features_fractional.parquet"))

print(f"原始分数阶长表形状: {frac.shape}")

# 转为宽表（每个 object_id 一行）
pivot_frac = frac.pivot_table(
    index='object_id',
    values=[col for col in frac.columns if col != 'object_id'],
    aggfunc='first'
).reset_index()

pivot_frac.to_parquet(os.path.join(PROCESSED_DIR, "features_fractional_pivot.parquet"),
                      engine='pyarrow', index=False)

print(f"✅ 分数阶宽表生成完成！形状: {pivot_frac.shape}")
print(f"文件保存为: features_fractional_pivot.parquet")
print(f"object_id 数量: {pivot_frac['object_id'].nunique()}")