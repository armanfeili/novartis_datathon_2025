"""Quick test of pre-LOE features."""
import pandas as pd
from src.data import load_raw_data, prepare_base_panel, compute_pre_entry_stats, handle_missing_values
from src.features import make_features
from src.utils import load_config

config = load_config('configs/data.yaml')
raw = load_raw_data(config, 'train')
panel = prepare_base_panel(raw['volume'], raw['generics'], raw['medicine_info'])
panel = handle_missing_values(panel)
panel = compute_pre_entry_stats(panel, is_train=True)

df = make_features(panel.copy(), scenario=1, mode='train')

# List new features
pre_loe_cols = [c for c in df.columns if 'pre_loe' in c]
print(f"New pre-LOE features: {pre_loe_cols}")
print(f"Total features: {len(df.columns)}")
print()

# Pick a sample brand
sample_brand = df[['country', 'brand_name']].drop_duplicates().iloc[0]
sample = df[(df['country'] == sample_brand['country']) & (df['brand_name'] == sample_brand['brand_name'])]

print(f"Sample brand: {sample_brand.values}")
print(f"Pre-LOE rows (months < 0): {len(sample[sample['months_postgx'] < 0])}")
print(f"Post-LOE rows (months >= 0): {len(sample[sample['months_postgx'] >= 0])}")
print()

# Check that pre_loe features are SAME for all months (broadcast from pre-entry)
print("pre_loe_vol_mean values at different months (should be SAME for all rows):")
for m in [-12, -6, -1, 0, 6, 12, 23]:
    row = sample[sample['months_postgx'] == m]
    if len(row) > 0:
        val = row['pre_loe_vol_mean'].values[0]
        y_norm = row['y_norm'].values[0] if 'y_norm' in row.columns else 'N/A'
        print(f"  Month {m:3d}: pre_loe_vol_mean = {val:.4f}, y_norm = {y_norm}")

print()
print("Verification: These values should ALL be the same for 'pre_loe_vol_mean'")
print("because they are computed from pre-LOE period and broadcast to ALL rows.")
