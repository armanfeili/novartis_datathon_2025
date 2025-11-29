import sys
sys.path.insert(0, 'src')
from data_loader import load_all_data, merge_datasets, split_train_validation
from bucket_calculator import create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns
from models import BaselineModels
from evaluation import compute_pe_scenario1

# Load and prepare
v,g,m = load_all_data(train=True)
merged = merge_datasets(v,g,m)
aux_df = create_auxiliary_file(merged, save=False)
avg_j = aux_df[['country','brand_name','avg_vol']].copy()

print("\n" + "="*60)
print("FULL DATASET BUCKET DISTRIBUTION")
print("="*60)
print(aux_df['bucket'].value_counts())

# Feature eng
featured = create_all_features(merged, avg_j)

# Split
train_df, val_df = split_train_validation(featured)

# Check buckets in validation
val_brands = val_df[['country','brand_name']].drop_duplicates()
val_aux = aux_df.merge(val_brands, on=['country','brand_name'])
print("\n" + "="*60)
print("VALIDATION BUCKET DISTRIBUTION")
print("="*60)
print(val_aux['bucket'].value_counts())

# Generate predictions (baseline)
val_avg_j = avg_j.merge(val_brands, on=['country','brand_name'])
pred = BaselineModels.exponential_decay(val_avg_j, list(range(0,24)), 0.02)

# Get actuals - THIS IS KEY
val_actual = val_df[val_df['months_postgx'].isin(range(0,24))][['country','brand_name','months_postgx','volume']].copy()
print("\n" + "="*60)
print("VALIDATION ACTUAL DATA")
print("="*60)
print(f"Val actual rows: {len(val_actual)}")
actual_brands = val_actual[['country','brand_name']].drop_duplicates()
print(f"Val actual unique brands: {len(actual_brands)}")

# Check what months we have in val_actual
print(f"\nMonths in val_actual: {sorted(val_actual['months_postgx'].unique())}")

# Compute PE
pe_df = compute_pe_scenario1(val_actual, pred, aux_df)
print("\n" + "="*60)
print("PE DATAFRAME BUCKET DISTRIBUTION")
print("="*60)
print(pe_df['bucket'].value_counts())
print(f"PE df length: {len(pe_df)}")

# Check specifically which bucket 1 brands made it through
print("\n" + "="*60)
print("BUCKET 1 ANALYSIS")
print("="*60)
bucket1_brands = val_aux[val_aux['bucket']==1][['country','brand_name']]
print(f"Bucket 1 brands in validation: {len(bucket1_brands)}")

# Check if these bucket 1 brands are in val_actual
bucket1_in_actual = val_actual.merge(bucket1_brands, on=['country','brand_name'])
print(f"Bucket 1 brands with actual data (0-23): {len(bucket1_in_actual[['country','brand_name']].drop_duplicates())}")

# Check if these bucket 1 brands are in predictions
bucket1_in_pred = pred.merge(bucket1_brands, on=['country','brand_name'])
print(f"Bucket 1 brands in predictions: {len(bucket1_in_pred[['country','brand_name']].drop_duplicates())}")

# Check months for bucket 1 brands in val_actual
bucket1_months = bucket1_in_actual['months_postgx'].unique()
print(f"Months available for bucket 1 brands: {sorted(bucket1_months)}")
