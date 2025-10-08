import pandas as pd

df = pd.read_csv('dataset/pairs.csv')

print("="*60)
print("DATASET DIAGNOSIS")
print("="*60)
print(f"\nTotal pairs: {len(df)}")
print(f"\nClass distribution:")
print(df['label'].value_counts())
print(f"\nLabel balance:")
print(df['label'].value_counts(normalize=True))

# Calculate expected train/val split
train_size = int(0.8 * len(df))
val_size = len(df) - train_size

print(f"\nExpected split (80/20):")
print(f"  Training: {train_size} samples")
print(f"  Validation: {val_size} samples")
print(f"\n⚠️  WARNING: If validation < 10 samples, results are unreliable!")

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Check for same-image pairs
same_pairs = df[df['baseline'] == df['candidate']]
print(f"\nSame-image pairs: {len(same_pairs)} ({len(same_pairs)/len(df)*100:.1f}%)")