from pathlib import Path
import csv

# Define paths
baseline_dir = Path("dataset/baseline")
regressed_dir = Path("dataset/regressed")
output_csv = Path("dataset/pairs.csv")

# Ensure baseline and regressed directories exist
assert baseline_dir.exists(), f"{baseline_dir} not found"
assert regressed_dir.exists(), f"{regressed_dir} not found"

# Generate pairs
pairs = []

baseline_files = sorted([f for f in baseline_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
regressed_files_set = set(f.name for f in regressed_dir.iterdir())

for file in baseline_files:
    if file.name in regressed_files_set:
        # Regressed example
        pairs.append((file.as_posix(), (regressed_dir / file.name).as_posix(), 1))
    else:
        # Clean example
        pairs.append((file.as_posix(), file.as_posix(), 0))

# Write to CSV
with output_csv.open("w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["baseline", "candidate", "label"])
    writer.writerows(pairs)

print(f"✅ Pairs CSV generated at: {output_csv}")
