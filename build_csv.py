"""
Build training CSV file from baseline and regressed screenshots.

This script:
1. Scans dataset/baseline/ and dataset/regressed/ directories
2. Creates matched pairs (baseline + regressed = label 1)
3. Creates clean pairs (baseline + baseline = label 0)
4. Balances the dataset
5. Saves to dataset/pairs.csv

Usage:
    python build_csv.py
"""

from pathlib import Path
import csv
import random

# ============================================
# CONFIGURATION
# ============================================
BASELINE_DIR = Path("dataset/baseline")
REGRESSED_DIR = Path("dataset/regressed")
OUTPUT_CSV = Path("dataset/pairs.csv")

# Balance settings
BALANCE_DATASET = True  # Balance positive/negative examples
MAX_NEGATIVE_RATIO = 2.0  # Max ratio of negative to positive (2.0 = 2:1)

# Shuffle settings
SHUFFLE_PAIRS = True  # Randomize order of pairs
RANDOM_SEED = 42  # For reproducibility


# ============================================


def get_image_files(directory):
    """Get all image files from directory."""
    if not directory.exists():
        return []

    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))

    return sorted(files)


def create_pairs():
    """Create training pairs from baseline and regressed images."""

    print("=" * 70)
    print("📋 BUILDING TRAINING CSV")
    print("=" * 70)

    # Check directories exist
    if not BASELINE_DIR.exists():
        print(f"❌ ERROR: {BASELINE_DIR} not found!")
        print("   Run: python generate_dataset.py first")
        return False

    if not REGRESSED_DIR.exists():
        print(f"❌ ERROR: {REGRESSED_DIR} not found!")
        print("   Run: python generate_dataset.py first")
        return False

    # Get all image files
    baseline_files = get_image_files(BASELINE_DIR)
    regressed_files = get_image_files(REGRESSED_DIR)

    print(f"\n📁 Found {len(baseline_files)} baseline images in {BASELINE_DIR}")
    print(f"📁 Found {len(regressed_files)} regressed images in {REGRESSED_DIR}")

    if len(baseline_files) == 0:
        print("\n❌ ERROR: No baseline images found!")
        print("   Run: python generate_dataset.py to create screenshots")
        return False

    # Create lookup dict for regressed files
    regressed_dict = {f.name: f for f in regressed_files}

    # Generate pairs
    positive_pairs = []  # label=1: baseline vs regressed (different)
    negative_pairs = []  # label=0: baseline vs baseline (same)

    print(f"\n🔍 Matching baseline and regressed images...")

    matched_count = 0
    unmatched_count = 0

    for baseline_file in baseline_files:
        # Check if there's a matching regressed version
        if baseline_file.name in regressed_dict:
            # Create positive pair (regression detected)
            regressed_file = regressed_dict[baseline_file.name]
            positive_pairs.append({
                'baseline': baseline_file.as_posix(),
                'candidate': regressed_file.as_posix(),
                'label': 1
            })
            matched_count += 1
            print(f"  ✅ Matched: {baseline_file.name}")
        else:
            unmatched_count += 1
            print(f"  ⚠️  No match: {baseline_file.name}")

        # Always create negative pair (same image = no regression)
        negative_pairs.append({
            'baseline': baseline_file.as_posix(),
            'candidate': baseline_file.as_posix(),
            'label': 0
        })

    print(f"\n📊 Pair Statistics:")
    print(f"   Matched pairs: {matched_count}")
    print(f"   Unmatched baselines: {unmatched_count}")
    print(f"   Positive pairs (regressions): {len(positive_pairs)}")
    print(f"   Negative pairs (clean): {len(negative_pairs)}")

    # Validate dataset
    if len(positive_pairs) == 0:
        print("\n⚠️  WARNING: No positive pairs (regressions) found!")
        print("   This means baseline and regressed folders have no matching filenames.")
        print("   Did you run generate_dataset.py correctly?")
        return False

    # Balance dataset if needed
    if BALANCE_DATASET and len(negative_pairs) > 0:
        max_negatives = int(len(positive_pairs) * MAX_NEGATIVE_RATIO)

        if len(negative_pairs) > max_negatives:
            print(f"\n⚖️  Balancing dataset...")
            print(f"   Original negatives: {len(negative_pairs)}")
            print(f"   Max allowed (ratio {MAX_NEGATIVE_RATIO}:1): {max_negatives}")

            # Randomly sample negatives
            random.seed(RANDOM_SEED)
            negative_pairs = random.sample(negative_pairs, max_negatives)
            print(f"   Downsampled to: {len(negative_pairs)}")

    # Combine all pairs
    all_pairs = positive_pairs + negative_pairs

    # Shuffle if requested
    if SHUFFLE_PAIRS:
        random.seed(RANDOM_SEED)
        random.shuffle(all_pairs)
        print(f"\n🔀 Shuffled {len(all_pairs)} pairs")

    # Calculate statistics
    total_pairs = len(all_pairs)
    pos_count = len(positive_pairs)
    neg_count = len(negative_pairs)
    pos_ratio = (pos_count / total_pairs * 100) if total_pairs > 0 else 0
    neg_ratio = (neg_count / total_pairs * 100) if total_pairs > 0 else 0

    print(f"\n📊 Final Dataset:")
    print(f"   Total pairs: {total_pairs}")
    print(f"   Positive (label=1): {pos_count} ({pos_ratio:.1f}%)")
    print(f"   Negative (label=0): {neg_count} ({neg_ratio:.1f}%)")
    print(f"   Balance ratio: {neg_count / pos_count:.2f}:1" if pos_count > 0 else "")

    # Write to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['baseline', 'candidate', 'label'])
        writer.writeheader()
        writer.writerows(all_pairs)

    print(f"\n💾 CSV saved to: {OUTPUT_CSV}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if total_pairs < 50:
        print(f"   ⚠️  Only {total_pairs} pairs - need at least 50 for decent training")
        print("   Run: python generate_dataset.py --samples 20")
    elif total_pairs < 100:
        print(f"   ⚠️  Only {total_pairs} pairs - consider adding more")
        print("   Run: python generate_dataset.py --samples 30")
    else:
        print(f"   ✅ Dataset size is good ({total_pairs} pairs)")

    if pos_count < 20:
        print(f"   ⚠️  Only {pos_count} regression examples")
        print("   Need more regressed screenshots")
    else:
        print(f"   ✅ Sufficient regression examples ({pos_count})")

    if pos_count > 0:
        ratio = neg_count / pos_count
        if ratio > 3:
            print(f"   ⚠️  Dataset imbalanced ({ratio:.1f}:1 negative:positive)")
            print("   Consider reducing negative examples or adding more positive")
        elif ratio < 0.5:
            print(f"   ⚠️  Dataset imbalanced ({ratio:.1f}:1 negative:positive)")
            print("   Consider adding more negative examples")
        else:
            print(f"   ✅ Dataset is well balanced ({ratio:.1f}:1)")

    # Expected train/val split
    train_size = int(0.8 * total_pairs)
    val_size = total_pairs - train_size

    print(f"\n📈 Expected Train/Val Split (80/20):")
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {val_size}")

    if val_size < 10:
        print(f"   ⚠️  Validation set very small ({val_size} samples)")
        print("   Results may be unreliable - consider adding more data")
    elif val_size < 20:
        print(f"   ⚠️  Validation set small ({val_size} samples)")
        print("   Consider adding more data for reliable metrics")
    else:
        print(f"   ✅ Validation set size is good ({val_size} samples)")

    print("\n" + "=" * 70)
    print("✅ CSV BUILD COMPLETE!")
    print("=" * 70)
    print("\n🎯 Next steps:")
    print("   1. Review the CSV: cat dataset/pairs.csv | head -20")
    print("   2. Train model: python train.py")
    print("   3. Test predictions: python predict.py")
    print("=" * 70 + "\n")

    return True


def preview_csv(num_rows=10):
    """Preview the generated CSV file."""
    if not OUTPUT_CSV.exists():
        print(f"❌ CSV file not found: {OUTPUT_CSV}")
        return

    print(f"\n📄 Preview of {OUTPUT_CSV} (first {num_rows} rows):")
    print("=" * 70)

    with OUTPUT_CSV.open('r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            if i > num_rows:
                break

            baseline_name = Path(row['baseline']).name
            candidate_name = Path(row['candidate']).name
            label = row['label']
            label_text = "REGRESSION" if label == '1' else "CLEAN"

            print(f"{i:2d}. [{label_text:10s}] {baseline_name} + {candidate_name}")

    print("=" * 70)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Build training CSV from screenshots')
    parser.add_argument('--preview', action='store_true',
                        help='Preview the CSV after building')
    parser.add_argument('--no-balance', action='store_true',
                        help='Disable dataset balancing')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Disable shuffling of pairs')
    parser.add_argument('--max-negative-ratio', type=float, default=2.0,
                        help='Maximum ratio of negative to positive examples (default: 2.0)')

    args = parser.parse_args()

    # Update global settings
    global BALANCE_DATASET, SHUFFLE_PAIRS, MAX_NEGATIVE_RATIO
    if args.no_balance:
        BALANCE_DATASET = False
    if args.no_shuffle:
        SHUFFLE_PAIRS = False
    if args.max_negative_ratio:
        MAX_NEGATIVE_RATIO = args.max_negative_ratio

    # Build CSV
    success = create_pairs()

    # Preview if requested
    if success and args.preview:
        preview_csv()


if __name__ == "__main__":
    main()