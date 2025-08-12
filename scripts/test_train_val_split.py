import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def split_dataset(src_root, dst_root, train_ratio, val_ratio, test_ratio):
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-3:
        raise ValueError(f"Split ratios must sum to 1.0. Got {total_ratio}")

    os.makedirs(dst_root, exist_ok=True)

    train_dir = os.path.join(dst_root, 'train')
    val_dir = os.path.join(dst_root, 'val')
    test_dir = os.path.join(dst_root, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    split_stats = {}

    classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]

    print("Splitting dataset with stratification...")

    for cls in classes:
        class_src_path = os.path.join(src_root, cls)
        images = [f for f in os.listdir(class_src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) < 5:
            print(f"Warning: {cls} has only {len(images)} images, skipping...")
            continue

        train_val_imgs, test_imgs = train_test_split(
            images, test_size=test_ratio, random_state=42, shuffle=True
        )

        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_imgs, val_imgs = train_test_split(
            train_val_imgs, test_size=val_size_adjusted, random_state=42, shuffle=True
        )

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        for img in train_imgs:
            shutil.copy2(os.path.join(class_src_path, img), os.path.join(train_dir, cls, img))
        for img in val_imgs:
            shutil.copy2(os.path.join(class_src_path, img), os.path.join(val_dir, cls, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(class_src_path, img), os.path.join(test_dir, cls, img))

        split_stats[cls] = {
            'total': len(images),
            'train': len(train_imgs),
            'val': len(val_imgs),
            'test': len(test_imgs)
        }

    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)

    total_train = total_val = total_test = 0

    for cls, stats in split_stats.items():
        print(f"{cls}:")
        print(f"  Total: {stats['total']} | Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
        total_train += stats['train']
        total_val += stats['val']
        total_test += stats['test']

    print("="*60)
    print(f"OVERALL: Train: {total_train} | Val: {total_val} | Test: {total_test}")
    print(f"Ratios: Train: {total_train/(total_train+total_val+total_test):.2%} | Val: {total_val/(total_train+total_val+total_test):.2%} | Test: {total_test/(total_train+total_val+total_test):.2%}")

    print(f"\nDataset split completed! Check: {dst_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test folders")
    parser.add_argument('--src', type=str, default='./data_filtered_by_weeks', help='Source dataset root folder')
    parser.add_argument('--dst', type=str, default='./data_final_split', help='Destination folder for splits')
    parser.add_argument('--train', type=float, default=0.60, help='Train split ratio (e.g. 0.6)')
    parser.add_argument('--val', type=float, default=0.20, help='Validation split ratio (e.g. 0.2)')
    parser.add_argument('--test', type=float, default=0.20, help='Test split ratio (e.g. 0.2)')

    args = parser.parse_args()

    split_dataset(
        src_root=args.src,
        dst_root=args.dst,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test
    )
