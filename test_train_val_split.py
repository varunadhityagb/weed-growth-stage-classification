import os
import shutil
from sklearn.model_selection import train_test_split

# Set paths
SRC_ROOT = './data_filtered_by_weeks'  # Your filtered dataset directory
DST_ROOT = './data_final_split'

# Define split ratios
TRAIN_RATIO = 0.60 
VAL_RATIO = 0.20    
TEST_RATIO = 0.20   

# Create output directories
os.makedirs(DST_ROOT, exist_ok=True)

train_dir = os.path.join(DST_ROOT, 'train')
val_dir = os.path.join(DST_ROOT, 'val')
test_dir = os.path.join(DST_ROOT, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Statistics tracking
split_stats = {}

# Get all classes (weed_week combinations)
classes = [d for d in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, d))]

print("Splitting dataset with stratification...")

for cls in classes:
    class_src_path = os.path.join(SRC_ROOT, cls)
    images = [f for f in os.listdir(class_src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Skip classes with too few images
    if len(images) < 5:
        print(f"Warning: {cls} has only {len(images)} images, skipping...")
        continue
    
    # Split images into train+val and test first
    train_val_imgs, test_imgs = train_test_split(
        images, test_size=TEST_RATIO, random_state=42, shuffle=True
    )
    
    # Then split train+val into train and val
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_imgs, val_imgs = train_test_split(
        train_val_imgs, test_size=val_size_adjusted, random_state=42, shuffle=True
    )
    
    # Create class directories for each split
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
    
    # Copy files to respective directories
    for img in train_imgs:
        shutil.copy2(os.path.join(class_src_path, img), os.path.join(train_dir, cls, img))
    
    for img in val_imgs:
        shutil.copy2(os.path.join(class_src_path, img), os.path.join(val_dir, cls, img))
    
    for img in test_imgs:
        shutil.copy2(os.path.join(class_src_path, img), os.path.join(test_dir, cls, img))
    
    # Track statistics
    split_stats[cls] = {
        'total': len(images),
        'train': len(train_imgs),
        'val': len(val_imgs),
        'test': len(test_imgs)
    }

# Print split statistics
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

print(f"\nDataset split completed! Check: {DST_ROOT}")
