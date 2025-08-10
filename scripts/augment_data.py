import os
import cv2
import random
import albumentations as A
from tqdm import tqdm

# SETTINGS
train_dir = './data_final_split/train'
min_count_threshold = 100
target_count = 250

# Define augmentations
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=25, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.2),
])

# Track summary
summary = []

# Auto-select and augment classes with fewer than `min_count_threshold` images
for class_name in sorted(os.listdir(train_dir)):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    current_count = len(images)

    if current_count >= min_count_threshold:
        continue  # Skip well-populated classes

    needed = target_count - current_count
    if needed <= 0:
        continue

    for i in range(needed):
        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read: {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = augment(image=image)['image']
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        base_name, ext = os.path.splitext(img_name)
        aug_name = f"{base_name}_aug{i}{ext}"
        cv2.imwrite(os.path.join(class_path, aug_name), augmented)

    # Log summary
    final_count = len(os.listdir(class_path))
    summary.append({
        'class': class_name,
        'before': current_count,
        'added': needed,
        'after': final_count
    })

# Final summary
print("\nAUGMENTATION SUMMARY")
print("=" * 50)
for entry in summary:
    print(f"{entry['class']}:")
    print(f"   Before: {entry['before']}")
    print(f"   Added:  {entry['added']}")
    print(f"   After:  {entry['after']}")
print("=" * 50)
print("Auto-targeted augmentation complete!")
