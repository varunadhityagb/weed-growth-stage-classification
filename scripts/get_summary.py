import os

# Path to dataset splits
dataset_dir = './data_final_split'  # Change if your split folder is named differently
splits = ['train', 'val', 'test']
valid_exts = ('.jpg', '.jpeg', '.png')

# Store counts
class_counts = {}

# Loop through splits
for split in splits:
    split_path = os.path.join(dataset_dir, split)

    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        image_count = len([f for f in os.listdir(class_path) if f.lower().endswith(valid_exts)])

        # Initialize dict for this class
        if class_name not in class_counts:
            class_counts[class_name] = {'train': 0, 'val': 0, 'test': 0, 'total': 0}

        class_counts[class_name][split] += image_count
        class_counts[class_name]['total'] += image_count

# Final summary
print("TOTAL CLASSES: ", len(class_counts))
print("\n SUMMARY PER CLASS")
print("=" * 60)
print(f"{'Class Name':30} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
print("-" * 60)
total_train = total_val = total_test = 0
for class_name, counts in sorted(class_counts.items()):
    t, v, ts, ttl = counts['train'], counts['val'], counts['test'], counts['total']
    total_train += t
    total_val += v
    total_test += ts
    print(f"{class_name:30} {t:6} {v:6} {ts:6} {ttl:6}")
print("-" * 60)
print(f"{'TOTAL':30} {total_train:6} {total_val:6} {total_test:6} {total_train + total_val + total_test:6}")
print("=" * 60)
