import os
import shutil

def filter_and_organize_by_weeks(src_root, dst_root, selected_weeds):
    """
    Filter dataset to include only selected weeds and organize by weeks
    
    Args:
        src_root: Source directory (e.g., 'data_chopped_random')
        dst_root: Destination directory (e.g., 'data_filtered_by_weeks')
        selected_weeds: List of weed names to include
    """
    
    # Create destination directory
    os.makedirs(dst_root, exist_ok=True)
    
    # Track statistics
    stats = {}
    
    # Process each selected weed
    for weed_name in selected_weeds:
        src_weed_path = os.path.join(src_root, weed_name)
        
        if not os.path.exists(src_weed_path):
            print(f"Warning: {weed_name} not found in source directory")
            continue
            
        print(f"Processing {weed_name}...")
        stats[weed_name] = {}
        
        # Get all images for this weed
        images = [f for f in os.listdir(src_weed_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Group images by week
        images_by_week = {}
        
        for img in images:
            parts = img.split('_')
            if len(parts) >= 3:
                week = parts[2]  # Extract week (e.g., 'w0', 'w1', etc.)
                
                if week not in images_by_week:
                    images_by_week[week] = []
                images_by_week[week].append(img)
        
        # Create directories and copy images for each week
        for week, week_images in images_by_week.items():
            combined_class = f"{weed_name}_{week}"
            dst_class_path = os.path.join(dst_root, combined_class)
            os.makedirs(dst_class_path, exist_ok=True)
            
            # Copy all images for this weed-week combination
            for img in week_images:
                src_img_path = os.path.join(src_weed_path, img)
                dst_img_path = os.path.join(dst_class_path, img)
                shutil.copy2(src_img_path, dst_img_path)
            
            stats[weed_name][week] = len(week_images)
            print(f"  {combined_class}: {len(week_images)} images")
    
    return stats

# Configuration
SRC_ROOT = './data'
DST_ROOT = './data_filtered_by_weeks'

# ========== MODIFY THIS ARRAY TO SELECT YOUR WEEDS ==========
SELECTED_WEEDS = [
    'asian-flatsedge',
    'asiatic-dayflower',
    'indian-goosegrass',
    'korean-dock',
    'nipponicus-sedge'
    ]
# ============================================================

# Run the filtering and organization
print("Starting dataset filtering and week-based organization...")
statistics = filter_and_organize_by_weeks(SRC_ROOT, DST_ROOT, SELECTED_WEEDS)

# Print summary statistics
print("\n" + "="*50)
print("DATASET SUMMARY")
print("="*50)

total_images = 0
total_classes = 0

for weed_name, weeks in statistics.items():
    print(f"\n{weed_name.upper()}:")
    weed_total = 0
    for week, count in sorted(weeks.items()):
        print(f"  {weed_name}_{week}: {count} images")
        weed_total += count
        total_classes += 1
    print(f"  Total for {weed_name}: {weed_total} images")
    total_images += weed_total

print(f"\n" + "="*50)
print(f"OVERALL SUMMARY:")
print(f"Total classes (weed_week combinations): {total_classes}")
print(f"Total images: {total_images}")
print(f"Average images per class: {total_images/total_classes:.1f}")
print("="*50)

print(f"\nFiltered dataset saved to: {DST_ROOT}")
print("Ready for train/test/val splitting!")
