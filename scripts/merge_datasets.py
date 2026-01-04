import os
import shutil
import glob
import re

def create_dir_structure(base_path):
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(base_path, split, subdir), exist_ok=True)

def get_class_max_indices(directory):
    """
    Scans a directory for files named [Class]_[Number].jpg and returns 
    a dictionary {Class: MaxIndex}
    """
    class_max_indices = {}
    if not os.path.exists(directory):
        return class_max_indices
        
    for filename in os.listdir(directory):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.txt')):
            continue
            
        # Parse filename: Class_Number.ext
        match = re.match(r"([A-Z])_(\d+)\.", filename)
        if match:
            class_name = match.group(1)
            index = int(match.group(2))
            
            if class_name not in class_max_indices:
                class_max_indices[class_name] = 0
            if index > class_max_indices[class_name]:
                class_max_indices[class_name] = index
    
    return class_max_indices

def merge_datasets(dataset1_path, dataset2_path, output_path):
    create_dir_structure(output_path)
    
    # 1. Copy dataset1 (Base)
    print("Copying dataset1 (Base)...")
    for split in ['train', 'valid', 'test']:
        src_images = os.path.join(dataset1_path, split, 'images')
        src_labels = os.path.join(dataset1_path, split, 'labels')
        
        dst_images = os.path.join(output_path, split, 'images')
        dst_labels = os.path.join(output_path, split, 'labels')
        
        if os.path.exists(src_images):
            for f in os.listdir(src_images):
                shutil.copy2(os.path.join(src_images, f), os.path.join(dst_images, f))
        
        if os.path.exists(src_labels):
            for f in os.listdir(src_labels):
                shutil.copy2(os.path.join(src_labels, f), os.path.join(dst_labels, f))

    # 2. Scan for max indices in output_path (which now contains dataset1)
    # Scan all splits to get global max index per class
    print("Scanning for current max indices...")
    global_max_indices = {}
    for split in ['train', 'valid', 'test']:
        split_indices = get_class_max_indices(os.path.join(output_path, split, 'images'))
        for cls, idx in split_indices.items():
            if cls not in global_max_indices:
                global_max_indices[cls] = 0
            if idx > global_max_indices[cls]:
                global_max_indices[cls] = idx
    
    print(f"Current max indices: {global_max_indices}")

    # 3. Merge and Rename dataset2
    print("Merging dataset2 with renaming...")
    # dataset2 likely has just 'train', 'valid', 'test' with images/labels inside?
    # Checking custom_dataset_labelled structure...
    # Based on previous `list_dir`, it has `train`, `valid`, `test`.
    # And inside those `images` and `labels`? 
    # Let's assume standard structure. If not, we might need to adjust.
    # The previous `list_dir` of `custom_dataset_labelled` showed `train`, `valid`, `test`.
    # Let's verify if `images` and `labels` are inside `train`. 
    # Assuming standard YOLO export structure.
    
    for split in ['train', 'valid', 'test']:
        src_images_dir = os.path.join(dataset2_path, split, 'images')
        src_labels_dir = os.path.join(dataset2_path, split, 'labels')
        
        if not os.path.exists(src_images_dir):
            continue
            
        # Get list of images to process
        images = [f for f in os.listdir(src_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Sort to insure deterministic order
        images.sort()
        
        for image_file in images:
            # Parse class from filename (e.g., A_0001.jpg -> A)
            match = re.match(r"([A-Z])_", image_file)
            if not match:
                print(f"Skipping {image_file}: Pattern mismatch")
                continue
                
            class_name = match.group(1)
            
            # Increment index
            if class_name not in global_max_indices:
                global_max_indices[class_name] = 0
            
            global_max_indices[class_name] += 1
            new_index = global_max_indices[class_name]
            
            new_image_name = f"{class_name}_{new_index:04d}.jpg"
            
            # Copy Image
            src_img_path = os.path.join(src_images_dir, image_file)
            dst_img_path = os.path.join(output_path, split, 'images', new_image_name)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Copy Label
            label_file = os.path.splitext(image_file)[0] + '.txt'
            src_label_path = os.path.join(src_labels_dir, label_file)
            new_label_name = f"{class_name}_{new_index:04d}.txt"
            dst_label_path = os.path.join(output_path, split, 'labels', new_label_name)
            
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)
            else:
                print(f"Warning: Label missing for {image_file}")

    # 4. Create data.yaml
    print("Creating data.yaml...")
    # Copy from dataset1 and update path if needed
    src_yaml = os.path.join(dataset1_path, 'data.yaml')
    dst_yaml = os.path.join(output_path, 'data.yaml')
    if os.path.exists(src_yaml):
        with open(src_yaml, 'r') as f:
            content = f.read()
        
        # Ensure paths are relative and correct for the new location if needed
        # Or just keep standard 'train: ../train/images' etc if structure matches
        with open(dst_yaml, 'w') as f:
            f.write(content)
            
    print("Merge Complete!")

if __name__ == "__main__":
    base_dir = r"d:\Personal\Thesis\ASL-to-Voice-web"
    ds1 = os.path.join(base_dir, "dataset")
    ds2 = os.path.join(base_dir, "custom_dataset_labelled")
    out = os.path.join(base_dir, "combined_dataset")
    
    merge_datasets(ds1, ds2, out)
