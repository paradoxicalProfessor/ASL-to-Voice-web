import os
import glob

def rename_dataset(directory_path):
    """
    Renames images in the directory following the convention [Class]_[Number].jpg
    e.g., A_0001.jpg, B_0001.jpg
    """
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    # Get all files
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    # Filter for image extensions if necessary (assuming user wants all images)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in files if f.lower().endswith(image_extensions)]

    # Group files by class prefix (assumes prefix is part before first '_')
    # If no underscore, maybe take first character? 
    # Looking at the file list: "A_0070.jpeg", "B_0109.jpg" -> Prefix is A, B.
    
    files_by_class = {}
    
    for filename in files:
        parts = filename.split('_')
        if len(parts) >= 2:
            class_prefix = parts[0]
        else:
            # Fallback if no underscore, perhaps just use the first letter or handle as 'Unclassified'
            # But the user's files seem to all have underscores based on the dump.
            # Let's check for files like "A0070.jpg" if they exist, but the list showed A_...
            class_prefix = filename[0] # Fallback to first char if splitting fails? 
            # Actually, looking at "A_0070.jpeg", split('_') -> ['A', '0070.jpeg'] -> prefix 'A'.
            pass

        if class_prefix not in files_by_class:
            files_by_class[class_prefix] = []
        files_by_class[class_prefix].append(filename)

    # Sort keys to process in order
    sorted_classes = sorted(files_by_class.keys())

    for class_prefix in sorted_classes:
        class_files = files_by_class[class_prefix]
        # Sort files to ensure deterministic renaming (optional: natural sort?)
        class_files.sort() 
        
        print(f"Processing class '{class_prefix}' with {len(class_files)} images...")
        
        for index, filename in enumerate(class_files, start=1):
            # Construct new filename
            new_filename = f"{class_prefix}_{index:04d}.jpg"
            
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
            
            # Skip if name is already correct
            if filename == new_filename:
                continue

            # Handle case where new_filename already exists (collision)
            # This can happen if we are renaming A_0100 to A_0001 but A_0001 already exists (e.g. if we are re-running).
            # But here we are renaming to A_0001, A_0002... and current files are A_0070...
            # If we generate a name that conflicts with an *existing unprocessed* file, we might overwrite it.
            # E.g. Rename X -> Y, but Y exists and is scheduled to be renamed to Z later.
            # To be safe, we can rename to a temporary name first or check existence.
            # Given the sequential nature and the fact that we start from 1, and existing files start from 70, 
            # we are likely safe from collision with *unprocessed* files, UNLESS there is an overlap in ranges.
            # (e.g. if we had A_0001 and A_0002 already).
            # A safer approach: Rename all to a temp extension first, then rename back. 
            # OR, check if target exists.
            
            if os.path.exists(new_path) and new_filename not in [f for f in class_files[:index-1]]: 
                 # Only a problem if it's a file we haven't processed/renamed *into* yet?
                 # Actually, if new_path exists, it's either:
                 # 1. A file we just renamed (impossible, we are at current index)
                 # 2. A file that was already there.
                 
                 # Strategy: Rename to a temporary name, then doing a second pass would be safer.
                 # Let's do a simple temp rename for all, then final rename.
                 pass

    # Revised Strategy: Two-pass rename
    # 1. Rename all files in the group to `{OriginalName}_TEMP_RENAME`
    # 2. Rename all those temp files to target `{Class}_{Index}.jpg`
    # This prevents collisions.
    
    for class_prefix in sorted_classes:
        class_files = files_by_class[class_prefix]
        class_files.sort()
        
        # Pass 1: Add suffix to avoid collisions
        temp_files = []
        for filename in class_files:
            old_path = os.path.join(directory_path, filename)
            temp_name = f"{filename}.tmp_rename"
            temp_path = os.path.join(directory_path, temp_name)
            os.rename(old_path, temp_path)
            temp_files.append(temp_name)
        
        # Pass 2: Rename to target
        for index, temp_filename in enumerate(temp_files, start=1):
            old_path = os.path.join(directory_path, temp_filename)
            new_filename = f"{class_prefix}_{index:04d}.jpg"
            new_path = os.path.join(directory_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {temp_filename} -> {new_filename}")

if __name__ == "__main__":
    dataset_dir = r"d:\Personal\Thesis\ASL-to-Voice-web\custom_dataset"
    rename_dataset(dataset_dir)
