import os
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa

def augment_images(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    # Define augmentation sequence
    # 1. 50% probability of horizontal flip
    # 2. Randomly crop between 0 and 20 percent of the image (keep_size=True implies resize back to original)
    # 3. Random rotation of between -5 and +5 degrees
    # 4. Random shear of between -5deg to +5deg horizontally and -5deg to +5deg vertically
    # 5. Random brightness adjustment of between -25 and +25 percent
    # 6. Random Gaussian blur of between 0 and 1.25 pixels
    
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Crop(percent=(0, 0.2), keep_size=True),
        iaa.Affine(
            rotate=(-5, 5),
            shear={"x": (-5, 5), "y": (-5, 5)},
            mode='edge' # Handling empty pixels from rotation/shear
        ),
        iaa.Multiply((0.75, 1.25)),
        iaa.GaussianBlur(sigma=(0, 1.25))
    ])

    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Get list of files first
    files = [f for f in os.listdir(directory) if os.path.splitext(f.lower())[1] in supported_extensions]
    
    total_files = len(files)
    print(f"Found {total_files} source images. Creating 3 augmented versions for each (total {total_files * 3} new images).")

    for i, filename in enumerate(files):
        # Skip files that look like they are already augmented (if running partially)
        # But here we assume clean state or just appending.
        if "_aug_" in filename:
            continue

        filepath = os.path.join(directory, filename)
        file_root, file_ext = os.path.splitext(filename)
        
        try:
            # Load image and convert to RGB (imgaug expects RGB or Gray)
            img = Image.open(filepath).convert('RGB')
            image_np = np.array(img)
            
            # Generate 3 augmented versions
            # we call augment_images multiple times
            images_aug = seq(images=[image_np for _ in range(3)])

            for j, img_aug_np in enumerate(images_aug):
                img_aug = Image.fromarray(img_aug_np)
                new_filename = f"{file_root}_aug_{j+1}{file_ext}"
                new_filepath = os.path.join(directory, new_filename)
                img_aug.save(new_filepath)
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total_files} source images...")
                
        except Exception as e:
            print(f"Failed to process '{filename}': {e}")

    print("Augmentation complete.")

if __name__ == "__main__":
    dataset_dir = r"d:\Personal\Thesis\ASL-to-Voice-web\custom dataset"
    augment_images(dataset_dir)
