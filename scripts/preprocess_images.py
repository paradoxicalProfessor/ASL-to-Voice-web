import os
from PIL import Image, ImageOps

def preprocess_images(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    files = [f for f in os.listdir(directory) if os.path.splitext(f.lower())[1] in supported_extensions]
    total_files = len(files)
    print(f"Found {total_files} images in '{directory}'. Processing...")

    for i, filename in enumerate(files):
        filepath = os.path.join(directory, filename)
        
        try:
            with Image.open(filepath) as img:
                # 1. Auto-orientation of pixel data (with EXIF-orientation stripping)
                # ImageOps.exif_transpose returns a new image with the orientation applied
                # If no EXIF orientation, it returns the original image (or copy)
                transposed_img = ImageOps.exif_transpose(img)
                
                # 2. Resize to 416x416 (Stretch)
                # We use LANCZOS for high-quality downsampling
                resized_img = transposed_img.resize((416, 416), Image.Resampling.LANCZOS)
                
                # Save the processed image, overwriting the original
                # We don't verify EXIF is stripped, but exif_transpose applies it to pixels.
                # Saving without passing 'exif' argument usually strips it or writes a default one.
                resized_img.save(filepath)
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total_files} images...")
                
        except Exception as e:
            print(f"Failed to process '{filename}': {e}")

    print("Processing complete.")

if __name__ == "__main__":
    # Use raw string for Windows path or forward slashes
    dataset_dir = r"d:\Personal\Thesis\ASL-to-Voice-web\custom dataset"
    preprocess_images(dataset_dir)
