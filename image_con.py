import os
from PIL import Image

# Path to the folder containing images
image_folder = 'wigs_images'

# Function to convert images to JPEG format
def convert_images_to_jpg(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('png', 'jpeg', 'bmp', 'gif', 'tiff','webp')):
            file_path = os.path.join(folder_path, filename)
            try:
                # Open the image file
                with Image.open(file_path) as img:
                    # Define the new file name with .jpg extension
                    new_filename = os.path.splitext(filename)[0] + '.jpg'
                    new_file_path = os.path.join(folder_path, new_filename)
                    
                    # Convert image to RGB (JPEG doesn't support transparency)
                    img = img.convert('RGB')
                    
                    # Save the image in JPEG format
                    img.save(new_file_path, 'JPEG')
                    print(f"Converted {filename} to {new_filename}")
                    
                    # Optionally, remove the original file
                    # os.remove(file_path)
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Convert images
convert_images_to_jpg(image_folder)
