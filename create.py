import pandas as pd
import os

# Path to your images folder
image_dir = 'new_training_set'

# Check if directory exists
if not os.path.isdir(image_dir):
    raise FileNotFoundError(f"The directory {image_dir} does not exist.")

# Create a dictionary to store filenames and their labels
labels_dict = {}

# Iterate through subfolders
for label in os.listdir(image_dir):
    subfolder_path = os.path.join(image_dir, label)
    
    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Get all image filenames in the subfolder
        image_filenames = [f for f in os.listdir(subfolder_path) if f.endswith(('.jpg', '.png'))]
        
        # Rename and process each image
        for index, filename in enumerate(image_filenames):
            old_image_path = os.path.join(subfolder_path, filename)
            new_filename = f"{label}_{index+1}.jpg"  # or .png based on the original file extension
            new_image_path = os.path.join(subfolder_path, new_filename)
            
            # Rename the image file
            os.rename(old_image_path, new_image_path)
            
            # Add to dictionary
            labels_dict[new_filename] = label

# Convert dictionary to DataFrame
df_labels = pd.DataFrame(list(labels_dict.items()), columns=['image_name', 'label'])

# Save DataFrame to CSV
csv_file_path = 'labels.csv'
df_labels.to_csv(csv_file_path, index=False)

print(f"{csv_file_path} file has been created.")
