import os

# IMPORTANT: Update this with the path to your main dataset folder
dataset_path = 'New Plant Diseases Dataset/'
train_dir = os.path.join(dataset_path, 'train')
valid_dir = os.path.join(dataset_path, 'valid')

# Let's get the class names from the folder structure
classes = sorted(os.listdir(train_dir))
if '.DS_Store' in classes: # For Mac users
    classes.remove('.DS_Store')

print("--- Class Distribution Check ---\n")

# Function to count files in each class directory
def count_files(directory):
    print(f"Checking directory: {directory}\n")
    class_counts = {}
    for plant_class in classes:
        class_path = os.path.join(directory, plant_class)
        if os.path.isdir(class_path):
            num_files = len(os.listdir(class_path))
            class_counts[plant_class] = num_files
            print(f" - {plant_class}: {num_files} files")
    return class_counts

# Count files in the training set
train_counts = count_files(train_dir)
print("\n" + "="*30 + "\n")

# Count files in the validation set
valid_counts = count_files(valid_dir)
print("\n" + "="*30 + "\n")

print("--- Summary ---")
total_train_files = sum(train_counts.values())
total_valid_files = sum(valid_counts.values())
print(f"Total training files: {total_train_files}")
print(f"Total validation files: {total_valid_files}")
