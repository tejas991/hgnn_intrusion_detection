# Dataset Upload and Placement

print("\nLoading Dataset")

# We'll use Colab's file upload tool for this
from google.colab import files
import os
import gc  # probably needed later
from sklearn.model_selection import train_test_split

print("Upload the NSL-KDD files:")
print("- Prefer 'KDDTrain+_20Percent.txt' over full 'KDDTrain+.txt'")
print("- You'll need both training and test sets: 'KDDTrain+.txt' and 'KDDTest+.txt'")

uploaded_files = files.upload()  # This opens up the file picker in Colab

# Tejas --> i want to organize in raw folder

os.makedirs("data/raw", exist_ok=True)

for fname in uploaded_files:
    original_path = fname
    target_path = f"data/raw/{fname}"
    os.rename(original_path, target_path)
    size_in_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"Saved {fname} to {target_path} ({size_in_mb:.2f} MB)")

print("\nScanning raw data folder")
raw_files = os.listdir('data/raw')
print("Files detected:", raw_files)

# Decide which training file to use
if any('20Percent' in f for f in raw_files):
    # we can use the smaller version
    train_filename = [f for f in raw_files if '20Percent' in f][0]
    max_train_samples = 30000
    print(f"20% training file detected: '{train_filename}' (we'll use up to {max_train_samples} samples)")
else:
    # Full Version
    train_filename = [f for f in raw_files if 'Train' in f and '20Percent' not in f][0]
    max_train_samples = 25000
    print(f"Full training file: '{train_filename}' (capped at {max_train_samples} samples)")

# Always cap test samples too
max_test_samples = 15000


print("\nLoading training set")
train_data_path = f"data/raw/{train_filename}"
X_train_all, y_train_all = processor.process_data_smart(
    train_data_path,
    is_training=True,
    max_samples=max_train_samples
)

print("Training set loaded")

# Find the test file (assuming only one has 'Test' in its name)
test_filename = [f for f in raw_files if 'Test' in f][0]
print(f"\nLoading test set: {test_filename}")
X_test, y_test = processor.process_data_smart(
    f"data/raw/{test_filename}",
    is_training=False,
    max_samples=max_test_samples
)

# Display shapes for sanity check
print(f"\nTrain data shape: {X_train_all.shape}")
print(f"Test data shape: {X_test.shape}")


print("\nSplitting train into training and validation sets")

# 80/20 stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all,
    test_size=0.2,
    stratify=y_train_all,
    random_state=42  # just so it's repeatable
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

#Cleanup

# Done with the full train arrays, clear them out to save memory
del X_train_all, y_train_all
gc.collect()

print("\nDataset loading complete")
