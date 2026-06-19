import os
from pathlib import Path

train_path = "D:\\Medical Image Analysis\\dataset\\Training"
test_path = "D:\\Medical Image Analysis\\dataset\\Testing"

print("Checking paths...")
print(f"Train path exists: {os.path.exists(train_path)}")
print(f"Test path exists: {os.path.exists(test_path)}")

# Count files
if os.path.exists(train_path):
    for root, dirs, files in os.walk(train_path):
        print(f"{root}: {len(files)} files, {len(dirs)} dirs")

print("\nImporting torch libraries...")
from torchvision import datasets, transforms
print("Imports successful!")

print("\nCreating transform...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("Loading training dataset...")
train_dataset = datasets.ImageFolder(
    root=train_path,
    transform=transform
)

print(f"Train dataset loaded! Classes: {train_dataset.class_to_idx}")
