import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("Loading dataset configuration...")

train_path = "D:\\Medical Image Analysis\\dataset\\Training"
test_path = "D:\\Medical Image Analysis\\dataset\\Testing"

print(f"Train path: {train_path}")
print(f"Test path: {test_path}")

print("\nCreating image transforms...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("Loading training dataset...")
train_dataset = datasets.ImageFolder(
    root=train_path,
    transform=transform
)
print(f"✓ Training dataset loaded! Total samples: {len(train_dataset)}")

print("Loading testing dataset...")
test_dataset = datasets.ImageFolder(
    root=test_path,
    transform=transform
)
print(f"✓ Testing dataset loaded! Total samples: {len(test_dataset)}")

print("\nClass mappings:")
print(train_dataset.class_to_idx)

print("\nCreating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

print("\n✓ All datasets ready for training!")