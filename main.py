import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

from vgg import vgg_residual_blocks_V2

# Configuration

LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
EPOCHS = 100
NUM_CLASSES = 200 # Tiny ImageNet has 200 classes
IMAGE_SIZE = 64
SUBSET_SIZE = 0 # 0 means train on entire training set


# Data Loading & Transforms

# Standard normalization for ImageNet-trained models
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    normalize,
])

# Just resize and normalize for validation
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize,
])

# Custom wrapper to bridge HF `datasets` and PyTorch `DataLoader`
class HfDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        # Ensure image is 3-channel RGB
        image = item['image'].convert("RGB")
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# Training
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        with torch.autocast(device_type=device_str):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 99:  # Print every 100 batches
            batch_time = time.time() - start_time
            print(f'Train Epoch: {epoch} [{(batch_idx + 1) * BATCH_SIZE}/{len(train_loader.dataset)} ({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\t'
                  f'Loss: {running_loss / 100:.6f} | '
                  f'Acc: {100.*correct/total:.2f}% | '
                  f'Time: {batch_time:.2f}s')
            running_loss = 0.0
            start_time = time.time()

# Validation
def validate(epoch):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            with torch.autocast(device_type=device_str):
                output = model(data)
                val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    print(f'\nValidation Set: Epoch: {epoch}, Avg. Loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({val_acc:.2f}%)\n')
    
    return val_acc


if __name__ == '__main__':
    
    # Use GPU
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Load Dataset
    print("Loading Tiny ImageNet dataset from Hugging Face...")
    tiny_imagenet = load_dataset("zh-plus/tiny-imagenet")

    if SUBSET_SIZE != 0:
        indices = torch.randperm(len(tiny_imagenet['train'])).tolist()
        train_subset_indices = indices[:SUBSET_SIZE]
        train_subset = tiny_imagenet['train'].select(train_subset_indices)
        print(f"Using a training subset of {len(train_subset)} images.")
        train_data = HfDatasetWrapper(train_subset, train_transform)
    else:       
        train_data = HfDatasetWrapper(tiny_imagenet['train'], train_transform)
        
    val_data = HfDatasetWrapper(tiny_imagenet['valid'], val_transform) 

    # Create DataLoaders
    train_loader = DataLoader(train_data, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True, 
                            num_workers=4,  
                            pin_memory=True) 

    val_loader = DataLoader(val_data, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False, 
                            num_workers=4,
                            pin_memory=True)

    print("Data loaded and DataLoaders created.")
    print(f"Training on {len(train_data)} images, validating on {len(val_data)} images.")

    # Initialize Model, Loss, and Optimizer
    model = vgg_residual_blocks_V2(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 
                        lr=LEARNING_RATE,           
                        momentum=0.9,
                        weight_decay=WEIGHT_DECAY)
    
    scaler = torch.GradScaler(device=device)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    validation_accuracies = []

    for epoch in range(1, EPOCHS + 1):
        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        train(epoch)
        current_val_acc = validate(epoch)
        validation_accuracies.append(current_val_acc)
        scheduler.step()
        
        # Save the best model
        if current_val_acc > best_val_acc:
            print(f"Validation accuracy improved ({best_val_acc:.2f}% -> {current_val_acc:.2f}%).")
            best_val_acc = current_val_acc
            
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # Plot validation accuracy over epochs
    print("Generating validation accuracy plot...")
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, validation_accuracies, marker='o', linestyle='-', label='Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(1, EPOCHS + 1, 3))
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/validation_accuracy_plot.png') 
    print("Plot saved as validation_accuracy_plot.png")