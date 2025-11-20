import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
import optuna
import time

from vgg import vgg_residual_blocks_V2

# Configuration 
BATCH_SIZE = 128
NUM_CLASSES = 200
IMAGE_SIZE = 128
EPOCHS_PER_TRIAL = 10 # Train for 10 epochs per trial
NUM_TRIALS = 50

# Data Loading & Transforms

# Standard normalization for ImageNet-trained models
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
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
        image = item['image'].convert("RGB")
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Simplified train function
def train_epoch(model, device, train_loader, criterion, optimizer, scaler):
    model.train()
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device_str):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
# Simplified validate function
def validate_epoch(model, device, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            with torch.autocast(device_type=device_str):
                output = model(data)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_acc = 100. * correct / total
    return val_acc

# Optuna objective function
def objective(trial):

    model = vgg_residual_blocks_V2(num_classes=NUM_CLASSES).to(device)
    
    # Tune learning rate and weight decay
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True)
    
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    scaler = torch.GradScaler(device=device)

    for epoch in range(EPOCHS_PER_TRIAL):
        train_epoch(model, device, train_loader, criterion, optimizer, scaler)
        val_acc = validate_epoch(model, device, val_loader)
        
        trial.report(val_acc, epoch)
        
        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_acc

if __name__ == '__main__':

    start_time = time.time()

    # Use GPU
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Load Dataset
    tiny_imagenet = load_dataset("zh-plus/tiny-imagenet")
    indices = torch.randperm(len(tiny_imagenet['train'])).tolist()
    train_subset = tiny_imagenet['train'].select(indices[:10000]) # 10000 images
    val_data_set = HfDatasetWrapper(tiny_imagenet['valid'], val_transform)
    train_data_set = HfDatasetWrapper(train_subset, train_transform)
    
    train_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Starting Optuna hyperparameter search...")
    
    # Maximize validation accuracy.
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, # Wait 5 trials before pruning
            n_warmup_steps=3    # Wait 3 epochs before pruning
        )
    )
    
    # Run the optimization
    study.optimize(
        objective,
        n_trials=NUM_TRIALS 
    )

    print("\nSearch complete!")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Max Accuracy): {trial.value:.4f}%")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value:.6f}")

    end_time = time.time()
    print(f"Time for {NUM_TRIALS} trials: {end_time-start_time:.2f}s")