import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Ensure output directory exists
os.makedirs("module_6_improving_performance", exist_ok=True)

# 1. Prepare Datasets (Standard vs Augmented)
print("=== Step 1: Loading & Augmenting Dataset ===")

# Normal transforms
normal_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Augmented transforms (Adding Data Augmentation: Random rotation, Random horizontal flip)
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FashionMNIST
train_dataset_clean = datasets.FashionMNIST(root='./data', train=True, download=True, transform=normal_transform)
train_dataset_aug = datasets.FashionMNIST(root='./data', train=True, download=True, transform=aug_transform)
val_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=normal_transform)

# Subset indices
train_size = 2000
val_size = 500
overfit_size = 200

# Training subset indices
train_indices = list(range(0, train_size))
# Validation subset indices (disjoint from training)
val_indices = list(range(train_size, train_size + val_size))
# Small subset indices for deliberate overfitting
overfit_indices = list(range(0, overfit_size))

train_clean_subset = Subset(train_dataset_clean, train_indices)
train_aug_subset = Subset(train_dataset_aug, train_indices)
val_subset = Subset(val_dataset, val_indices)
overfit_subset = Subset(train_dataset_clean, overfit_indices)

# Loaders
batch_size = 64
train_clean_loader = DataLoader(train_clean_subset, batch_size=batch_size, shuffle=True)
train_aug_loader = DataLoader(train_aug_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
overfit_loader = DataLoader(overfit_subset, batch_size=16, shuffle=True)  # Small batch size for overfit

# 2. Architectures for Demonstration

# Model A: High Bias (Underfitting) - Very small capacity, high weight decay, too few epochs
class UnderfittingModel(nn.Module):
    def __init__(self):
        super(UnderfittingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 8),  # Extremely small capacity
            nn.ReLU(),
            nn.Linear(8, 10)
        )
    def forward(self, x): return self.network(x)

# Model B: High Variance (Overfitting) - High capacity, trained on very small data, no regularization
class OverfittingModel(nn.Module):
    def __init__(self):
        super(OverfittingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x): return self.network(x)

# Model C: Generalizing (Good Fit) - Reasonable capacity, dropout, weight decay, data augmentation
class GoodFitModel(nn.Module):
    def __init__(self):
        super(GoodFitModel, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),  # Regularization: Dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.network(x)

# Helper function to train and collect curves
def train_model(model, loader, val_loader, optimizer, criterion, epochs):
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_count += inputs.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_count += inputs.size(0)
                
        history['train_loss'].append(train_loss / train_count)
        history['val_loss'].append(val_loss / val_count)
    return history

print("\n=== Step 2: Running Underfitting Demonstration ===")
model_under = UnderfittingModel()
# Apply massive L2 weight decay (0.5) to cause deliberate underfitting
opt_under = optim.Adam(model_under.parameters(), lr=0.01, weight_decay=0.5)
hist_under = train_model(model_under, train_clean_loader, val_loader, opt_under, nn.CrossEntropyLoss(), epochs=5)
print(f"Underfit Model Final Train Loss: {hist_under['train_loss'][-1]:.4f} | Val Loss: {hist_under['val_loss'][-1]:.4f}")

print("\n=== Step 3: Running Overfitting Demonstration ===")
model_over = OverfittingModel()
# No weight decay, no dropout, trained on tiny subset (200 samples)
opt_over = optim.Adam(model_over.parameters(), lr=0.001)
hist_over = train_model(model_over, overfit_loader, val_loader, opt_over, nn.CrossEntropyLoss(), epochs=30)
print(f"Overfit Model Final Train Loss: {hist_over['train_loss'][-1]:.4f} | Val Loss: {hist_over['val_loss'][-1]:.4f}")

print("\n=== Step 4: Running Good Fit Demonstration ===")
model_good = GoodFitModel()
# Uses dropout, weight decay (1e-4), and Augmented Data Loader
opt_good = optim.Adam(model_good.parameters(), lr=0.001, weight_decay=1e-4)
hist_good = train_model(model_good, train_aug_loader, val_loader, opt_good, nn.CrossEntropyLoss(), epochs=15)
print(f"Good Fit Model Final Train Loss: {hist_good['train_loss'][-1]:.4f} | Val Loss: {hist_good['val_loss'][-1]:.4f}")

# Plot comparisons
print("\n=== Step 5: Plotting Comparisons ===")
plt.figure(figsize=(15, 5))

# Plot Underfitting
plt.subplot(1, 3, 1)
plt.plot(hist_under['train_loss'], label='Train Loss', color='red')
plt.plot(hist_under['val_loss'], label='Val Loss', color='red', linestyle='--')
plt.title('High Bias (Underfitting)\n(Low capacity, high L2 decay)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 2.5)
plt.legend()
plt.grid(True)

# Plot Overfitting
plt.subplot(1, 3, 2)
plt.plot(hist_over['train_loss'], label='Train Loss', color='orange')
plt.plot(hist_over['val_loss'], label='Val Loss', color='orange', linestyle='--')
plt.title('High Variance (Overfitting)\n(No Dropout/L2, small data)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 2.5)
plt.legend()
plt.grid(True)

# Plot Good Fit
plt.subplot(1, 3, 3)
plt.plot(hist_good['train_loss'], label='Train Loss', color='green')
plt.plot(hist_good['val_loss'], label='Val Loss', color='green', linestyle='--')
plt.title('Generalized (Good Fit)\n(Augmentation, Dropout, L2 decay)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 2.5)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('module_6_improving_performance/fit_comparison.png', dpi=150)
plt.close()
print("Plots saved as 'module_6_improving_performance/fit_comparison.png'")

# 3. K-Fold Cross-Validation Demonstration
print("\n=== Step 6: 3-Fold Cross-Validation Demonstration ===")
# Convert subset dataset into array tensors to do easy KFold splitting
x_data = []
y_data = []
for i in range(len(train_clean_subset)):
    img, label = train_clean_subset[i]
    x_data.append(img)
    y_data.append(label)
x_data = torch.stack(x_data)
y_data = torch.tensor(y_data)

kfold = KFold(n_splits=3, shuffle=True, random_seed=42) if hasattr(KFold, 'random_seed') else KFold(n_splits=3, shuffle=True, random_state=42)
fold_results = []

for fold, (train_ids, val_ids) in enumerate(kfold.split(x_data)):
    print(f"--- Training Fold {fold+1}/3 ---")
    
    # Define sub loaders
    fold_train_subset = TensorDataset(x_data[train_ids], y_data[train_ids])
    fold_val_subset = TensorDataset(x_data[val_ids], y_data[val_ids])
    
    fold_train_loader = DataLoader(fold_train_subset, batch_size=64, shuffle=True)
    fold_val_loader = DataLoader(fold_val_subset, batch_size=64, shuffle=False)
    
    # Create fresh model
    fold_model = GoodFitModel()
    fold_opt = optim.Adam(fold_model.parameters(), lr=0.002, weight_decay=1e-4)
    fold_criterion = nn.CrossEntropyLoss()
    
    # Train for 3 epochs
    for epoch in range(1, 4):
        fold_model.train()
        for inputs, targets in fold_train_loader:
            fold_opt.zero_grad()
            outputs = fold_model(inputs)
            loss = fold_criterion(outputs, targets)
            loss.backward()
            fold_opt.step()
            
    # Validate
    fold_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in fold_val_loader:
            outputs = fold_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = correct / total
    fold_results.append(acc)
    print(f"Fold {fold+1} Accuracy: {acc*100:.2f}%")

print(f"\nAverage 3-Fold Validation Accuracy: {np.mean(fold_results)*100:.2f}%")

# 4. Hyperparameter Tuning Demonstration
print("\n=== Step 7: Hyperparameter Tuning (Learning Rate Grid Search) ===")
learning_rates = [0.01, 0.001, 0.0001]
best_lr = None
best_acc = 0.0

for lr in learning_rates:
    print(f"Evaluating Learning Rate: {lr}...")
    tune_model = GoodFitModel()
    tune_opt = optim.Adam(tune_model.parameters(), lr=lr, weight_decay=1e-4)
    tune_criterion = nn.CrossEntropyLoss()
    
    # Train 3 epochs
    for epoch in range(3):
        tune_model.train()
        for inputs, targets in train_clean_loader:
            tune_opt.zero_grad()
            outputs = tune_model(inputs)
            loss = tune_criterion(outputs, targets)
            loss.backward()
            tune_opt.step()
            
    # Validate
    tune_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = tune_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = correct / total
    print(f" -> Validation Accuracy: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_lr = lr

print(f"\nBest Learning Rate: {best_lr} with Validation Accuracy: {best_acc*100:.2f}%")
print("=== Module 6 Performance Improvements Complete! ===")
