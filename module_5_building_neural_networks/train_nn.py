import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Ensure output directory exists
os.makedirs("module_5_building_neural_networks", exist_ok=True)

# 1. Dataset Loading and Splitting
print("=== Step 1: Loading FashionMNIST Dataset ===")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

# Download full datasets
full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# To ensure fast training on CPU, we take subsets
# Training: 2000, Validation: 500, Test: 500
train_val_indices = list(range(2500))
test_indices = list(range(500))

train_val_subset = Subset(full_train_dataset, train_val_indices)
test_subset = Subset(full_test_dataset, test_indices)

# Manual Training-Validation split (80% Train, 20% Val)
train_size = 2000
val_size = 500
train_subset, val_subset = random_split(train_val_subset, [train_size, val_size])

print(f"Dataset Split Sizes:")
print(f" - Training Set:   {len(train_subset)}")
print(f" - Validation Set: {len(val_subset)}")
print(f" - Test Set:       {len(test_subset)}")

# Create Data Loaders
batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# 2. Model Architecture Definition
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(FeedForwardNN, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            
            # Layer Stacking: Dense Block 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),       # Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.3),            # Dropout for regularization
            
            # Dense Block 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output Layer (logits)
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

model = FeedForwardNN()
print("\n=== Step 2: Model Architecture ===")
print(model)

# 3. Model Compilation (Optimizer, Loss, and Metrics setup)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 30

# 4. Callbacks: Early Stopping and Checkpointing Classes
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

checkpoint_path = "module_5_building_neural_networks/best_model.pth"

# 5. Training Loop with Metrics Tracking
train_losses = []
val_losses = []
train_accs = []
val_accs = []

early_stopping = EarlyStopping(patience=5)
best_val_loss = float('inf')

print("\n=== Step 3: Starting Training ===")
for epoch in range(1, epochs + 1):
    # Training Phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()
        
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train
    
    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_val += targets.size(0)
            correct_val += predicted.eq(targets).sum().item()
            
    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = correct_val / total_val
    
    # Save statistics
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accs.append(epoch_train_acc)
    val_accs.append(epoch_val_acc)
    
    print(f"Epoch [{epoch}/{epochs}] - "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")
    
    # Checkpointing (Save if val loss improved)
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), checkpoint_path)
        print(f" => Saved best model checkpoint to '{checkpoint_path}' (Val Loss: {best_val_loss:.4f})")
        
    # Early Stopping check
    early_stopping(epoch_val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}.")
        break

# 6. Evaluation on Test Set using checkpointed model
print("\n=== Step 4: Final Test Evaluation (Best Model) ===")
# Load the saved best model weights
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

correct_test = 0
total_test = 0
test_loss = 0.0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_test += targets.size(0)
        correct_test += predicted.eq(targets).sum().item()

final_test_loss = test_loss / total_test
final_test_acc = correct_test / total_test
print(f"Test Loss: {final_test_loss:.4f} | Test Accuracy: {final_test_acc*100:.2f}%")

# 7. Plotting and Visualizing Results
print("\n=== Step 5: Plotting Training Curves ===")
plt.figure(figsize=(12, 5))

# Plot Losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Plot Accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy', color='#2ca02c', linewidth=2)
plt.plot(val_accs, label='Validation Accuracy', color='#d62728', linewidth=2)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plot_path = "module_5_building_neural_networks/training_curves.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved successfully as '{plot_path}'.")
plt.close()

print("\n=== Module 5 Training Complete! ===")
