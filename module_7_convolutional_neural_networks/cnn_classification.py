import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Ensure directory exists
os.makedirs("module_7_convolutional_neural_networks", exist_ok=True)

# -------------------------------------------------------------
# Part 1: Custom CNN on FashionMNIST & Feature Map Visualization
# -------------------------------------------------------------
print("=== Part 1: Custom CNN on FashionMNIST ===")

# Transforms for FashionMNIST (Grayscale, 1-channel)
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset_mnist = datasets.FashionMNIST(root='./data', train=True, download=True, transform=mnist_transform)
val_dataset_mnist = datasets.FashionMNIST(root='./data', train=True, download=True, transform=mnist_transform)

# Subset to speed up CPU training
train_size = 2000
val_size = 500
train_mnist_subset = Subset(train_dataset_mnist, list(range(train_size)))
val_mnist_subset = Subset(val_dataset_mnist, list(range(train_size, train_size + val_size)))

train_loader_mnist = DataLoader(train_mnist_subset, batch_size=64, shuffle=True)
val_loader_mnist = DataLoader(val_mnist_subset, batch_size=64, shuffle=False)

# Custom CNN Architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Feature Extractor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14 -> 7x7
        
        # Classifier Head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

cnn_model = CustomCNN()
print("\nCustom CNN Structure:")
print(cnn_model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.002)

cnn_epochs = 5
cnn_train_losses = []
cnn_val_losses = []

print("\nTraining Custom CNN...")
for epoch in range(1, cnn_epochs + 1):
    cnn_model.train()
    running_loss = 0.0
    total = 0
    for inputs, targets in train_loader_mnist:
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
    
    epoch_train_loss = running_loss / total
    
    # Validation
    cnn_model.eval()
    running_val_loss = 0.0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in val_loader_mnist:
            outputs = cnn_model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * inputs.size(0)
            val_total += inputs.size(0)
            
    epoch_val_loss = running_val_loss / val_total
    cnn_train_losses.append(epoch_train_loss)
    cnn_val_losses.append(epoch_val_loss)
    
    print(f"Epoch [{epoch}/{cnn_epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

# Visualize first layer feature maps
print("\nExtracting and Plotting Feature Maps...")
cnn_model.eval()
sample_img, sample_lbl = train_mnist_subset[0]
sample_tensor = sample_img.unsqueeze(0)  # Add batch dimension: [1, 1, 28, 28]

with torch.no_grad():
    # Pass through first conv layer
    first_conv_out = cnn_model.conv1(sample_tensor)  # Shape: [1, 16, 28, 28]
    first_conv_out = cnn_model.relu1(cnn_model.bn1(first_conv_out))

# Plot original image and activation maps
fig, axes = plt.subplots(3, 6, figsize=(12, 6))
axes = axes.flatten()

# Original Image
axes[0].imshow(sample_img.squeeze(), cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

# Plot the 16 activation maps (from channels 0 to 15)
for i in range(16):
    feature_map = first_conv_out[0, i].numpy()
    axes[i + 1].imshow(feature_map, cmap='viridis')
    axes[i + 1].set_title(f"Filter {i+1}")
    axes[i + 1].axis('off')

# Hide remaining unused subplot
axes[17].axis('off')

plt.tight_layout()
feat_map_path = "module_7_convolutional_neural_networks/feature_maps.png"
plt.savefig(feat_map_path, dpi=150)
plt.close()
print(f"Feature maps saved successfully as '{feat_map_path}'.")

# -------------------------------------------------------------
# Part 2: Transfer Learning and Fine-tuning on FashionMNIST (RGB Expanded)
# -------------------------------------------------------------
print("\n=== Part 2: Transfer Learning on FashionMNIST (RGB-Expanded) ===")

# MobileNetV2 expects 3 channels. We resize to 64x64 and replicate grayscale channels to RGB.
tl_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.Grayscale(num_output_channels=3),  # Replicates the channel 3 times to get an RGB image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use the already downloaded FashionMNIST dataset (download=False)
train_dataset_tl = datasets.FashionMNIST(root='./data', train=True, download=False, transform=tl_transform)
val_dataset_tl = datasets.FashionMNIST(root='./data', train=False, download=False, transform=tl_transform)

# Subset to speed up CPU execution (500 train, 200 validation)
tl_train_size = 500
tl_val_size = 200
train_tl_subset = Subset(train_dataset_tl, list(range(tl_train_size)))
val_tl_subset = Subset(val_dataset_tl, list(range(tl_val_size)))

train_loader_cifar = DataLoader(train_tl_subset, batch_size=32, shuffle=True)
val_loader_cifar = DataLoader(val_tl_subset, batch_size=32, shuffle=False)

# Load Pretrained Model
print("Loading Pretrained MobileNetV2...")
try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model_pt = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
except ImportError:
    from torchvision.models import mobilenet_v2
    model_pt = mobilenet_v2(pretrained=True)

# Step A: Freeze all feature extraction layers
for param in model_pt.parameters():
    param.requires_grad = False

# Step B: Replace Classifier Head
in_features = model_pt.classifier[1].in_features
model_pt.classifier[1] = nn.Linear(in_features, 10) # 10 classes in CIFAR-10

print("MobileNetV2 classifier head modified. Frozen backbone layers.")

# Train Classifier Head Only
optimizer_tl = optim.Adam(model_pt.classifier[1].parameters(), lr=0.005)
tl_epochs = 3
tl_losses = []

print("\n--- Training Classifier Head (Transfer Learning) ---")
for epoch in range(1, tl_epochs + 1):
    model_pt.train()
    running_loss = 0.0
    total = 0
    for inputs, targets in train_loader_cifar:
        optimizer_tl.zero_grad()
        outputs = model_pt(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_tl.step()
        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
    
    epoch_loss = running_loss / total
    tl_losses.append(epoch_loss)
    print(f"Epoch [{epoch}/{tl_epochs}] (Head-Only) - Loss: {epoch_loss:.4f}")

# Step C: Fine-tuning
print("\n--- Fine-tuning (Unfreezing Deep Layers) ---")
# Unfreeze the last block features[18] of MobileNetV2
for param in model_pt.features[18].parameters():
    param.requires_grad = True

# We use a very low learning rate for fine tuning
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_pt.parameters()), lr=0.0001)
ft_epochs = 2
ft_losses = []

for epoch in range(1, ft_epochs + 1):
    model_pt.train()
    running_loss = 0.0
    total = 0
    for inputs, targets in train_loader_cifar:
        optimizer_ft.zero_grad()
        outputs = model_pt(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_ft.step()
        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
        
    epoch_loss = running_loss / total
    ft_losses.append(epoch_loss)
    print(f"Epoch [{epoch}/{ft_epochs}] (Fine-Tuning) - Loss: {epoch_loss:.4f}")

# Combine transfer learning and fine tuning losses for plotting
total_tl_losses = tl_losses + ft_losses

# Save comparison training curves
print("\n=== Step 3: Saving Training Curves ===")
plt.figure(figsize=(10, 5))
plt.plot(cnn_train_losses, label='Custom CNN Train Loss', marker='o')
plt.plot(cnn_val_losses, label='Custom CNN Val Loss', marker='x')
plt.plot(total_tl_losses, label='MobileNetV2 Transfer Learning Loss', linestyle='--', marker='s')
plt.title('Training Loss Comparison: Custom CNN vs. Pretrained MobileNetV2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

comparison_plot_path = "module_7_convolutional_neural_networks/cnn_vs_transfer.png"
plt.savefig(comparison_plot_path, dpi=150)
plt.close()
print(f"Comparison plot saved successfully as '{comparison_plot_path}'.")
print("=== Module 7 CNN Tasks Complete! ===")
