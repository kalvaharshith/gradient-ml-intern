# Internship Work - Task 2: Deep Learning

This repository contains implementation work and educational content for **Task 2: Deep Learning** of the Machine Learning Internship. It covers building feed-forward neural networks, diagnosing and improving performance, and applying convolutional neural networks (CNNs) for image classification and transfer learning.

Each module is self-contained with its own code script, explanation, and visual outputs (plots).

---

## 📁 Repository Structure

The project is structured as follows:

*   **`module_5_building_neural_networks/`**: Core neural network design using PyTorch.
    *   `README.md`: Theory on Dense layers, Sequential architectures, Early Stopping, and Checkpointing.
    *   `train_nn.py`: Modular training script using a subset of FashionMNIST. Trains a feed-forward neural network with Dropout, Batch Normalization, and custom callbacks.
    *   `training_curves.png`: Training and validation loss/accuracy curves.
*   **`module_6_improving_performance/`**: Advanced optimization, regularization, data augmentation, and hyperparameter tuning.
    *   `README.md`: Discussion on Underfitting vs. Overfitting, L1/L2 penalties, and cross-validation theory.
    *   `regularization_and_tuning.py`: Script comparing Underfitting, Overfitting, and Generalized architectures. Runs hyperparameter tuning and cross-validation.
    *   `fit_comparison.png`: Plots comparing training/validation loss across underfit, overfit, and regularized runs.
*   **`module_7_convolutional_neural_networks/`**: Convolutional Neural Networks (CNNs) and Transfer Learning.
    *   `README.md`: Mechanics of convolution, stride, padding, pooling, and transfer learning workflows.
    *   `cnn_classification.py`: Builds a custom CNN, extracts feature maps, and applies transfer learning/fine-tuning using a pretrained MobileNetV2.
    *   `feature_maps.png`: Visual representations of convolutional filters and their activation maps.
    *   `cnn_vs_transfer.png`: Training curve comparisons between the custom CNN and pretrained transfer learning.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.11+ installed. Install the required libraries:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

### 2. Running the Modules
Each folder contains a runnable script that downloads required subsets of datasets, trains the models (using CPU), prints progress logs, and saves visualization plots.

Run Module 5 (Neural Network Basics):
```bash
python module_5_building_neural_networks/train_nn.py
```

Run Module 6 (Regularization and Tuning):
```bash
python module_6_improving_performance/regularization_and_tuning.py
```

Run Module 7 (CNNs & Transfer Learning):
```bash
python module_7_convolutional_neural_networks/cnn_classification.py
```

---

## 📊 Summary of Learning Outcomes

1.  **Module 5**: Designed sequential models incorporating dense layers, batch normalization, and dropout. Used early stopping and checkpoints to ensure robust model selection.
2.  **Module 6**: Mitigated overfitting using L2 regularization and dropout strategies. Applied random data augmentations and evaluated models using K-fold cross-validation.
3.  **Module 7**: Built an image classification pipeline using Conv2D layers, padding, pooling, and stride. Employed transfer learning with frozen backbones and fine-tuned deep layers of pretrained vision models.
