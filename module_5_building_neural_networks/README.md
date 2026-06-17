# Module 5: Building Neural Networks

This module covers the core concepts and design decisions required to construct and train feed-forward neural networks. Below is a detailed breakdown of the theoretical concepts, implementation details, and workflow structures.

---

## 🧠 Core Concepts

### 1. Sequential Models & Layer Stacking
A **Sequential model** is a linear stack of layers where each layer has exactly one input tensor and one output tensor. It is the most common way to build feed-forward networks (MLPs).
*   **Layer Stacking**: Deep neural networks derive power from stacking representation layers. Each successive layer extracts increasingly abstract features from the input data.
    $$\mathbf{h}^{(1)} = \sigma(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})$$
    $$\mathbf{h}^{(2)} = \sigma(\mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$

### 2. Dense (Linear) Layers
A **Dense layer** (or Fully Connected layer, `nn.Linear` in PyTorch) connects every neuron in the input to every neuron in the output.
*   **Equation**: $\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$
*   **Weights ($\mathbf{W}$)**: Learnable parameter matrix of shape `[out_features, in_features]`.
*   **Biases ($\mathbf{b}$)**: Learnable parameter vector of shape `[out_features]`.
*   **Activation Function ($\sigma$)**: Non-linear mapping (e.g., ReLU: $f(x)=\max(0, x)$) allowing the network to model non-linear boundaries.

### 3. Batch Normalization
**Batch Normalization** (`nn.BatchNorm1d` in PyTorch) stabilizes training by normalizing the layer inputs (zero mean, unit variance) across each mini-batch.
*   **Mechanics**:
    $$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$
    $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
    $$y_i = \gamma \hat{x}_i + \beta$$
*   **Benefits**:
    - Mitigates internal covariate shift.
    - Allows for higher learning rates.
    - Acts as a mild form of regularization.

### 4. Dropout
**Dropout** (`nn.Dropout` in PyTorch) randomly zeroes out a fraction $p$ of the input units during training at each step.
*   **Goal**: Prevents co-adaptation of features (neurons relying on specific other neurons), forcing the network to learn redundant representations.
*   **Behavior**:
    - **Training**: Zeroes outputs with probability $p$, and scales the remaining outputs by $\frac{1}{1-p}$ to keep the expected sum constant.
    - **Evaluation**: Disabled (weights are fully active).

---

## 🛠️ Model Compilation & Training Lifecycle

### 1. Loss Functions & Optimizers
Unlike Keras, PyTorch doesn't have an explicit `model.compile()` step. Instead, we manually define the optimizer, loss function, and metrics:
*   **Loss Function**: Measures how far predictions are from ground truth. For multi-class classification, we use **Cross-Entropy Loss** (`nn.CrossEntropyLoss`).
*   **Optimizer**: Updates model parameters based on gradients. We use **Adam** (`optim.Adam`), which maintains adaptive per-parameter learning rates.
*   **Metrics**: Custom variables tracked during loops (e.g., Accuracy, precision).

### 2. Dataset Splitting Strategy
To ensure generalizability, the dataset is divided into three distinct sets:
*   **Training Set (80%)**: Used by the optimizer to update model parameters.
*   **Validation Set (10%)**: Used during training to monitor performance and prevent overfitting (via early stopping).
*   **Test Set (10%)**: Unseen data used only for the final evaluation of generalization performance.

---

## 🔄 Callbacks & Workflow Safeguards

> [!NOTE]
> In PyTorch, we implement custom training loop callbacks to achieve Keras-style callback behaviors.

### 1. Early Stopping
Monitors validation loss. If validation loss stops decreasing for a specified number of epochs (called the **patience**), training is terminated early.
*   **Benefit**: Prevents wasteful computation and overfitting.

### 2. Model Checkpointing
Saves the state dictionary (`model.state_dict()`) of the network whenever the validation metric improves (e.g., validation loss reaches a new minimum).
*   **Benefit**: Recovers the best-performing model state after training, even if the model overfits in later epochs.

---

## 📊 Results Summary

Our script `train_nn.py` trains a feedforward neural network on a subset of **FashionMNIST** images.

### Architecture:
1.  **Flatten**: $28 \times 28 \to 784$ input dimensions.
2.  **Dense Layer 1**: $784 \to 256$, followed by **BatchNorm1d**, **ReLU**, and **Dropout(p=0.3)**.
3.  **Dense Layer 2**: $256 \to 128$, followed by **BatchNorm1d**, **ReLU**, and **Dropout(p=0.3)**.
4.  **Output Layer**: $128 \to 10$ logits.

Training outputs are saved in the current directory, including `training_curves.png` and `best_model.pth`.
