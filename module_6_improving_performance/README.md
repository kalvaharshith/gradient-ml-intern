# Module 6: Improving Model Performance

This module focuses on diagnosing the behavior of machine learning models and applying various strategies to improve generalization performance and mitigate underfitting and overfitting.

---

## ⚖️ Bias-Variance Tradeoff

The prediction error of any machine learning model can be decomposed into three components: **Bias**, **Variance**, and **Irreducible Error**.

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

### 1. Bias (Underfitting)
*   **Definition**: Error introduced by approximating a real-world problem (which may be highly complex) by a much simpler model.
*   **Characteristics**: High training error and high validation error.
*   **Solutions**:
    - Increase model capacity (add more layers or units).
    - Decrease regularization (reduce weight decay or dropout rate).
    - Train for more epochs or use better optimization techniques.

### 2. Variance (Overfitting)
*   **Definition**: Error introduced by the model's sensitivity to small fluctuations in the training set. The model learns the noise instead of the signal.
*   **Characteristics**: Low training error but high validation error (a large gap between the two).
*   **Solutions**:
    - Add regularization (L1/L2 penalties, dropout).
    - Acquire more training data or apply **data augmentation**.
    - Reduce model capacity if it is excessively large.
    - Implement early stopping.

---

## 🛡️ Regularization Techniques

Regularization adds a penalty term to the loss function to prevent the model weights from growing too large or becoming over-specialized.

### 1. L1 & L2 Penalties
Let $L_0(\theta)$ be the original loss function (e.g., Cross-Entropy).

*   **L2 Regularization (Ridge / Weight Decay)**:
    Adds the squared sum of weights to the loss:
    $$L(\theta) = L_0(\theta) + \lambda \frac{1}{2} \|\mathbf{w}\|_2^2 = L_0(\theta) + \frac{\lambda}{2} \sum w_i^2$$
    - In PyTorch, L2 regularization is directly implemented via the `weight_decay` parameter in optimizers (e.g. `optim.Adam(..., weight_decay=1e-4)`).
    - It shrinks weights smoothly towards zero, reducing model complexity.

*   **L1 Regularization (Lasso)**:
    Adds the absolute sum of weights to the loss:
    $$L(\theta) = L_0(\theta) + \lambda \|\mathbf{w}\|_1 = L_0(\theta) + \lambda \sum |w_i|$$
    - Drives weights to exactly zero, resulting in sparse weight matrices (feature selection).
    - Must be added manually to the loss function in PyTorch during the training loop.

### 2. Dropout Strategies
*   **Varied Rates**: Higher dropout rates (e.g., $p=0.5$) for larger dense layers near the inputs, and lower rates (e.g., $p=0.1$ or $0.2$) for smaller layers.
*   **Spatial Dropout**: Used in CNNs to drop entire feature channels rather than individual pixels.

---

## 🖼️ Data Augmentation

Data augmentation artificially expands the training dataset by creating modified versions of existing samples. This forces the model to be invariant to specific transformations (e.g., translation, rotation).

*   **Transforms in PyTorch**:
    We use `torchvision.transforms` to apply on-the-fly random transformations:
    - `RandomHorizontalFlip()`: Mirrors images horizontally.
    - `RandomRotation(degrees)`: Rotates images within a specific range.
    - `RandomResizedCrop(size)`: Randomly crops and resizes parts of the image.

---

## ⚙️ Hyperparameter Tuning & Cross-Validation

### 1. Hyperparameter Search
Hyperparameters (learning rate, batch size, weight decay, dropout rate) are parameters set before training.
*   **Grid Search**: Systematic evaluation of every combination of hyperparameters in a specified grid.
*   **Random Search**: Randomly samples configurations from a distribution. Often more efficient than grid search because it explores more values of influential hyperparameters.

### 2. K-Fold Cross-Validation
Divides the training data into $K$ equal-sized folds. The model is trained on $K-1$ folds and validated on the remaining fold. This process is repeated $K$ times, and the average validation score is computed.
*   **Benefit**: Provides a robust estimate of generalization error, reducing the bias of a single train-validation split.

---

## 📊 Results Summary

Our script `regularization_and_tuning.py` demonstrates these concepts:

1.  **Underfitting Model**: 1 tiny hidden layer (8 units), high L2 penalty (`weight_decay=0.5`), and short training (3 epochs).
2.  **Overfitting Model**: 3 wide layers (512 units each), trained on a tiny subset of 200 samples for 50 epochs, with **no** dropout and **no** weight decay.
3.  **Good Fit (Regularized) Model**: 2 layers (256, 128 units), with dropout ($p=0.3$) and moderate weight decay (`1e-4`), trained on the full 2,000 samples.

Training curves for all three are plotted together in `fit_comparison.png` to demonstrate the bias-variance tradeoff.
Additionally, the script runs a grid search tuning learning rates and evaluates them using a **3-Fold Cross-Validation** loop.
