import numpy as np
import struct
import matplotlib.pyplot as plt
import pandas as pd

# ========== Load Dataset ==========
def load_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape((num, rows * cols))
        return data.astype(np.float32) / 255.0

def load_labels(filename):
    with open(filename, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# ========== 1. Sigmoid ==========
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# ========== 3. Forward and Reverse Autodiff Trace ==========
def trace_autodiff_example(x1, x2, y_true):
    # Primal
    v1 = x1
    v2 = x2
    v3 = v1 + v2
    v4 = v1 * v3
    v5 = sigmoid(v4)
    v6 = y_true * np.log(v5 + 1e-8) + (1 - y_true) * np.log(1 - v5 + 1e-8)  # BCE
    v7 = -v6

    # Forward mode: ∂v/∂x1 where dx1=1, dx2=0
    dv1 = 1
    dv2 = 0
    dv3 = dv1 + dv2
    dv4 = dv1 * v3 + v1 * dv3
    dv5 = sigmoid_derivative(v4) * dv4
    dv6 = y_true * (1 / (v5 + 1e-8)) * dv5 - (1 - y_true) * (1 / (1 - v5 + 1e-8)) * dv5
    dv7 = -dv6

    # Reverse mode (adjoint)
    dv7_rev = 1
    dv6_rev = -dv7_rev
    dv5_rev = dv6_rev * (y_true / (v5 + 1e-8) - (1 - y_true) / (1 - v5 + 1e-8))
    dv4_rev = dv5_rev * sigmoid_derivative(v4)
    dv3_rev = v1 * dv4_rev
    dv1_rev = dv4_rev * v3 + dv3_rev
    dv2_rev = dv3_rev

    table = pd.DataFrame({
        'Variable': ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7'],
        'Primal (v)': [v1, v2, v3, v4, v5, v6, v7],
        'Forward Tangent (ẋ)': [dv1, dv2, dv3, dv4, dv5, dv6, dv7],
        'Reverse Adjoint (v̄)': [dv1_rev, dv2_rev, dv3_rev, dv4_rev, dv5_rev, dv6_rev, dv7_rev]
    })
    return table

# ========== 2. SGD ==========
def your_sgd_logistic(X, y, eta=0.1, max_iters=100):
    w = np.zeros(X.shape[1])
    for i in range(max_iters):
        z = X @ w
        preds = sigmoid(z)
        grad = X.T @ (preds - y) / len(y)
        w -= eta * grad
        if i == 0:
            # Trace the first sample for autodiff trace
            trace = trace_autodiff_example(X[0, 1], X[0, 2], y[0])
    return w, trace

# ========== Show Misclassified ==========
def show_misclassified(X, y_true, y_pred, max_show=10):
    mis_idx = np.where(y_true != y_pred)[0][:max_show]
    if len(mis_idx) == 0:
        print("No misclassifications!")
        return
    plt.figure(figsize=(10, 2))
    for i, idx in enumerate(mis_idx):
        plt.subplot(1, len(mis_idx), i + 1)
        plt.imshow(X[idx, 1:].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f"T:{y_true[idx]}\nP:{y_pred[idx]}")
    plt.suptitle("Misclassified Samples")
    plt.show()

# ========== Plot Trace Graph ==========
def plot_autodiff_traces(trace_df):
    variables = trace_df['Variable']
    primal = trace_df['Primal (v)'].astype(float)
    forward = pd.to_numeric(trace_df['Forward Tangent (ẋ)'], errors='coerce')
    reverse = pd.to_numeric(trace_df['Reverse Adjoint (v̄)'], errors='coerce')
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].bar(variables, primal, color='skyblue')
    ax[0].set_ylabel("Primal (v)")
    ax[0].set_title("Primal Values")
    ax[1].bar(variables, forward, color='lightgreen')
    ax[1].set_ylabel("Forward Tangent (ẋ)")
    ax[1].set_title("Forward-Mode Autodiff")
    ax[2].bar(variables, reverse, color='salmon')
    ax[2].set_ylabel("Reverse Adjoint (v̄)")
    ax[2].set_title("Reverse-Mode Autodiff")
    ax[2].set_xlabel("Variables")
    plt.tight_layout()
    plt.show()

# ========== 3. Main ==========
def main():
    # === Load MNIST ===
    X_train = load_images("train-images.idx3-ubyte___")
    y_train = load_labels("train-labels.idx1-ubyte___")
    X_test = load_images("t10k-images.idx3-ubyte___")
    y_test = load_labels("t10k-labels.idx1-ubyte___")

    # === Binary Classification for Digit 5 ===
    TARGET_DIGIT = 5
    
    y_train_bin = np.where(y_train == TARGET_DIGIT, 1, 0)
    y_test_bin = np.where(y_test == TARGET_DIGIT, 1, 0)

    # === Add Bias Term ===
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    # === Train ===
    w, autodiff_trace = your_sgd_logistic(X_train, y_train_bin)

    # === Predict ===
    pred_probs = sigmoid(X_test @ w)
    preds = (pred_probs >= 0.5).astype(int)

    # === Accuracy ===
    acc = np.mean(preds == y_test_bin)
    print(f"\nTest Accuracy (is {TARGET_DIGIT} or not): {acc:.4f}")

    # === Show Misclassifications ===
    show_misclassified(X_test, y_test_bin, preds)

    # === Visualize Autodiff Trace ===
    print("\nAutodiff Trace Table (first sample):")
    print(autodiff_trace)
    plot_autodiff_traces(autodiff_trace)

if __name__ == "__main__":
    main()
