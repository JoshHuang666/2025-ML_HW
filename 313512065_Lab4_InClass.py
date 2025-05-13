import struct
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sgd_logistic(X, y, eta, max_iters):
    np.random.seed(42)  # 確保結果可復現
    w = np.zeros(X.shape[1])  # 初始化權重
    
    for _ in range(max_iters):
        idx = np.random.randint(0, X.shape[0])  # 隨機選取一個樣本
        x_i = X[idx]
        y_i = y[idx]
        pred = sigmoid(np.dot(w, x_i))
        gradient = (pred - y_i) * x_i
        w -= eta * gradient  # 梯度更新
        return w

def load_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images[:(len(images) // (rows * cols)) * rows * cols]
        return images.reshape(-1, rows * cols).astype(np.float32) / 255.0  # 正規化至 [0,1]

def load_labels(filename):
    with open(filename, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels[:num]
        
    return w
if __name__ == "__main__":
    # === 加載數據 ===
    X_train = load_images("train-images.idx3-ubyte___")
    y_train = load_labels("train-labels.idx1-ubyte___")
    X_test = load_images("t10k-images.idx3-ubyte___")
    y_test = load_labels("t10k-labels.idx1-ubyte___")

    # === 設定二元分類目標（最後一位數字） ===
    TARGET_DIGIT = 5  
    y_train_bin = np.where(y_train == TARGET_DIGIT, 1, 0)
    y_test_bin = np.where(y_test == TARGET_DIGIT, 1, 0)

    # === 添加偏置項 ===
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # === 設定參數 ===
    eta = 0.01  # 學習率
    max_iters = 10000  # 迭代次數

    # === 訓練模型 ===
    w = sgd_logistic(X_train, y_train_bin, eta, max_iters)

    # === 預測 ===
    pred_probs = sigmoid(np.dot(X_test, w))
    preds = (pred_probs >= 0.5).astype(int)

    # === 計算準確率 ===
    accuracy = np.mean(preds == y_test_bin)
    print(f"Accuracy: {accuracy:.4f}")
def show_misclassified(X, true_labels, pred_labels, max_show=10):
    mis_idx = np.where(true_labels != pred_labels)[0][:max_show]
    plt.figure(figsize=(10, 2))
    for i, idx in enumerate(mis_idx):
        plt.subplot(1, len(mis_idx), i + 1)
        plt.imshow(X[idx, 1:].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f"T:{true_labels[idx]} P:{pred_labels[idx]}")
    plt.suptitle("Misclassified Samples")
    plt.show()

# 顯示錯誤分類的樣本
show_misclassified(X_test, y_test_bin, preds)


