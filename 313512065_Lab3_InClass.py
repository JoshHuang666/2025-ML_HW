import numpy as np

# ------------------------
# 1. 定義激活函數
def tanh(x):
    return np.tanh(x)

def hard_tanh(x):
    return np.clip(x, -1, 1)

def softplus(x):
    return np.log1p(np.exp(x))  # log(1+exp(x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

# 2. 選擇測試的激活函數
activation_function = relu  # 可更改為其他函數

# 3. 定義輸入與權重
x_raw = np.array([[0.5], [0.2], [0.1]])  # (3, 1)
x = np.vstack(([1.0], x_raw))  # 加上 bias x0 = 1

W1 = np.array([
    [0.1, 0.1, 0.2, 0.3],
    [0.2, -0.3, 0.4, 0.1],
    [0.05, 0.2, -0.2, 0.1],
    [0.0, 0.3, -0.1, 0.2]
])  # (4, 4) -> 4 hidden nodes, 4 inputs (incl. bias)

W2 = np.array([
    [0.2, 0.3, -0.1, 0.5, 0.1],
    [-0.2, 0.4, 0.3, -0.1, 0.2]
])  # (2, 5) -> 2 outputs, 5 inputs (incl. bias)

# 4. 前饋計算 (Forward Pass)
# Step 1: 計算隱藏層 pre-activation 值
a1 = np.dot(W1, x)  # (4, 1)

# Step 2: 套用激活函數
z1 = activation_function(a1)  # (4, 1)

# Step 3: 加入偏置單元
z1_aug = np.vstack(([1.0], z1))  # (5, 1)

# Step 4: 計算輸出層結果
y = np.dot(W2, z1_aug)  # (2, 1)

# 5. 印出結果
print("輸入 x (含偏置):\n", x.T)
print("隱藏層 pre-activation a1:\n", a1.T)
print("隱藏層輸出 z1:\n", z1.T)
print("隱藏層 (含偏置) z1_aug:\n", z1_aug.T)
print("最終輸出 y:\n", y.T)
