import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Load a grayscale image and normalize
image = cv2.imread('original.png', cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Step 2: General Convolution Function
def convolve2d(image, kernel, padding=0, stride=1):
    # 1. Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))

    # 2. Apply zero-padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    H, W = image.shape
    kH, kW = kernel.shape

    # 3. Calculate output size
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1

    # 4. Create output
    output = np.zeros((out_h, out_w), dtype=np.float32)

    # 5. Perform convolution
    for i in range(0, out_h):
        for j in range(0, out_w):
            region = image[i*stride:i*stride+kH, j*stride:j*stride+kW]
            output[i, j] = np.sum(region * kernel)

    return output

# Step 3: Define edge detection filters (Sobel kernels)
vertical_filter = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

horizontal_filter = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# Step 4: Convolve image with filters
vertical_edges = convolve2d(image, vertical_filter, padding=1, stride=1)
horizontal_edges = convolve2d(image, horizontal_filter, padding=1, stride=1)

vertical_stride = convolve2d(image, vertical_filter, padding=1, stride=2)
horizontal_stride = convolve2d(image, horizontal_filter, padding=1, stride=2)

# Step 5: Binarize and visualize
def binarize(img, threshold=0.5):
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)
    return (img > threshold).astype(np.float32)

# Binarize all outputs
v_bin = binarize(vertical_edges)
h_bin = binarize(horizontal_edges)
v_str_bin = binarize(vertical_stride)
h_str_bin = binarize(horizontal_stride)

# Show all 5 views
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(2, 3, 2)
plt.imshow(v_bin, cmap='gray')
plt.title('Vertical Edges (pad=1)')

plt.subplot(2, 3, 3)
plt.imshow(h_bin, cmap='gray')
plt.title('Horizontal Edges (pad=1)')

plt.subplot(2, 3, 5)
plt.imshow(v_str_bin, cmap='gray')
plt.title('Vertical Edges (stride=2)')

plt.subplot(2, 3, 6)
plt.imshow(h_str_bin, cmap='gray')
plt.title('Horizontal Edges (stride=2)')

plt.tight_layout()
plt.show()
