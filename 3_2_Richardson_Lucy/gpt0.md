 To implement **Richardson-Lucy Deconvolution (RLD)** for image restoration using **Python with OpenCV and NumPy**, we'll simulate the process from scratch. This approach does **not** require any additional libraries beyond `numpy` and OpenCV (`cv2`).

---

### 📌 Problem Summary

Given:

* A blurred image `B`, which is a convolution of original image `A` with a kernel `K`.
* The goal is to **reconstruct an approximation of `A`**, denoted as `A'`, using **Richardson-Lucy Deconvolution**.

---

### 🔁 Richardson-Lucy Deconvolution Algorithm

The RLD update formula is:

$$
A_{n+1} = A_n \cdot \left( \frac{B}{A_n * K} * K^{\text{flip}} \right)
$$

Where:

* `*` = convolution,
* `/` = element-wise division,
* `K^{flip}` = flipped kernel (180 degrees),
* `A_n` = current estimate of the image,
* `B` = observed (blurred) image.

---

### ✅ Python Implementation with OpenCV + NumPy

Here's a minimal implementation:

```python
import cv2
import numpy as np

def flip_kernel(kernel):
    return np.flip(np.flip(kernel, 0), 1)

def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

def richardson_lucy(B, K, iterations=20):
    B = B.astype(np.float32) + 1e-8  # Avoid division by zero
    A = np.full(B.shape, 0.5, dtype=np.float32)  # Initial guess (can also use B.copy())
    K_flipped = flip_kernel(K)

    for _ in range(iterations):
        conv = convolve(A, K) + 1e-8
        ratio = B / conv
        correction = convolve(ratio, K_flipped)
        A *= correction

    return np.clip(A, 0, 1)

# Load blurred image (B)
B = cv2.imread('blurred_input.png', cv2.IMREAD_GRAYSCALE) / 255.0

# Define blur kernel (e.g., Gaussian)
K = cv2.getGaussianKernel(9, 2)
K = K @ K.T  # Create 2D Gaussian kernel

# Run RLD
restored = richardson_lucy(B, K, iterations=30)

# Show and save result
cv2.imshow("Restored", restored)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("restored_output.png", (restored * 255).astype(np.uint8))
```

---

### 🧪 Optional Kernels You Can Try

```python
# Mean blur kernel
K = np.ones((5, 5), np.float32) / 25

# Motion blur (horizontal)
K = np.zeros((5, 5))
K[2, :] = 1.0 / 5

# Diagonal blur
K = np.eye(5) / 5
```

---

Would you like a working example image pair (`A`, `B`) or a full GUI-style ImageJ-compatible plugin version as well?

