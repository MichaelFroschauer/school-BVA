 Great — let’s expand the Richardson-Lucy Deconvolution (RLD) implementation to match the lecture/lab instructions. You want:

1. ✅ **RLD algorithm exactly as in slides**
2. 🔁 **Multiple kernel types**
3. 📈 **Noise in kernel or blurred image `B`**
4. 🎲 **Different initial guesses for `A'` (e.g. random, constant, `B`, image)**

---

## 🧠 1. RLD Algorithm (as in Lecture/Lab)

From a standard formulation as in most image processing courses:

$$
A_{n+1} = A_n \cdot \left( \left( \frac{B}{A_n * K + \epsilon} \right) * K^{flip} \right)
$$

Where:

* `*` = convolution
* `K^flip` = 180° rotated kernel
* `ε` = small constant to avoid division by zero

We will include proper flipping and boundary handling.

---

## 🧪 2. Various Kernels

Let’s define:

* Mean blur
* Gaussian blur
* Motion (horizontal)

---

## 🔊 3. Noise Handling

Additive Gaussian noise can be added to:

* the blurred image `B`
* the kernel `K`

---

## 🎨 4. Initial Estimates for A′

Initial estimate options:

* `B` (blurred image)
* Constant gray (e.g., 0.5 or 127)
* Random noise
* A fixed image like “Donald Duck” (mock or real placeholder)

---

## ✅ Full Python Implementation

```python
import cv2
import numpy as np

def flip_kernel(kernel):
    return np.flip(np.flip(kernel, axis=0), axis=1)

def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

def add_noise(img, noise_level=0.01):
    noise = np.random.normal(0, noise_level, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)

def richardson_lucy(B, K, iterations=20, A_init='random', noise_level_B=0.0, noise_level_K=0.0):
    B = B.astype(np.float32) / 255.0
    B = add_noise(B, noise_level=noise_level_B)

    K = K.astype(np.float32)
    K /= np.sum(K)
    K = add_noise(K, noise_level=noise_level_K)
    K /= np.sum(K)  # renormalize after noise

    K_flipped = flip_kernel(K)

    # Initial estimate A'
    if A_init == 'random':
        A = np.random.rand(*B.shape).astype(np.float32)
    elif A_init == 'gray':
        A = np.full(B.shape, 0.5, dtype=np.float32)
    elif A_init == 'B':
        A = B.copy()
    elif isinstance(A_init, np.ndarray):
        A = cv2.resize(A_init, (B.shape[1], B.shape[0])) / 255.0
    else:
        raise ValueError("Unknown initialization method")

    eps = 1e-8
    for _ in range(iterations):
        conv = convolve(A, K) + eps
        ratio = B / conv
        correction = convolve(ratio, K_flipped)
        A *= correction

    return np.clip(A, 0, 1)

# -- KERNEL DEFINITIONS --
def get_kernels():
    kernels = {}

    # Mean blur
    kernels['mean'] = np.ones((5, 5), np.float32) / 25

    # Gaussian blur
    gk1d = cv2.getGaussianKernel(9, 2)
    kernels['gaussian'] = gk1d @ gk1d.T

    # Horizontal motion blur
    motion = np.zeros((9, 9), np.float32)
    motion[4, :] = 1.0 / 9
    kernels['motion'] = motion

    return kernels

# -- TEST DRIVER --
def test_all(image_path, init_types, noise_levels, kernel_names):
    B = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernels = get_kernels()
    results = []

    for kernel_name in kernel_names:
        K = kernels[kernel_name]
        for A_init in init_types:
            for noise_B in noise_levels:
                for noise_K in noise_levels:
                    result = richardson_lucy(B, K, iterations=30,
                                             A_init=A_init,
                                             noise_level_B=noise_B,
                                             noise_level_K=noise_K)
                    title = f"{kernel_name}_init-{A_init}_nB-{noise_B}_nK-{noise_K}"
                    out_img = (result * 255).astype(np.uint8)
                    cv2.imwrite(f"restored_{title}.png", out_img)
                    results.append(title)
                    print(f"Saved: {title}")
    return results

# -- TEST EXECUTION --
# Use any grayscale input image you like
image_path = 'blurred_input.png'  # You need to prepare this

# If you want a specific image as initial estimate
donald = cv2.imread('donald_duck.png', cv2.IMREAD_GRAYSCALE)

# Run tests
results = test_all(
    image_path=image_path,
    init_types=['random', 'gray', 'B', donald],
    noise_levels=[0.0, 0.01, 0.05],
    kernel_names=['mean', 'gaussian', 'motion']
)
```

---

### 📝 Notes

* Ensure all input images are grayscale.
* Use your own images like `'blurred_input.png'` or `'donald_duck.png'`.
* Outputs will be saved as `restored_<config>.png`.

Would you like me to provide example input images and kernels, or bundle this into a script with CLI options?

