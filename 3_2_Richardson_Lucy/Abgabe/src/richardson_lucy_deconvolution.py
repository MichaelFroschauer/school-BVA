import numpy as np
import cv2 as cv

def richardson_lucy(image_blurred, psf, iterations=20, init_guess="gray"):
    # Normalize PSF (Point Spread Function or Kernel) so that the sum of all its values is 1.
    psf = psf / psf.sum()

    # Flip the PSF for convolution. This is a necessary step in the Richardson-Lucy algorithm.
    psf_mirror = np.flip(psf)

    # Initialize estimate A'
    if isinstance(init_guess, str):
        if init_guess == "random":
            # Initialize with random values between 0 and 1.
            estimate = np.random.rand(*image_blurred.shape).astype(np.float32)
        elif init_guess == "blurred":
            # Initialize with the blurred image itself.
            estimate = image_blurred.copy().astype(np.float32)
        elif init_guess == "gray":
            # Initialize with a gray image (middle value of 127 for each pixel).
            estimate = np.full_like(image_blurred, 127, dtype=np.float32)
        else:
            raise ValueError(f"Unknown init_guess value: {init_guess}")
    else:
        # If an image type for initial guess is provided, use it directly.
        estimate = init_guess.astype(np.float32)


    # Iterative RLD algorithm
    for i in range(iterations):
        # Apply the PSF to the current estimate to simulate the blurred version of the estimate.
        estimate_conv = cv.filter2D(estimate, -1, psf, borderType=cv.BORDER_REFLECT)
        # Calculate the ratio of the blurred image to the current estimate's blurred version.
        ratio = image_blurred / (estimate_conv + 1e-7)  # Avoid division by zero
        # Convolve the ratio with the flipped PSF to compute the correction factor.
        correction = cv.filter2D(ratio, -1, psf_mirror, borderType=cv.BORDER_REFLECT)
        # Update the estimate by multiplying it with the correction.
        estimate *= correction

    # Clip the estimate to valid pixel values and convert it back to uint8 format for image display.
    return np.clip(estimate, 0, 255).astype(np.uint8)


def get_kernels():
    # Simple blur (mean)
    # 5x5 kernel with equal weights
    mean_kernel = np.ones((5, 5), np.float32) / 25

    # Gaussian blur kernel using a Gaussian kernel of size 9x9 with standard deviation of 2
    gauss_kernel = cv.getGaussianKernel(9, 2)
    gauss_kernel = gauss_kernel @ gauss_kernel.T # Convert to 2D by multiplying the kernel with its transpose

    # Horizontal motion blur kernel (1D)
    hor_kernel = np.zeros((1, 9))
    hor_kernel[0] = 1.0 / 9

    # Diagonal motion blur kernel (1D)
    diag_kernel = np.eye(9) / 9

    return {
        "mean": mean_kernel,
        "gauss": gauss_kernel,
        "horizontal": hor_kernel,
        "diagonal": diag_kernel,
    }


def add_noise(image, noise_level=5):
    # Generate Gaussian noise with the specified noise level
    noise = np.random.normal(0, noise_level, image.shape)
    # Add noise to the image and clip the result to valid pixel values
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

