from richardson_lucy_deconvolution import *

def showOrSave(name, image):
    cv.imshow(name, image)
    cv.imwrite(f"./output/{name}.png", image)

def tryWithDonald(blurred_img, kernel_name, kernel):
    img_init_guess = cv.imread("./images/donald_duck.png", cv.IMREAD_GRAYSCALE)
    showOrSave(f"initial_guess", img_init_guess)
    restored_donald = richardson_lucy(blurred_img, kernel, iterations=100, init_guess=img_init_guess)
    showOrSave(f"restored_{kernel_name}_donaldDuck", restored_donald)


def test_different_kernels_and_filters():
    img = cv.imread("./images/tripod.png", cv.IMREAD_GRAYSCALE)
    kernels = get_kernels()
    showOrSave(f"original", img)

    # Loop through each kernel and process the image
    for name, kernel in kernels.items():
        blurred = cv.filter2D(img, -1, kernel)
        showOrSave(f"{name}_blurred", blurred)

        # Try different initial guesses and restore the image using Richardson-Lucy deconvolution
        for guess in ["gray", "blurred", "random"]:
            restored = richardson_lucy(blurred, kernel, iterations=100, init_guess=guess)
            showOrSave(f"restored_{name}_{guess}", restored)

        tryWithDonald(blurred, name, kernel)


def test_different_iterations():
    img = cv.imread("./images/tripod.png", cv.IMREAD_GRAYSCALE)
    kernels = get_kernels()
    showOrSave(f"original", img)

    # Loop through each kernel and process the image
    for name, kernel in kernels.items():
        blurred = cv.filter2D(img, -1, kernel)
        showOrSave(f"{name}_blurred", blurred)

        for guess in ["blurred"]:
            restored = richardson_lucy(blurred, kernel, iterations=10, init_guess=guess)
            showOrSave(f"restored_{name}_{guess}_10", restored)

            restored = richardson_lucy(blurred, kernel, iterations=50, init_guess=guess)
            showOrSave(f"restored_{name}_{guess}_50", restored)

            restored = richardson_lucy(blurred, kernel, iterations=100, init_guess=guess)
            showOrSave(f"restored_{name}_{guess}_100", restored)



def test_different_noise_levels():
    img = cv.imread("./images/tripod.png", cv.IMREAD_GRAYSCALE)
    kernels = get_kernels()
    showOrSave(f"original", img)

    # Loop through each kernel and process the image
    for name, kernel in kernels.items():
        blurred = cv.filter2D(img, -1, kernel)
        showOrSave(f"{name}_blurred", blurred)
        guess = "blurred"
        for noise_level in [2, 5, 10]:
            noisy_blurred = add_noise(blurred.copy(), noise_level)
            showOrSave(f"noisy_blurred_{name}_{noise_level}", noisy_blurred)

            restored_noise = richardson_lucy(noisy_blurred, kernel, iterations=100, init_guess=guess)
            showOrSave(f"restored_noisy_blurred_{name}_{noise_level}", restored_noise)


def test_different_images():
    img_name = "galaxy"
    img = cv.imread(f"./images/{img_name}.png", cv.IMREAD_GRAYSCALE)
    kernels = get_kernels()
    showOrSave(f"{img_name}_original", img)

    # Loop through each kernel and process the image
    for name, kernel in kernels.items():
        blurred = cv.filter2D(img, -1, kernel)
        showOrSave(f"{img_name}_{name}_blurred", blurred)

        for guess in ["blurred"]:
            restored = richardson_lucy(blurred, kernel, iterations=100, init_guess=guess)
            showOrSave(f"{img_name}_restored_{name}_{guess}", restored)


if __name__ == '__main__':
    test_different_kernels_and_filters()
    test_different_iterations()
    test_different_noise_levels()
    test_different_images()

    cv.waitKey(0)
    cv.destroyAllWindows()