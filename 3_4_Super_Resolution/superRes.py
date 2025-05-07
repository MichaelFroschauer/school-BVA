import cv2
import matplotlib.pyplot as plt

# Load the pre-trained EDSR model
sr = cv2.dnn_superres.DnnSuperResImpl_create() #we need pip3 install opencv-contrib-python
path = "EDSR_x4.pb"  # Make sure you have downloaded the model from https://github.com/Saafke/EDSR_Tensorflow/blob/master/models/EDSR_x4.pb
sr.readModel(path)
sr.setModel("edsr", 4)  # EDSR model with scale factor 4

# Load the input image
# image = cv2.imread("./images/lena_small.png")
# imageOrig = cv2.imread("./images/lena3.png")
# image = cv2.imread("./images/tripod_small.png")
# imageOrig = cv2.imread("./images/tripod1.png")
image = cv2.imread("./images/3.jpg")
imageOrig = cv2.imread("./images/2.jpg")

# Apply super-resolution
result = sr.upsample(image)

diffImage = cv2.absdiff(result, imageOrig)


# Display the original and super-resolution images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('small image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.title('Super-Resolution image')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.title('difference image original')
plt.imshow(cv2.cvtColor(diffImage, cv2.COLOR_BGR2RGB))
plt.show()

#cv2.imwrite("./output/lena_upsampled.png", result)
cv2.imwrite("./output/2_upsampled.png", result)
