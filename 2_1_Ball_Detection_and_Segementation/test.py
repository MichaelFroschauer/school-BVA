
import numpy as np


a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

b = np.array([[104, 105, 106]])

#movementOverlay[motionMask[..., 0] > 0] = [0, 0, 255]

c = a[..., 0]
d = a[:, 0]

print(c)
print(d)
