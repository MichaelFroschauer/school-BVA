import math
import numpy as np

p1 = np.array([0, 0, 0])
p2 = np.array([3, 4, 0])

r = np.linalg.norm(p1 - p2).astype(float)
print(r)



def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(-0.5 * (distance / bandwidth) ** 2)



def gaussian_weight(dist, bandwidth):
    return np.exp(- (dist ** 2) / (2 * bandwidth ** 2))


print("gaussian_kernel: " + str(gaussian_kernel(3.0, 4.5)))
print("gaussian_weight: " + str(gaussian_weight(3.0, 4.5)))