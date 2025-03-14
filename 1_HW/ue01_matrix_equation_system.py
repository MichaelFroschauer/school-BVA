import numpy as np

# Given pairs of points
points_P = np.array([
    [1, 4],
    [-4, -2],
    [0.1, 5],
    [-1, 2],
    [3, 3],
    [7, -2],
    [5, 5],
    [-6, 3.3]
])
points_P_prime = np.array([
    [-1.26546, 3.222386],
    [-4.53286, 0.459128],
    [-1.64771, 3.831308],
    [-2.57985, 2.283247],
    [-0.28072, 2.44692],
    [1.322025, -0.69344],
    [1.021729, 3.299737],
    [-5.10871, 3.523542]
])


# Converting the points into a form that we can use for solving Ax = b
A = np.zeros((2 * len(points_P), 4))
b = np.zeros((2 * len(points_P), 1))

# Convert to
#   a    b    Tx   Ty
# [ x1  -y1   1    0  ]   [ a ]    [ x1' ]
# [ y1   x1   0    1  ] * [ b ] =  [ y1' ]
# [ x2  -y2   1    0  ]   [ Tx ]   [ x2' ]
# [ y2   x2   0    1  ]   [ Ty ]   [ y2' ]
#   ...                   ...      ...
for i, p in enumerate(points_P):
    A[2*i]   = [p[0], -p[1], 1, 0]
    A[2*i+1] = [p[1], p[0], 0, 1]
    b[2*i]   = points_P_prime[i][0]
    b[2*i+1] = points_P_prime[i][1]

# Solve using the method of the least squares
x, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)


# Extract transformation parameters from the least squares solution
a_opt = x[0, 0]   # Scaling factor in x-direction combined with rotation (a = s * cos(θ))
b_opt = x[1, 0]   # Scaling factor in y-direction combined with rotation (b = s * sin(θ))
Tx_opt = x[2, 0]  # Translation in x-direction
Ty_opt = x[3, 0]  # Translation in y-direction

# Calculate scaling and rotation
s_opt = np.sqrt(a_opt**2 + b_opt**2)
rot_opt = np.arctan2(b_opt, a_opt)
rot_deg_opt = np.degrees(rot_opt)

print(f"Optimierte Parameter:")
print(f"  Rotation: {rot_deg_opt:.4f} Grad")
print(f"  Skalierung: {s_opt:.4f}")
print(f"  Translation Tx: {Tx_opt:.4f}")
print(f"  Translation Ty: {Ty_opt:.4f}")