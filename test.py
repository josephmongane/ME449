import numpy as np
import modern_robotics as mr

a = [1,2, 0]
b = mr.VecToso3(a)

# Problem 9 matrix exponentials
matrix1 = mr.VecToso3(np.array([1,2,0]))
matrix = np.array([[0,0.5,-1],
                   [-0.5,0,2],
                   [1,-2,0]])
# Rotation Axis 
w1 = np.array([0,0,1])
w2 = np.array([0,1,0])
w3 = np.array([0,1,0])
w4 = np.array([0,1,0])
w5 = np.array([0,0,-1])
w6 = np.array([0,1,0])

axis = [w1, w2, w3, w4, w5, w6]

# Rotation Matricies
R13 = np.array([[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]])
Rs2 = np.array([[-0.6964, 0.1736, 0.6964], [-0.1228, -0.9848, 0.1228], [0.7071, 0, 0.7071]])
R25 = np.array([[-0.7566, -0.1198, -0.6428], [-0.1564, 0.9877, 0], [0.6348, 0.1005, -0.7661]])
R12 = np.array([[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]])
R34 = np.array([[0.6428, 0, -0.7660], [0, 1, 0], [0.7660, 0, 0.6428]])
Rs6 = np.array([[0.9418, 0.3249, -0.0859], [0.3249, -0.9456, -0.0151], [-0.0861, -0.0136, -0.9962]])
R6b = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

# Calculations 
Rs1 = Rs2 @ R12.T
R12 = R12
R23 = R12.T @ R13
R34 = R34
R45 = R34.T @ R13.T @ R12 @ R25 
R56 = R25.T @ Rs2.T @ Rs6 
Rsb = Rs6 @ R6b

config = [Rs1, R12, R23, R34, R45, R56]

# Calculating Angles
thetas = []
i = 0
for i in range(len(axis)):
    thetas.append(sum(mr.so3ToVec(mr.MatrixLog3(config[i])) * axis[i]))

print(str(thetas))
print(str(Rsb))