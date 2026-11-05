import numpy as np
import modern_robotics as mr

a = [1,2, 0]
b = mr.VecToso3(a)

# Problem 9 matrix exponentials
matrix1 = mr.VecToso3(np.array([1,2,0]))
matrix = np.array([[0,0.5,-1],
                   [-0.5,0,2],
                   [1,-2,0]])
#matricies
tsa = np.array([[0,-1,0,0],
                [0,0,-1,0],
                [1,0,0,1],
                [0,0,0,1]])
tsb = np.array([[1,0,0,0],
                [0,0,1,2],
                [0,-1,0,0],
                [0,0,0,1]])

pa = np.array([0,0,1])
pb = np.array([0,2,0])

tab = mr.TransInv(tsa) @ tsb
print(str(tsb))

vect = tsb @ np.array([1,2,3,1])
print(str(vect))

twist_s = np.array([3,2,1,-1,-2,-3])

twist_a = mr.Adjoint(mr.TransInv(tsa)) @ twist_s
print("new twist\n")
print(str(twist_a))

ans = mr.MatrixLog6(tsa)
ans2 = mr.se3ToVec(ans)
S, theta = mr.AxisAng6(ans2)
print(str(theta))

stheta = np.array([0,1,2,3,0,0])
ans = mr.MatrixExp6(mr.VecTose3(stheta))
print(str(ans))

# problem 10
wrench = np.array([1,0,0,2,1,0])

new_wrench = mr.Adjoint(tsb) @ wrench
print(str(new_wrench))

# problem 11
trans = np.array([[0,-1,0,3],
                 [0,1,0,0],
                 [0,0,1,1],
                 [0,0,0,1]])
new_trans = mr.TransInv(trans)
print(str(new_trans))

# Problem 12
twist = np.array([1,0,0,0,2,3])
new_mat = mr.VecTose3(twist)
print(str(new_mat))

# problem 13
q, s, h = [np.array([0,0,2]), np.array([1,0,0]), 1]
new_s = mr.ScrewToAxis(q,s,h)
print(str(new_s))

#problem 14
trans = np.array([[0,-1.5708,0,2.3562],
                  [1.5708,0,0,-2.3562],
                  [0,0,0,1],
                  [0,0,0,0]])
new_trans = mr.MatrixExp6(trans)
print(str(new_trans))

#problem 15
trans = np.array([[0,-1,0,3],
                  [1,0,0,0],
                  [0,0,1,1],
                  [0,0,0,1]])
new_trans = mr.MatrixLog6(trans)
print("\n" + str(new_trans))