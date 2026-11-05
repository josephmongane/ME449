# Problem 1
import numpy as np
import modern_robotics as mr

a = [1,2, 0]
b = mr.VecToso3(a)

# Problem 9 matrix exponentials
matrix1 = mr.VecToso3(np.array([1,2,0]))
matrix = np.array([[0,0.5,-1],
                   [-0.5,0,2],
                   [1,-2,0]])

# Problem 1
matrix = np.array([[0,-1,0],
                   [0,0,-1],
                   [1,0,0]])
ans = mr.so3ToVec(mr.MatrixLog3(matrix))
print("\nProblem 1:\n")
print(ans)

# Problem 2
R1 = mr.MatrixExp3((mr.VecToso3(np.array([1,0,0]))*np.pi/2))
R2 = mr.MatrixExp3((mr.VecToso3(np.array([0,0,1]))*np.pi))
R = R1 @ R2
axis, angle = mr.AxisAng3(R)
print("\nProblem 2: \n")
print(axis[0])
print(angle)

#Problem 3 (d)
Tsa = np.array([[0,-1,0,3],
                [0,0,-1,0],
                [1,0,0,0],
                [0,0,0,1]])
Tsb = np.array([[1,0,0,0],
                [0,0,-1,2],
                [0,-1,0,0],
                [0,0,0,1]])
Tab = mr.TransInv(Tsa) @ Tsb
print("\nProblem 3: \n")
print(Tab)

# Problem 3 (e)
T = Tsb
T1 = Tsa @ T
T2 = T @ Tsa
print("\nProblem 3 (e):\n")
print(T1)
print(T2)

# Problem 3 (f)
pb = np.array([1,2,3,1])
p_new = Tsb * pb
print("\nProblem 3 (f): \n")
print(p_new)

# Problem 3 (g)
ps = np.array([1,2,3,1])
pprime = Tsb @ ps
pdoubleprime = mr.TransInv(Tsb) @ ps
print("\nProblem 3 (g): \n")
print(pprime)
print(pdoubleprime)

# Problem 3 (h)
Vs = np.array([3,2,1,-1,-2,-3])
Va = mr.Adjoint(mr.TransInv(Tsa)) @ Vs
print(Va)

#Problem 3 (i)
S = mr.se3ToVec(mr.MatrixLog6(Tsa))
print("\nProblem 3 (i): \n")
print(S)

#Problem 3 (J)
input = np.array([0,1,2,3,0,0])
Tnew = mr.MatrixExp6(mr.VecTose3(input))
print("\nProblem 3 (J): \n")
print(Tnew)

#Problem 5 
Tsb = np.array([[1,0,0,4],
                [0,1,0,1],
                [0,0,1,0],
                [0,0,0,1]])
Sb = np.array([0,0,1,0,2,0])
Ss = np.array([0,0,1,1,-2,0])

bprime = mr.MatrixExp6(mr.VecTose3(Ss*np.pi/2)) @ Tsb
print("\n Problem 5: (d)\n")
print(bprime)
bprime =  Tsb @ mr.MatrixExp6(mr.VecTose3(Ss*np.pi/2)) 
print("\n Problem 5: (e)\n")
print(bprime)
bprime =  mr.MatrixExp6(mr.VecTose3(Sb*np.pi/2)) @ Tsb
print("\n Problem 5: (f)\n")
print(bprime)

#Problem 8;
M = np.array([[1,0,0,0],[0,1,0,2],[0,0,1,1],[0,0,0,1]])
Slist = np.array([[0,0,1,0,0,0],
                  [0,0,1,1,0,0],
                  [0,0,1,2,0,0],
                  [0,0,0,0,0,1]]).T
Blist = np.array([[0,0,1,-2,0,0],
                  [0,0,1,-1,0,0],
                  [0,0,1,0,0,0],
                  [0,0,0,0,0,1]]).T
theta = np.array([0, np.pi/2, -np.pi/2, 1])

T_space = mr.FKinSpace(M, Slist, theta)
T_body  = mr.FKinBody(M, Blist, theta)
print("\nProblem 8:\n")
print(T_space)
print(T_body)

#Problem 9 
W1, W2, L1, L2, H1, H2 = 0.109, 0.082, 0.425, 0.392, 0.089, 0.095

M = np.array([[-1,0,0,L1+L2],
              [ 0,0,1,W1+W2],
              [ 0,1,0,H1-H2],
              [ 0,0,0,1]])

Blist = np.array([[0, 1,0, W1+W2,      0, L1+L2],
                  [0, 0,1, H2,  -(L1+L2),     0],
                  [0, 0,1, H2,       -L2,     0],
                  [0, 0,1, H2,         0,     0],
                  [0,-1,0, -W2,        0,     0],
                  [0, 0,1, 0,          0,     0]]).T

# Check: should equal Ad(M^-1) @ Slist
Slist = np.array([[0,0, 1, 0,     0,     0],
                  [0,1, 0, -H1,   0,     0],
                  [0,1, 0, -H1,   0,     L1],
                  [0,1, 0, -H1,   0,     L1+L2],
                  [0,0,-1, -W1,   L1+L2, 0],
                  [0,1, 0, H2-H1, 0,     L1+L2]]).T
print("\nProblem 9: \n")
print(Blist)

# Problem 11 (a) 
f = np.array([5,0,0])
r = np.array([2+(np.sqrt(2)/2), (np.sqrt(2)/2), 0])
m = np.cross(r, f)
F = np.concat((m,f))

Jspace = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [1, 1, 1],
                   [0, 0,  np.sqrt(2)/2],
                   [0, -1, -(2+np.sqrt(2))/2],
                   [0, 0, 0]])
tau = Jspace.T @ F
print("\nProblem 11 (a):  \n")
print(F)
print(tau)

# Problem 11 (b) 
f = np.array([5,5,0])
r = np.array([2+(np.sqrt(2)/2), (np.sqrt(2)/2), 0])
m = np.cross(r, f)
F = np.concat((m,f))

Jspace = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [1, 1, 1],
                   [0, 0,  np.sqrt(2)/2],
                   [0, -1, -(2+np.sqrt(2))/2],
                   [0, 0, 0]])
tau = Jspace.T @ F
print("\nProblem 11 (b):  \n")
print(F)
print(tau)