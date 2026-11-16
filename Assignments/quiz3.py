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
f = np.array([2,0,0])
r = np.array([2+(np.sqrt(2)/2), (np.sqrt(2)/2), 0])
m = np.cross(r, f)
F = np.concat((m,f))
print(F)

Jspace = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [1, 1, 1],
                   [0, 0,  np.sqrt(2)/2],
                   [0, -1, -(2+np.sqrt(2))/2],
                   [0, 0, 0]])
tau = Jspace.T @ F
print(tau)


th = [0,0,0,np.pi/2, -np.pi/2]
# problem 2 
Jac = np.array([[0,0,0,0],
                [0,0,0,0],
               [1,1,1,1],
               [(np.sin(th[4]) + np.sin(th[3]+th[4]) + np.sin(th[2]+th[3]+th[4])), 
                (np.sin(th[4]) + np.sin(th[3]+th[4])),
                (np.sin(th[4])),
                0],
               [(1 + np.cos(th[3]) + np.cos(th[3]+th[4]) + np.cos(th[2]+th[3]+th[4])),
                (1 + np.cos(th[4]) + np.cos(th[3]+th[4])),
                (1 + np.cos(th[4])),
                (1)],
                [0,0,0,0]]
                )

Fb = np.array([0,0,10,10,10,0])

torq = Jac.T @ Fb
print(torq)

#problem 3

s1 = [0,0,1,0,0,0]
s2 = [1,0,0,0,2,0]
s3 = [0,0,0,0,1,0]

slist = np.array([s1, s2, s3]).T
thlist = np.array([np.pi/2, np.pi/2, 1])

jac = mr.JacobianSpace(slist, thlist)
print("Problem 3\n")
print(jac)

#problem 4 
s1 = [0,1,0,3,0,0]
s2 = [-1,0,0,0,3,0]
s3 = [0,0,0,0,0,1]

slist = np.array([s1, s2, s3]).T
thlist = np.array([np.pi/2, np.pi/2, 1])
jac = mr.JacobianBody(slist, thlist)
print("Problem 4: \n")
print(jac)

#problem 5 
Jb = np.array([
    [-0.105,   0,      0.006, -0.045,   0,     0.006, 0],
    [-0.889,   0.006,  0,     -0.844,   0.006, 0,     0],
    [ 0,      -0.105,  0.889,  0,       0,     0,     0]
])

A = Jb @ Jb.T


print("Problem 5: \n")
val, vecs = np.linalg.eigh(A)          # eigh: A is symmetric, gives real sorted output
i = np.argmax(val)
print("Longest axis direction:", vecs[:, i])
print("Semi-axis lengths:", np.sqrt(val))
print("Problem 6: \n")
val = np.max(val)
print(np.sqrt(0.9916))