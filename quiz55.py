import modern_robotics as mr
import numpy as np

Tf = 5  # Total time
t = 3  # Current time
s = mr.QuinticTimeScaling(Tf, t)
print(s)  # Output: ~0.16308

Xstart = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
Xend   = np.array([[0,0,1,1],
                   [1,0,0,2],
                   [0,1,0,3],
                   [0,0,0,1]])
output = mr.ScrewTrajectory(Xstart=Xstart, Xend=Xend, Tf=10, N=10, method=3)
print(output[-2])
output = mr.CartesianTrajectory(Xstart=Xstart, Xend=Xend, Tf=10, N=10, method=5)
print("\n")
print(output[-2])
