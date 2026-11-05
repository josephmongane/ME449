import numpy as np
import modern_robotics as mr

#problem 2 
Tsd = np.array([[-0.585, -0.811, 0, 0.076],
                [0.811, -0.585, 0, 2.608],
                [0,0,1,0],
                [0,0,0,1]])
Jac = np.array([[0,0,1,0,3,0],
                [0,0,1,0,2,0],
                [0,0,1,0,1,0]]).T
Mnot = np.array([[1,0,0,3],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]])
thnot = np.array([0.7854,0.7854,0.7854])
ev = 0.0001
ew = 0.001
angles = mr.IKinBody(Jac, Mnot, Tsd, thnot, ew, ev)
print(angles)