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
ans = np.sqrt(3) 
print(str(ans))

#problem 4
print("Problem 4: \n")
space_config = np.array([[0,0,1,0,-1,0],
                   [0,1,0,0,0,1],
                   [0,1,0,1,0,2.732],
                   [0,1,0,-0.732,0,3.732],
                   [0,0,0,0,0,1],
                   [0,0,1,0,-3.732,0]]).T
M = np.array([[1,0,0,3.732],
              [0,1,0,0],
              [0,0,1,2.732],
              [0,0,0,1]])
pi = np.pi
print(pi)
theta_list = np.array([-pi/2,pi/2,pi/3,-pi/4,1,pi/6])
print(theta_list)
new_config = mr.FKinSpace(M, space_config, theta_list)
print(str(new_config))

#problem 5 
print("Problem 5: \n")
body_config = np.array([[0,0,1,0,2.732,0],
                        [0,1,0,2.732,0,-2.732],
                        [0,1,0,3.732,0,-1],
                        [0,1,0,2,0,0],
                        [0,0,0,0,0,1],
                        [0,0,1,0,0,0]]).T
print(str(body_config))
new_config = mr.FKinBody(M,body_config,theta_list)
print(new_config)