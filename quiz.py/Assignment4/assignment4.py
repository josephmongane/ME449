import modern_robotics as mr
import numpy as np
import math as math

def write_csv(filename, ans):
    """Write one csv row per iteration: the joint vector, comma-separated."""
    f = open(filename, "w")
    for row in ans:
        f.write(",".join("%10.6f" % theta for theta in row) + "\n")
    f.close()

def Puppet(thetalist0, dthetalist0, g, Mlist, Slist, Glist, t, dt, damp, stiff, rlen):
    """
    Function Inpus 
    \n thetalist: an n-vector of initial joint angles (units: rad)
    \n dthetalist: an n-vector of initial joint rates (units: rad/s)
    \n g: the gravity 3-vector in the {s} frame (units: m/s2)
    \n Mlist: the configurations of the link frames relative to each other at the home configuration.
    \n Slist: the screw axes Si in the space frame when the robot is at its home configuration
    \n Glist: the spatial inertia matrices Gi of the links 
    \n t: the total simulation time (units: s)
    \n dt: the simulation timestep (units: s)
    \n damping: a scalar indicating the viscous damping at each joint (units: Nms/rad)
    \n stiffness: a scalar indicating the stiffness of the springy string (units: N/m)
    \n restLength: a scalar indicating the length of the spring when it is at rest (units: m)
    Outputs 
    - thetamat - 
    - dthetamat
    """
    thetamat = []
    dthetamat = []
    thetalist = thetalist0
    dthetalist = dthetalist0
    t_vect = np.arange(0,t,dt)
    # Home configuration of the end-effector frame in {s}
    M_ee = np.eye(4)
    for Mi in Mlist:
        M_ee = M_ee @ np.array(Mi)
    for t in t_vect:
        # Step in time
        cur_posn = refrencePos(t)
        end_config_space = mr.FKinSpace(M_ee, Slist, thetalist)
        R_sb, end_vec_space = mr.TransToRp(end_config_space)
        spring_vec = cur_posn - end_vec_space
        spring_len = np.linalg.norm(spring_vec)
        # Spring force on the end-effector, expressed in {s}
        if spring_len > rlen:
            f_s = stiff*(spring_len - rlen)*spring_vec/spring_len
        else:
            f_s = np.zeros(3)
        # Ftip is the wrench the robot applies to the environment, in the end-effector frame {b}.
        # The spring pulls on the robot, so the robot applies -f; force acts at the {b} origin (no moment).
        Ftip = np.r_[np.zeros(3), -R_sb.T @ f_s]
        # Joint torques: viscous damping only
        tau_list = -damp*np.array(dthetalist)
        ddthetalist = mr.ForwardDynamics(thetalist, dthetalist, tau_list, g, Ftip, Mlist, Glist, Slist)
        thetalist_next, dthetalist_next = mr.EulerStep(thetalist, dthetalist, ddthetalist, dt)
        thetamat.append(thetalist_next)
        dthetamat.append(dthetalist_next)
        thetalist, dthetalist = thetalist_next, dthetalist_next
    return thetamat, dthetamat 

def refrencePos(time):
    """
    Input = Time 
    Output = [x y z] vector of the spring position in time 
    """
    x, y, z = 1, 2, 3
    return np.array([x, y, z])

# Part 1 simulating the falling Robot 
th0 = [0,0,0,0,0,0] # Initial joint Angles
dtheta = [0,0,0,0,0,0] # Initial Joint angles 
g = [0, 0, -9.81] #  gravity vector 

# UR5 Kinematic parameters
M01 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
M12 = [[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]
M23 = [[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]
M34 = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]
M45 = [[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]
M56 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]
M67 = [[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]]

# Inertial Matrix for each linkage 
G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])

# Spatial inertia Matrix 
Glist = [G1, G2, G3, G4, G5, G6] 

# Confiurations of the center of mass of each linkage relative to prev
Mlist = [M01, M12, M23, M34, M45, M56, M67] 

# List of screw axis at home configuration
Slist = [[0,         0,         0,         0,        0,        0],
         [0,         1,         1,         1,        0,        1],
         [1,         0,         0,         0,       -1,        0],
         [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
         [0,         0,         0,         0,  0.81725,        0],
         [0,         0,     0.425,   0.81725,        0,  0.81725]]

#---------Part 1-------------#
t = 5 
dt = 0.01
damp = 0
stiff = 0
rlen = 0
traj, dtraj = Puppet(th0, dtheta, g, Mlist, Slist, Glist, t, dt, damp, stiff, rlen)
print("\nResults of Part 1\n")
print(traj)
write_csv("Theta_traj", traj)
print(dtraj)
