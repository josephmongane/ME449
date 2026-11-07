import numpy as np
import modern_robotics as mr
import math as math
import matplotlib.pyplot as plt

# Simulating for Ur5 6R robot arm (Modern Robotics, Example 4.5), units in meters
### W1 = 109 mm, W2 = 82 mm, L1 = 425 mm, L2 = 392 mm, H1 = 89 mm, H2 = 95 mm
#    joint 1: (0, 1, 0, W1 + W2, 0, L1 + L2)
#    joint 2: (0, 0, 1, H2, −L1 − L2, 0)
#    joint 3: (0, 0, 1, H2, −L2, 0)
#    joint 4: (0, 0, 1, H2, 0, 0)
#    joint 5: (0, −1, 0, −W2, 0, 0)
#    joint 6: (0, 0, 1, 0, 0, 0)
W1, W2 = 0.109, 0.082
L1, L2 = 0.425, 0.392
H1, H2 = 0.089, 0.095

# Home configuration of the end-effector
M = np.array([[-1, 0, 0, L1 + L2],
              [ 0, 0, 1, W1 + W2],
              [ 0, 1, 0, H1 - H2],
              [ 0, 0, 0, 1]])
# Body screw axes, one per column
B_list = np.array([[0,  1, 0, W1 + W2, 0,        L1 + L2],
                   [0,  0, 1, H2,      -L1 - L2, 0],
                   [0,  0, 1, H2,      -L2,      0],
                   [0,  0, 1, H2,      0,        0],
                   [0, -1, 0, -W2,     0,        0],
                   [0,  0, 1, 0,       0,        0]]).T
# Desired end-effector configuration
Tsd = np.array([[ 0,  0, -1, 0],
                [ 0, -1,  0, 0.5],
                [-1,  0,  0, 0.5],
                [ 0,  0,  0, 1]])

theta_list_large = np.array([0, 0, 0, 0, 0, 0])
theta_list_small = np.array([1.1, -1.2, 1.0, 0.3, -0.3, -1.5])
eomg = 0.001
ev = 0.001
#definng Iking Body iterates
def IKinBodyIterates(Blist, M, Tsd, thetalist, eomg, ev):
    """
    Takes inputs (Iterations, Blist, M, T, thetalist, eomg, ev)
    Outputs ([iter:({Vi:np.array, Mi:np.array, error_V_bi:np.array, ang_error:float, lin_error:float})])
     prints out a report for each iteration of the Newton-Raphson process

     Iteration 3:
        joint vector:
        0.221, 0.375, 2.233, 1.414
        SE(3) end-effector config:
        1.000 0.000 0.000 3.275
        0.000 1.000 0.000 4.162
        0.000 0.000 1.000 -5.732
        0 0 0 1
        error twist V_b: (0.232, 0.171, 0.211, 0.345, 1.367, -0.222)
        angular error ||omega_b||: 0.357
        linear error ||v_b||: 1.427
    """
    iter_data = []
    theta_i = np.array(thetalist, dtype=float)
    Tsb = mr.FKinBody(M, Blist, theta_i)
    iter = 0
    Vb = mr.se3ToVec(mr.MatrixLog6((mr.TransInv(Tsb) @ Tsd)))
    mag_vb = np.linalg.norm(Vb[-3:])
    mag_wb = np.linalg.norm(Vb[:3])
    Jac = mr.JacobianBody(Blist, theta_i)
    iter_theta_list = [theta_i]  # one row per iteration
    # store iteration 0
    iter_data.append({"thetas":theta_i, "posn":Tsb[0:3, 3], "angle_error":mag_wb, "linear_error":mag_vb})
    print_iteration(iter, theta_i, Tsb, Vb, mag_wb, mag_vb)
    while (mag_wb > eomg) or (mag_vb > ev):
        # Update all the values of the interation
        theta_new = theta_i + (np.linalg.pinv(Jac) @ Vb)
        theta_new = np.array([math.atan2(np.sin(th), np.cos(th)) for th in theta_new])
        Tsb = mr.FKinBody(M,Blist,theta_new)
        Vb = mr.se3ToVec(mr.MatrixLog6((mr.TransInv(Tsb) @ Tsd)))
        mag_vb = np.linalg.norm(Vb[-3:])
        mag_wb = np.linalg.norm(Vb[:3])
        Jac = mr.JacobianBody(Blist, theta_new)
        iter = iter + 1
        theta_i = theta_new
        iter_theta_list.append(theta_i)
        # store iteration data 
        iter_data.append({"thetas":theta_i, "posn":Tsb[0:3, 3], "angle_error":mag_wb, "linear_error":mag_vb})
        print_iteration(iter, theta_i, Tsb, Vb, mag_wb, mag_vb)
    return np.array(iter_theta_list), iter_data  # shape (iterations + 1, n_joints)


def print_iteration(iter, theta, Tsb, Vb, mag_wb, mag_vb):
    """Prints the report for one iteration."""
    print(f"Iteration {iter}:")
    print("  joint vector:")
    print("  " + ", ".join(f"{t:.3f}" for t in theta))
    print("  SE(3) end-effector config:")
    for row in Tsb:
        print("  " + " ".join(f"{x:.3f}" for x in row))
    print("  error twist V_b: (" + ", ".join(f"{x:.3f}" for x in Vb) + ")")
    print(f"  angular error ||omega_b||: {mag_wb:.3f}")
    print(f"  linear error ||v_b||: {mag_vb:.3f}")
    print()


def write_csv(filename, ans):
    """Write one csv row per iteration: the joint vector, comma-separated."""
    f = open(filename, "w")
    for row in ans:
        f.write(",".join("%10.6f" % theta for theta in row) + "\n")
    f.close()


ans_large, data_large = IKinBodyIterates(B_list, M, Tsd, theta_list_large, eomg, ev)
ans_small, data_small = IKinBodyIterates(B_list, M, Tsd, theta_list_small, eomg, ev)
write_csv("long_iterates.csv", ans_large)
write_csv("short_iterates.csv", ans_small)

# ---------------- Plots ----------------
labels = ["Large Iterator", "Small iterator"]
runs = [data_large, data_small]
colors = ["#2a78d6", "#eb6834"]  # blue, orange

# Angular and linear error per iteration: two charts, one y-axis each
fig, (ax_ang, ax_lin) = plt.subplots(1, 2, figsize=(11, 4.5))
for data, label, color in zip(runs, labels, colors):
    x = np.arange(len(data))
    angle_array = [i["angle_error"] for i in data]
    distance_array = [i["linear_error"] for i in data]
    ax_ang.plot(x, angle_array, "-o", color=color, linewidth=2, markersize=5, label=label)
    ax_lin.plot(x, distance_array, "-o", color=color, linewidth=2, markersize=5, label=label)
ax_ang.axhline(eomg, color="#52514e", linestyle="--", linewidth=1, label="tolerance")
ax_lin.axhline(ev, color="#52514e", linestyle="--", linewidth=1, label="tolerance")
for ax, title, ylabel in [(ax_ang, "Angular error per iteration", "Angular error ||ω_b|| (rad)"),
                          (ax_lin, "Linear error per iteration", "Linear error ||v_b|| (m)")]:
    ax.set_title(title)
    ax.set_xlabel("Iterations (n)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.legend()
fig.tight_layout()

# 3D end-effector position (x, y, z), lines connecting successive iterations
fig3d = plt.figure(figsize=(7, 6))
ax3d = fig3d.add_subplot(projection="3d")
for data, label, color in zip(runs, labels, colors):
    posn = np.array([i["posn"] for i in data])
    ax3d.plot(posn[:, 0], posn[:, 1], posn[:, 2], "-o", color=color, linewidth=2, markersize=5, label=label)
    ax3d.scatter(*posn[0], color=color, s=80, marker="s")  # start point
ax3d.scatter(*Tsd[0:3, 3], color="#0b0b0b", s=120, marker="*", label="Desired position")
ax3d.set_title("End-effector position per iteration")
ax3d.set_xlabel("x (m)")
ax3d.set_ylabel("y (m)")
ax3d.set_zlabel("z (m)")
ax3d.legend()
fig3d.tight_layout()

fig.savefig("error_plots.png", dpi=150)
fig3d.savefig("position_3d.png", dpi=150)
plt.show()
