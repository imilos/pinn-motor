import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.io

#mat = scipy.io.loadmat('jera1.mat')
#t = mat['t'][100:480].flatten()-mat['t'][100]
#Mem = mat['Mem'][100:480].flatten()

mat = np.loadtxt('Mem_finetuned.txt', delimiter=',')
t = mat[:,0].flatten()
Mem = mat[:,1].flatten()

Mem_interp = interp1d(t, Mem, kind='cubic')

# Constants
Jm=0.000620 # [kg*m2]
Jl=0.000220 # [kg*m2]
cs=350      # [Nm/rad]
bv=0.004    # [Nms/rad]
MemMAX=2.8    # [Nm]
T=0.043     # [s]

# Define the system of ODEs
def odes(t, y):
    theta_m, theta_m_dot, theta_l, theta_l_dot = y
    
    # Equations of motion
    theta_m_ddot = (-cs * (theta_m - theta_l) - bv * (theta_m_dot - theta_l_dot) + Mem_interp(t)) / Jm 
    #theta_m_ddot = (-cs * (theta_m - theta_l) - bv * (theta_m_dot - theta_l_dot) + MemMAX * np.sin(2 * np.pi / T * t)) / Jm
    theta_l_ddot = (cs * (theta_m - theta_l) + bv * (theta_m_dot - theta_l_dot)) / Jl
    
    return [theta_m_dot, theta_m_ddot, theta_l_dot, theta_l_ddot]

# Initial conditions
theta_m_0 = 0.0      # Initial position of mass m
theta_m_dot_0 = 0.0  # Initial velocity of mass m
theta_l_0 = 0.0      # Initial position of mass l
theta_l_dot_0 = 0.0  # Initial velocity of mass l

y0 = [theta_m_0, theta_m_dot_0, theta_l_0, theta_l_dot_0]

# Time span
#t_span = (0, 1.2*T)
#t_eval = np.linspace(0, 1.2*T, 200)
t_span = (np.min(t), np.max(t))
t_eval = t #np.linspace(t_span[0], t_span[1], 500)

# Solve the system
solution = solve_ivp(odes, t_span, y0, t_eval=t_eval)

# Extract the results
t = solution.t
theta_m = solution.y[0]  # Angular position of mass m
theta_l = solution.y[2]  # Angular position of mass l

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, theta_m, label=r"$\theta_m$ (Angular position of m)")
plt.plot(t, theta_l, label=r"$\theta_l$ (Angular position of l)", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Angular Position (rad)")
plt.title("Angular Positions of Two Masses Over Time")
plt.legend()
plt.grid()
plt.show()

t_test = t
np.savetxt("theta_l21.txt", np.hstack((t_test.reshape(-1,1), theta_l.reshape(-1,1))), header="t, theta_l", delimiter=', ')

