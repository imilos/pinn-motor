import deepxde as dde
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.io import loadmat
from scipy.interpolate import CubicSpline

# Parameters
Jm_s = tf.Variable(initial_value=0., dtype=tf.float32, trainable=True, name='Jm_s')

# Transform s into the desired range
def transform_variable(s, lower, upper):
    return lower + (upper - lower) * tf.sigmoid(s)

#Jm_var = transform_variable(Jm_s, 0.0005, 0.0007)

#Jm = 0.000620  # [kg*m2
Jl = 0.000220  # [kg*m2]
cs = 350       # [Nm/rad]
bv = 0.004     # [Nms/rad]
MemMAX = 2.85  # [Nm]
T = 0.043      # [s]

# Load and preprocess data
mat = loadmat('jera1.mat')
t = mat['t'][100:498].flatten() - mat['t'][100]
Mem = mat['Mem'][100:498].flatten()

def Mem_func(t):
    if t < T:
        return MemMAX * np.sin(2 * np.pi / T * t)
    else:
        return 0.

#Mem = np.vectorize(Mem_func)(t)

# Convert interpolation data to TensorFlow tensors
t_tf = tf.constant(t, dtype=tf.float32)
Mem_tf = tf.constant(Mem, dtype=tf.float32)

# Use TensorFlow Probability for cubic interpolation
def tf_cubic_interp(x):
    return tfp.math.interp_regular_1d_grid(x, t_tf[0], t_tf[-1], Mem_tf, axis=0)

def ode_system(x, y):
    Jm_var = transform_variable(Jm_s, 0.0005, 0.0009)
    theta_m, theta_l = y[:, 0:1], y[:, 1:]
    dtheta_m = dde.grad.jacobian(y, x, i=0)
    dtheta_l = dde.grad.jacobian(y, x, i=1)
    d2theta_m = dde.grad.jacobian(dtheta_m, x)
    d2theta_l = dde.grad.jacobian(dtheta_l, x)

    # Use TensorFlow cubic interpolation
    Mem_interp = tf_cubic_interp(x)

    loss1 = Jm_var * d2theta_m + cs * (theta_m - theta_l) + bv * (dtheta_m - dtheta_l) - Mem_interp
    loss2 = Jl * d2theta_l - cs * (theta_m - theta_l) - bv * (dtheta_m - dtheta_l)

    return [loss1, loss2]

def boundary(_, on_initial):
    return on_initial

def error_0_dtheta_m(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 0.

def error_0_dtheta_l(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=1, j=None) - 0.

geom = dde.geometry.TimeDomain(0.0, 1.2*T)
ic1 = dde.icbc.IC(geom, lambda x: 0, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 0, boundary, component=1)
ic3 = dde.icbc.OperatorBC(geom, error_0_dtheta_m, boundary)
ic4 = dde.icbc.OperatorBC(geom, error_0_dtheta_l, boundary)

# Define the 4 points as a numpy array: [t, theta_m, theta_l]
points = np.array([
#    [0.0086, 0.04669, 0.045495],   # (t, theta_m, theta_l)
#[2.05E-02,	3.58E-01,	3.57E-01],
#[2.39E-02,	4.82E-01,	4.82E-01],
#[2.96E-02,	6.89E-01,	6.89E-01],
[3.42E-02,	8.31E-01,   8.32E-01],
[4.12E-02,	9.63E-01,	9.64E-01],
[5.00E-02,	9.96E-01,	9.95E-01],
[5.03E-02,	9.96E-01,	9.95E-01],
[5.06E-02,	9.96E-01,	9.95E-01],
[5.08E-02,	9.96E-01,	9.95E-01],
[5.11E-02,	9.96E-01,	9.95E-01]

])

# Define PointSetBC for theta_m (component 0)
theta_m_bc = dde.icbc.PointSetBC(points[:, 0:1], points[:, 1], component=0)
# Define PointSetBC for theta_l (component 1)
theta_l_bc = dde.icbc.PointSetBC(points[:, 0:1], points[:, 2], component=1)

data = dde.data.PDE(geom, ode_system, [ic1, ic2, ic3, ic4, theta_m_bc, theta_l_bc], 1000, 16, train_distribution='Hammersley')

layer_size = [1] + [80] * 2 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

variable_Jm = dde.callbacks.VariableValue(Jm_s, period=1000)

model.compile("adam", lr=1e-3, external_trainable_variables=[Jm_s])
losshistory, train_state = model.train(epochs=200000, callbacks=[variable_Jm])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

t_test = np.linspace(0, 1.2*T, 200)[:, None]
y_pred = model.predict(t_test)
theta_m = y_pred[:, 1:2]
np.savetxt("theta_l.txt", np.hstack((t_test, theta_m)), header="t, theta_l", delimiter=', ')

Jm_var_value = transform_variable(Jm_s, 0.0005, 0.0009).numpy()
print("Jm calculated: ", Jm_var_value)

