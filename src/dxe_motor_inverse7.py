import deepxde as dde
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.io import loadmat
from scipy.interpolate import CubicSpline

# Parameters
Jm = 0.000620  # [kg*m2]
Jl = 0.000220  # [kg*m2]
cs = 350       # [Nm/rad]
bv = 0.004     # [Nms/rad]
MemMAX = 1.    # 
T = 0.043      # [s]

# Load and preprocess data
mat = np.loadtxt('theta_l211.txt',delimiter=',')
t = mat[:,0].flatten()
theta_l = mat[:,1].flatten()

# Create SciPy interpolant (for computing derivatives)
theta_l_interp1 = CubicSpline(t, theta_l)
dtheta_l1 = theta_l_interp1.derivative(1)
d2theta_l1 = theta_l_interp1.derivative(2)

# Convert data to TensorFlow tensors
t_tf = tf.constant(t, dtype=tf.float32)
theta_l_tf1 = tf.constant(theta_l, dtype=tf.float32)

# TensorFlow interpolation using tfp.math.interp_regular_1d_grid
def theta_l_tf(x):
    return tfp.math.interp_regular_1d_grid(x, t_tf[0], t_tf[-1], theta_l_tf1, axis=0)

# Precompute derivative arrays (as TensorFlow constants)
dtheta_l_tf = tf.convert_to_tensor(dtheta_l1(t), dtype=tf.float32)
d2theta_l_tf = tf.convert_to_tensor(d2theta_l1(t), dtype=tf.float32)

def dtheta_tf(x):
    return tfp.math.interp_regular_1d_grid(x, t_tf[0], t_tf[-1], dtheta_l_tf, axis=0)

def d2theta_tf(x):
    return tfp.math.interp_regular_1d_grid(x, t_tf[0], t_tf[-1], d2theta_l_tf, axis=0)


def ode_system(x, y):
    theta_m, Mem = y[:, 0:1], y[:, 1:2]

    theta_l_interp = theta_l_tf(x)
    dtheta_l = dtheta_tf(x)
    d2theta_l = d2theta_tf(x)

    dtheta_m = dde.grad.jacobian(y, x, i=0)
    d2theta_m = dde.grad.jacobian(dtheta_m, x)

    loss1 = Jm * d2theta_m + cs * (theta_m - theta_l_interp) + bv * (dtheta_m - dtheta_l) - Mem
    loss2 = Jl * d2theta_l - cs * (theta_m - theta_l_interp) - bv * (dtheta_m - dtheta_l)

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

anchor_points = t.reshape(-1, 1)[t<1.2*T]
data = dde.data.PDE(geom, ode_system, [ic1, ic2], 8000, 8, train_distribution='Hammersley', anchors=anchor_points)

layer_size = [1] + [80] * 2 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)


model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=100000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Save the trained model
model.save("model.ckpt")

#t_test = np.linspace(0, 1.2*T, 200)[:, None]
t_test = anchor_points
y_pred = model.predict(t_test)
Mem = y_pred[:, 1:2]
np.savetxt("Mem_original.txt", np.hstack((t_test, Mem)), header="t, Mem", delimiter=', ')



# --- RESTORE AND FINE-TUNE ---

# Redefine the PDE 
data_new = dde.data.PDE(geom, ode_system, [ic1, ic2], 8000, 8, train_distribution='Hammersley', anchors=anchor_points)

# Reinitialize model with restored weights
model_new = dde.Model(data_new, net)
model_new.compile("adam", lr=2e-4)  # Recompile with potentially new learning rate

# Make a dummy forward pass to initialize the model
dummy_x = np.array([[0.0]])  # A single time point to trigger variable initialization
model_new.predict(dummy_x)

model_new.net.load_weights("model.ckpt-100000.weights.h5")  # Load saved weights


# Train again
losshistory, train_state = model_new.train(epochs=400000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Save the fine-tuned model
model_new.save("model_finetuned.ckpt")

# Save results
#t_test = np.linspace(0, 1.2*T, 200)[:, None]
t_test = anchor_points
y_pred = model_new.predict(t_test)
Mem = y_pred[:, 1:2]
np.savetxt("Mem_finetuned.txt", np.hstack((t_test, Mem)), header="t, Mem", delimiter=', ')
