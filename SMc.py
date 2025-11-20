import numpy as np
import matplotlib.pyplot as plt


# Define the system dynamics (e.g., a simple mass-spring-damper)
def system_dynamics(x, u, t):
    m = 1.0  # mass
    b = 0.5  # damping coefficient
    k = 1.0  # spring constant
    nk=0.08 #cross coupling

    # State variables: x[0] = position, x[1] = velocity
    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = (u - b * x[1] * x[1] - k * x[0] * x[0] - nk * x[0] * x[1]) / m
    return x_dot


# Define the desired trajectory (e.g., a step input to a target position)
def desired_trajectory(t):
    return 1.0, 0.0  # desired position, desired velocity


# Sliding Mode Control implementation
def sliding_mode_controller(x, x_d, x_d_dot, lambda_val, eta):
    # Error definition
    e = x[0] - x_d
    e_dot = x[1] - x_d_dot

    # Sliding surface definition (s = e_dot + lambda * e)
    s = e_dot + lambda_val * e
   

    # Equivalent control (for nominal system)
    # Assuming u_eq = m * (x_d_ddot - lambda * e_dot) + b * x[1] + k * x[0]
    # For simplicity here, we'll assume x_d_ddot is 0 for a constant desired position
    # and focus on the switching part for robustness.
    u_eq = 0  # In a full implementation, this would compensate for known dynamics

    # Switching control (to drive s to zero)
    u_sw = -eta * np.sign(s)

    # Total control input
    u = u_eq + u_sw
    return u,  s


# Simulation parameters
dt = 0.01  # time step
t_end = 10.0  # simulation end time
time = np.arange(0, t_end, dt)

# Controller parameters
lambda_val = 2.0  # slope of the sliding surface
eta = 5.0  # switching gain (adjust for robustness vs. chattering)

# Initial conditions
x = np.array([0.0, 0.0])  # initial position, initial velocity

# Store results
positions = []
velocities = []
control_inputs = []
sliding_surfaces = []
e_dot_l = 0
# Simulation loop
for t in time:
    x_d, x_d_dot = desired_trajectory(t)

    u, s = sliding_mode_controller(x, x_d, x_d_dot, lambda_val, eta)

    # Apply control input and update system state
    x_dot = system_dynamics(x, u, t)
    x += x_dot * dt

    # Store data
    positions.append(x[0])
    velocities.append(x[1])
    control_inputs.append(u)
    sliding_surfaces.append(s)

# Plotting results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, positions, label="Actual Position")
plt.plot(
    time, [desired_trajectory(t)[0] for t in time], "r--", label="Desired Position"
)
plt.ylabel("Position")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, velocities, label="Actual Velocity")
plt.plot(
    time, [desired_trajectory(t)[1] for t in time], "r--", label="Desired Velocity"
)
plt.ylabel("Velocity")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, control_inputs, label="Control Input")
plt.ylabel("Control Input")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plotting sliding surface
plt.figure()
plt.plot(time, sliding_surfaces)
plt.title("Sliding Surface Evolution")
plt.xlabel("Time (s)")
plt.ylabel("Sliding Surface (s)")
plt.grid(True)
plt.show()
