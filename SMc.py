import numpy as np
import matplotlib.pyplot as plt


class system:

    def __init__(self, m, b, k, nk):
        self.m = m  # mass
        self.b = b  # damping coefficient
        self.k = k  # spring constant
        self.nk = nk  # cross coupling
        self.x = np.array([0, 0, 0])

    def system_dynamics(self, x, xdot, u, dt):
        xdotdot = (u - self.b * xdot - self.k * x - self.nk * x * xdot) / self.m
        xdot = xdot + xdotdot * dt
        x = x + xdot * dt + 0.5 * xdotdot * dt * dt
        self.x = np.array([x, xdot, xdotdot])
        return self.x

    def det_eq_u(self, x, a):
        u = (
            a * self.m
            + self.b * self.x[1]
            + self.k * self.x[0]
            + self.nk * self.x[0] * self.x[1]
        )
        return u





# Define the desired trajectory (e.g., a step input to a target position)
def desired_trajectory(t):
    return 1.0, 0.0  # desired position, desired velocity


class controls:

    def __init__(self, lemda, eta):
        self.lemda = lemda
        self.eta = eta

    def find_u(self, x, xd):
        self.e = x[0] - xd[0]
        self.e_dot = x[1] - xd[1]
        self.s = self.e_dot + self.lemda * self.e
        self.usmd = -self.eta * self.s
        return self.usmd, self.s, self.e, self.e_dot, self.lemda




# Simulation parameters
dt = 0.01  # time step
t_end = 10.0  # simulation end time
time = np.arange(0, t_end, dt)
m = 1.0  # mass
b = 0.5  # damping coefficient
k = 1.0  # spring constant
nk = 0.08  # cross coupling
# Controller parameters
lambda_val = 2  # slope of the sliding surface
eta = 5  # switching gain (adjust for robustness vs. chattering)

# Initial conditions
x = np.array([0.0, 0.0, 0.0])  # initial position, initial velocity

# Store results
positions = []
velocities = []
control_inputs = []
sliding_surfaces = []

element = system(m, b, k, nk)
consys = controls(lambda_val, eta)
# Simulation loop
for t in time:
    x_d, x_d_dot = desired_trajectory(t)

    usmd, s, e, e_dot, lemda = consys.find_u(x, np.array([x_d, x_d_dot]))
    ueq = element.det_eq_u(x, -e_dot * lemda)
    x = element.system_dynamics(x[0], x[1], ueq + usmd, dt)

    # Store data
    positions.append(x[0])
    velocities.append(x[1])
    control_inputs.append(ueq + usmd)
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
