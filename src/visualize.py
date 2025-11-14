import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_trajectory(trajectory):
    steps, N, _ = trajectory.shape
    fig = plt.figure()
    ax = plt.axes()
    x = trajectory[0, :, 0]
    y = trajectory[0, :, 1]
    scatter = ax.scatter(x, y)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    def update(frame):
        scatter.set_offsets(trajectory[frame, :, :2])
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=30, blit=True)
    return ani
