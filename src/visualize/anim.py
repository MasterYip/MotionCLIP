import numpy as np
import torch
import imageio
import matplotlib
from textwrap import wrap

# from action2motion
# Define a kinematic tree for the skeletal struture
humanact12_kinematic_chain = [[0, 1, 4, 7, 10],
                              [0, 2, 5, 8, 11],
                              [0, 3, 6, 9, 12, 15],
                              [9, 13, 16, 18, 20, 22],
                              [9, 14, 17, 19, 21, 23]]  # same as smpl

smpl_kinematic_chain = humanact12_kinematic_chain

mocap_kinematic_chain = [[0, 1, 2, 3],
                         [0, 12, 13, 14, 15],
                         [0, 16, 17, 18, 19],
                         [1, 4, 5, 6, 7],
                         [1, 8, 9, 10, 11]]

vibe_kinematic_chain = [[0, 12, 13, 14, 15],
                        [0, 9, 10, 11, 16],
                        [0, 1, 8, 17],
                        [1, 5, 6, 7],
                        [1, 2, 3, 4]]

action2motion_kinematic_chain = vibe_kinematic_chain

# G1 Robot kinematic chain (30 bodies)
# Format: connections between body indices
g1_kinematic_chain = [
    [0, 1, 2, 3, 4, 5, 6],        # Left leg: pelvis -> left foot
    [0, 7, 8, 9, 10, 11, 12],     # Right leg: pelvis -> right foot
    [0, 13, 14, 15],              # Torso: pelvis -> torso
    [15, 16, 17, 18, 19, 20, 21, 22],  # Left arm: torso -> left wrist
    [15, 23, 24, 25, 26, 27, 28, 29],  # Right arm: torso -> right wrist
]

colors_blue = ["#4D84AA", "#5B9965",  "#61CEB9", "#34C1E2", "#80B79A"]
colors_orange = ["#DD5A37", "#D69E00",  "#B75A39", "#FF6D00", "#DDB50E"]
colors_purple = ["#6B31DB", "#AD40A8",  "#AF2B79", "#9B00FF", "#D836C1"]

def add_shadow(img, shadow=15):
    img = np.copy(img)
    mask = img > shadow
    img[mask] = img[mask] - shadow
    img[~mask] = 0
    return img


def load_anim(path, timesize=None):
    data = np.array(imageio.mimread(path, memtest=False))[..., :3]
    if timesize is None:
        return data
    # take the last frame and put shadow repeat the last frame but with a little shadow
    lastframe = add_shadow(data[-1])
    alldata = np.tile(lastframe, (timesize, 1, 1, 1))

    # copy the first frames
    lenanim = data.shape[0]
    alldata[:lenanim] = data[:lenanim]
    return alldata


def plot_3d_motion(motion, length, save_path, params, title="", interval=50, palette=None,
                   view_point=(-90, -90)):
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    from matplotlib.animation import FuncAnimation, writers  # noqa: F401
    # import mpl_toolkits.mplot3d.axes3d as p3
    matplotlib.use('Agg')
    pose_rep = params["pose_rep"]
    appearance = params['appearance_mode']

    fig = plt.figure(figsize=[2.6, 2.8])
    ax = fig.add_subplot(111, projection='3d')
    # ax = p3.Axes3D(fig)
    # ax = fig.gca(projection='3d')

    def init():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_zlim(-0.7, 0.7)

        ax.view_init(azim=view_point[0], elev=view_point[1])
        # ax.set_axis_off()
        ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        if appearance == 'motionclip':
            ax.set_axis_off()

    colors = ['red', 'magenta', 'black', 'green', 'blue']

    if appearance == 'motionclip':
        if palette == 'orange':
            colors = colors_orange
        else:
            colors = colors_blue

    if pose_rep != "xyz":
        raise ValueError("It should already be xyz.")

    if torch.is_tensor(motion):
        motion = motion.numpy()

    # invert axis
    # motion[:, 1, :] = -motion[:, 1, :]
    # motion[:, 2, :] = -motion[:, 2, :]
    # this hack is not needed for amass

    """
    Debug: to rotate the bodies
    import src.utils.rotation_conversions as geometry
    glob_rot = [0, 1.5707963267948966, 0]
    global_orient = torch.tensor(glob_rot)
    rotmat = geometry.axis_angle_to_matrix(global_orient)
    motion = np.einsum("ikj,ko->ioj", motion, rotmat)
    """

    if motion.shape[0] == 18:
        kinematic_tree = action2motion_kinematic_chain
    elif motion.shape[0] == 24:
        kinematic_tree = smpl_kinematic_chain
    else:
        kinematic_tree = None

    # Cache pelvis/root trajectory for ground reference line
    pelvis_traj = motion[0]  # (3, T)

    def update(index):
        while ax.lines:
            ax.lines[0].remove()
        while ax.collections:
            ax.collections[0].remove()
        if kinematic_tree is not None:
            for chain, color in zip(kinematic_tree, colors):
                ax.plot(motion[chain, 0, index],
                        motion[chain, 1, index],
                        motion[chain, 2, index], linewidth=4.0, color=color)
        else:
            ax.scatter(motion[1:, 0, index], motion[1:, 1, index],
                       motion[1:, 2, index], c="red")
            ax.scatter(motion[:1, 0, index], motion[:1, 1, index],
                       motion[:1, 2, index], c="blue")

        # Draw ground reference line using pelvis trajectory (x-z plane)
        traj_len = min(index + 1, pelvis_traj.shape[1])
        ax.plot(pelvis_traj[0, :traj_len],
            pelvis_traj[1, :traj_len],
            pelvis_traj[2, :traj_len],
            color='gray', linewidth=2.0, alpha=0.6)

    wraped_title = '\n'.join(wrap(title, 20))
    ax.set_title(wraped_title)

    ani = FuncAnimation(fig, update, frames=length, interval=interval, repeat=False, init_func=init)

    plt.tight_layout()
    # pillow have problem droping frames
    # ani.save(save_path, writer='ffmpeg', fps=1000/interval)
    # ani.save(save_path, writer='avconv', fps=1000/interval)
    ani.save(save_path, writer='pillow', fps=1000/interval)
    plt.close()


def plot_3d_motion_dico(x):
    motion, length, save_path, params, kargs = x
    plot_3d_motion(motion, length, save_path, params, **kargs)


def plot_3d_motion_g1(motion, length, save_path, params, title="", interval=50, palette=None,
                      view_point=(90, 0)):
    """Plot G1 robot motion with 30 body positions."""
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    matplotlib.use('Agg')
    
    appearance = params.get('appearance_mode', 'motionclip')

    fig = plt.figure(figsize=[2.6, 2.8])
    ax = fig.add_subplot(111, projection='3d')

    def init():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_zlim(-0.7, 0.7)

        ax.view_init(azim=view_point[0], elev=view_point[1])
        ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        if appearance == 'motionclip':
            ax.set_axis_off()

    colors = colors_blue
    if appearance == 'motionclip':
        if palette == 'orange':
            colors = colors_orange
        else:
            colors = colors_blue

    if torch.is_tensor(motion):
        motion = motion.numpy()
    
    # Expected input format: (30, 3, num_frames) like the original plot_3d_motion
    # Keep it in this format for consistency with existing visualization pipeline
    if len(motion.shape) == 3:
        if motion.shape[0] == 30 and motion.shape[1] == 3:
            # Already in correct format (30, 3, T)
            pass
        elif motion.shape[1] == 30 and motion.shape[2] == 3:
            # Format is (T, 30, 3), transpose to (30, 3, T)
            motion = motion.transpose(1, 2, 0)
        elif motion.shape[0] == 3 and motion.shape[1] == 30:
            # Format is (3, 30, T), transpose to (30, 3, T)
            motion = motion.transpose(1, 0, 2)
        else:
            # Try to infer: if first dimension is largest, assume it's T
            if motion.shape[0] > motion.shape[1] and motion.shape[0] > motion.shape[2]:
                # Likely (T, 30, 3) or (T, 3, 30)
                if motion.shape[1] == 30:
                    motion = motion.transpose(1, 2, 0)  # (T, 30, 3) -> (30, 3, T)
                else:
                    motion = motion.transpose(2, 1, 0)  # (T, 3, 30) -> (30, 3, T)
    
    # Verify shape
    if motion.shape[0] != 30 or motion.shape[1] != 3:
        print(f"Warning: Unexpected motion shape {motion.shape}, expected (30, 3, T)")

    kinematic_tree = g1_kinematic_chain

    # Cache pelvis trajectory in world space (before centering), to draw ground reference line
    pelvis_traj = motion[0]  # (3, T)

    def update(index):
        while ax.lines:
            ax.lines[0].remove()
        while ax.collections:
            ax.collections[0].remove()
        
        # Extract positions for current frame: motion[:, :, index] gives (30, 3)
        # Center pelvis to origin for clearer articulation view
        positions_world = motion[:, :, index]  # (30, 3)
        pelvis = positions_world[0:1]
        positions = positions_world - pelvis  # centered skeleton
        
        # Draw kinematic chains
        for chain, color in zip(kinematic_tree, colors):
            chain_pos = positions[chain]  # (len(chain), 3)
            ax.plot(chain_pos[:, 0], chain_pos[:, 1], chain_pos[:, 2],
                   linewidth=4.0, color=color)
        
        # Draw joints as scatter points
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='black', s=20, alpha=0.6)
        
        # Highlight pelvis (root)
        ax.scatter(positions[0:1, 0], positions[0:1, 1], positions[0:1, 2],
                  c='red', s=40)

        # Draw ground reference line using pelvis trajectory (x-z plane), keeps translation context
        traj_len = min(index + 1, pelvis_traj.shape[1])
        pelvis_traj_rel = pelvis_traj - pelvis_traj[:, traj_len-1:traj_len]  # relative to current pelvis
        ax.plot(pelvis_traj_rel[0, :traj_len],
                pelvis_traj_rel[1, :traj_len],
                pelvis_traj_rel[2, :traj_len],
                color='gray', linewidth=2.0, alpha=0.6)

    wraped_title = '\n'.join(wrap(title, 20))
    ax.set_title(wraped_title)

    ani = FuncAnimation(fig, update, frames=length, interval=interval, repeat=False, init_func=init)

    plt.tight_layout()
    ani.save(save_path, writer='pillow', fps=1000/interval)
    plt.close()


def plot_3d_motion_dico_g1(x):
    """Wrapper for G1 motion plotting with dictionary unpacking."""
    motion, length, save_path, params, kargs = x
    plot_3d_motion_g1(motion, length, save_path, params, **kargs)
