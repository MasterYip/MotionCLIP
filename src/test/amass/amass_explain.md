# AMASS Dataset vs G1 Retargeted AMASS Dataset Explained

## Overview

This document explains the difference between the original AMASS dataset (SMPL-based) and the G1 retargeted AMASS dataset (robot-based), including the meaning of each field and how they're used in MotionCLIP.

---

## About SMPL Models

**SMPL** (Skinned Multi-Person Linear model):
- **24 joints** total (1 root + 23 body joints)
- **6890 vertices** in the mesh
- No detailed hand or face modeling
- Joint names from `SMPL_JOINT_NAMES` (see smpl_utils.py)

**SMPL+H** (SMPL + Hands):
- **52 joints** total (22 base body + 30 hand joints)
- **6890 vertices** (same body mesh as SMPL)
- Includes MANO hand model with detailed finger articulation
- **This is what AMASS uses**
- Joint structure:
  - Joints 0-21: Base body (pelvis, spine, legs, arms, head)
  - Joints 22-36: Left hand (15 finger joints)
  - Joints 37-51: Right hand (15 finger joints)
- Joint names from `SMPLH_JOINT_NAMES` (see smpl_utils.py)

**SMPL-X** (SMPL eXpressive):
- **55 joints** for body + **145 total** including detailed face
- **10475 vertices** (body + hands + face)
- Includes hands, facial expressions, and eye gaze
- Most expressive but computationally expensive
- Joint names from `JOINT_NAMES` (see smpl_utils.py)

---

## Original AMASS Dataset

AMASS (Archive of Motion Capture as Surface Shapes) is a large-scale motion capture dataset that unifies 15+ different MoCap datasets into a common SMPL body model representation.

### File Structure
```
ACCAD/Female1Gestures_c3d/D2 - Wait 1_poses.npz
```

### Data Fields

#### 1. **poses** - SMPL+H Body Pose Parameters
- **Shape**: `(4624, 156)` 
- **Dtype**: float64
- **Range**: -1.87 to 3.07 (radians)
- **Meaning**: 
  - SMPL+H pose parameters in **axis-angle representation**
  - 156 dimensions = 52 joints × 3 (axis-angle per joint)
  - **Joint Structure** (according to `smpl_utils.py`):
    - **Joints 0-21 (Base body)**: 66 parameters (22 × 3)
      - Joint 0: `pelvis` (root/global orientation)
      - Joints 1-21: body joints (hips, spine, legs, arms, neck, head, collars, shoulders, elbows, wrists)
      - See `BASE_JOINTS` and `SMPLH_JOINT_NAMES[0:22]` in smpl_utils.py
    - **Joints 22-36 (Left hand)**: 45 parameters (15 × 3)
      - Starts at `L_HAND_START_INDEX = 22`
      - Includes: index, middle, pinky, ring, thumb (3 joints each)
      - See `L_HAND_FULL` in smpl_utils.py
    - **Joints 37-51 (Right hand)**: 45 parameters (15 × 3)
      - Starts at `R_HAND_START_INDEX = 37`
      - Includes: index, middle, pinky, ring, thumb (3 joints each)
      - See `R_HAND_FULL` in smpl_utils.py
  - Each joint rotation is represented as **axis-angle** (rotation axis × rotation angle)
  - Total: 22 + 15 + 15 = **52 joints**
  
#### 2. **trans** - Global Translation
- **Shape**: `(4624, 3)`
- **Dtype**: float64
- **Range**: -0.41 to 0.93 (meters)
- **Meaning**: 
  - Global root (pelvis) position in 3D space (x, y, z)
  - Represents where the character is in the world coordinate system
  - Changes over time as the character moves through space

#### 3. **betas** - Shape Parameters
- **Shape**: `(16,)`
- **Dtype**: float64
- **Range**: -1.98 to 2.81
- **Meaning**:
  - SMPL shape parameters (PCA coefficients)
  - Control body shape variations (height, weight, proportions)
  - Constant per sequence (doesn't change over time)
  - 16 principal components (can be 10 or 16 depending on SMPL version)

#### 4. **dmpls** - Dynamic Shape Deformations
- **Shape**: `(4624, 8)`
- **Dtype**: float64
- **Range**: -7.05 to 5.48
- **Meaning**:
  - DMPL (Dynamic Mesh Pose Learning) parameters
  - Capture soft-tissue dynamics and pose-dependent deformations
  - Model muscle bulging, skin stretching, etc.
  - Time-varying corrections to the SMPL body shape

#### 5. **gender** - Subject Gender
- **Shape**: `()`
- **Dtype**: string
- **Value**: "female" (or "male", "neutral")
- **Meaning**:
  - Specifies which SMPL model variant to use
  - Different SMPL models for male/female/neutral body types

#### 6. **mocap_framerate** - Capture Frame Rate
- **Shape**: `()`
- **Dtype**: float64
- **Value**: 120.0 (fps)
- **Meaning**:
  - Original motion capture sampling rate
  - In this case: 120 frames per second
  - Note: G1 retargeted version is at 30 fps (downsampled)

### SMPL+H Joint Names (52 Joints)

From `SMPLH_JOINT_NAMES` in smpl_utils.py:

**Base Body Joints (0-21)**:
```python
0: 'pelvis'           # Root
1: 'left_hip'         # Left leg
2: 'right_hip'        # Right leg
3: 'spine1'           # Lower spine
4: 'left_knee'
5: 'right_knee'
6: 'spine2'           # Mid spine
7: 'left_ankle'
8: 'right_ankle'
9: 'spine3'           # Upper spine
10: 'left_foot'
11: 'right_foot'
12: 'neck'
13: 'left_collar'
14: 'right_collar'
15: 'head'
16: 'left_shoulder'
17: 'right_shoulder'
18: 'left_elbow'
19: 'right_elbow'
20: 'left_wrist'
21: 'right_wrist'
```

**Left Hand Joints (22-36)**:
```python
22: 'left_index1'     # Index finger base
23: 'left_index2'     # Index finger mid
24: 'left_index3'     # Index finger tip
25: 'left_middle1'
26: 'left_middle2'
27: 'left_middle3'
28: 'left_pinky1'
29: 'left_pinky2'
30: 'left_pinky3'
31: 'left_ring1'
32: 'left_ring2'
33: 'left_ring3'
34: 'left_thumb1'
35: 'left_thumb2'
36: 'left_thumb3'
```

**Right Hand Joints (37-51)**:
```python
37: 'right_index1'
38: 'right_index2'
39: 'right_index3'
40: 'right_middle1'
41: 'right_middle2'
42: 'right_middle3'
43: 'right_pinky1'
44: 'right_pinky2'
45: 'right_pinky3'
46: 'right_ring1'
47: 'right_ring2'
48: 'right_ring3'
49: 'right_thumb1'
50: 'right_thumb2'
51: 'right_thumb3'
```

### G1 Robot Structure (29 DOFs, 30 Bodies)

#### G1 DOF Names (29 Actuated Joints)

**Left Leg (DOFs 0-5)**:
```python
0: 'left_hip_pitch_joint'
1: 'left_hip_roll_joint'
2: 'left_hip_yaw_joint'
3: 'left_knee_joint'
4: 'left_ankle_pitch_joint'
5: 'left_ankle_roll_joint'
```

**Right Leg (DOFs 6-11)**:
```python
6: 'right_hip_pitch_joint'
7: 'right_hip_roll_joint'
8: 'right_hip_yaw_joint'
9: 'right_knee_joint'
10: 'right_ankle_pitch_joint'
11: 'right_ankle_roll_joint'
```

**Waist/Torso (DOFs 12-14)**:
```python
12: 'waist_yaw_joint'
13: 'waist_roll_joint'
14: 'waist_pitch_joint'
```

**Left Arm (DOFs 15-21)**:
```python
15: 'left_shoulder_pitch_joint'
16: 'left_shoulder_roll_joint'
17: 'left_shoulder_yaw_joint'
18: 'left_elbow_joint'
19: 'left_wrist_roll_joint'
20: 'left_wrist_pitch_joint'
21: 'left_wrist_yaw_joint'
```

**Right Arm (DOFs 22-28)**:
```python
22: 'right_shoulder_pitch_joint'
23: 'right_shoulder_roll_joint'
24: 'right_shoulder_yaw_joint'
25: 'right_elbow_joint'
26: 'right_wrist_roll_joint'
27: 'right_wrist_pitch_joint'
28: 'right_wrist_yaw_joint'
```

#### G1 Body Names (30 Rigid Bodies)

**Root**:
```python
0: 'pelvis'
```

**Left Leg (Bodies 1-6)**:
```python
1: 'left_hip_pitch_link'
2: 'left_hip_roll_link'
3: 'left_hip_yaw_link'
4: 'left_knee_link'
5: 'left_ankle_pitch_link'
6: 'left_ankle_roll_link'
```

**Right Leg (Bodies 7-12)**:
```python
7: 'right_hip_pitch_link'
8: 'right_hip_roll_link'
9: 'right_hip_yaw_link'
10: 'right_knee_link'
11: 'right_ankle_pitch_link'
12: 'right_ankle_roll_link'
```

**Torso (Bodies 13-15)**:
```python
13: 'waist_yaw_link'
14: 'waist_roll_link'
15: 'torso_link'
```

**Left Arm (Bodies 16-22)**:
```python
16: 'left_shoulder_pitch_link'
17: 'left_shoulder_roll_link'
18: 'left_shoulder_yaw_link'
19: 'left_elbow_link'
20: 'left_wrist_roll_link'
21: 'left_wrist_pitch_link'
22: 'left_wrist_yaw_link'
```

**Right Arm (Bodies 23-29)**:
```python
23: 'right_shoulder_pitch_link'
24: 'right_shoulder_roll_link'
25: 'right_shoulder_yaw_link'
26: 'right_elbow_link'
27: 'right_wrist_roll_link'
28: 'right_wrist_pitch_link'
29: 'right_wrist_yaw_link'
```

### How SMPL Works

**SMPL (Skinned Multi-Person Linear model)** is a parametric body model:

```
SMPL(β, θ, Φ) → Mesh
```

Where:
- **β (betas)**: Shape parameters - define body shape
- **θ (poses)**: Pose parameters - define joint rotations
- **Φ (dmpls)**: Dynamic parameters - add soft-tissue dynamics

**Forward Kinematics Process**:
1. Start with template mesh (6890 vertices)
2. Apply shape blendshapes using β → shaped body
3. Apply pose blendshapes using θ → posed body
4. Apply skinning (Linear Blend Skinning) → final mesh
5. Apply dynamic deformations using Φ → realistic deformations

**Joint Locations**: Extracted from mesh using a joint regressor matrix

---

## G1 Retargeted AMASS Dataset

This is the same AMASS motion data, but retargeted to the Unitree G1 humanoid robot. The human motions are converted to robot-feasible joint trajectories.

### File Structure
```
ACCAD/Female1Gestures_c3d/D2-Wait1_poses_120_jpos.npz
```
Note: `_120_jpos` suffix indicates 120Hz original data, retargeted to joint positions

### Data Fields

#### 1. **dof_positions** - Robot Joint Angles
- **Shape**: `(1156, 29)`
- **Dtype**: float32
- **Range**: -2.28 to 1.88 (radians)
- **Meaning**:
  - Joint angle trajectories for G1 robot
  - 29 DOFs (Degrees of Freedom) = actuated joints on G1
  - Each column corresponds to one joint (see `dof_names`)
  - Direct control values for robot motors
  - **This is the primary motion data for G1**

#### 2. **dof_velocities** - Robot Joint Velocities
- **Shape**: `(1156, 29)`
- **Dtype**: float32
- **Range**: -32.43 to 16.36 (rad/s)
- **Meaning**:
  - Angular velocities for each joint
  - Derivative of `dof_positions` with respect to time
  - Useful for dynamic simulation and control
  - Can be used for velocity-based losses during training

#### 3. **body_positions** - Robot Body 3D Positions
- **Shape**: `(1156, 30, 3)`
- **Dtype**: float32
- **Range**: -0.36 to 1.35 (meters)
- **Meaning**:
  - 3D positions (x, y, z) of each robot body/link
  - 30 bodies/links in G1 robot
  - Result of forward kinematics from `dof_positions`
  - **This is used for xyz-based training in MotionCLIP**
  - Can be visualized as skeleton stick figure

#### 4. **body_rotations** - Robot Body Orientations
- **Shape**: `(1156, 30, 4)`
- **Dtype**: float32
- **Range**: -0.99 to 0.99
- **Meaning**:
  - Orientation of each robot body in quaternion format (w, x, y, z)
  - Quaternions are normalized: w² + x² + y² + z² = 1
  - Result of forward kinematics
  - Can be converted to rotation matrices or axis-angle if needed

#### 5. **body_linear_velocities** - Body Linear Velocities
- **Shape**: `(1156, 30, 3)`
- **Dtype**: float32
- **Range**: -1.92 to 1.80 (m/s)
- **Meaning**:
  - Linear velocity (vx, vy, vz) of each body's center of mass
  - Derivative of `body_positions`
  - Useful for physics simulation and momentum tracking

#### 6. **body_angular_velocities** - Body Angular Velocities
- **Shape**: `(1156, 30, 3)`
- **Dtype**: float32
- **Range**: -7.39 to 15.06 (rad/s)
- **Meaning**:
  - Angular velocity (ωx, ωy, ωz) of each body
  - Rotational motion dynamics
  - Useful for physics simulation

#### 7. **dof_names** - Joint Names
- **Shape**: `(29,)`
- **Dtype**: string (Unicode)
- **Sample**: `['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', ...]`
- **Meaning**:
  - Names of all actuated joints in G1
  - Mapping for `dof_positions` columns
  - **G1 Joint Structure** (29 DOFs total):
    - **Left Leg** (6 DOFs): hip (pitch/roll/yaw), knee, ankle (pitch/roll)
    - **Right Leg** (6 DOFs): hip (pitch/roll/yaw), knee, ankle (pitch/roll)
    - **Waist/Torso** (3 DOFs): waist (yaw/roll/pitch)
    - **Left Arm** (7 DOFs): shoulder (pitch/roll/yaw), elbow, wrist (roll/pitch/yaw)
    - **Right Arm** (7 DOFs): shoulder (pitch/roll/yaw), elbow, wrist (roll/pitch/yaw)
  - Total: 6 + 6 + 3 + 7 + 7 = **29 actuated joints**

#### 8. **body_names** - Body/Link Names
- **Shape**: `(30,)`
- **Dtype**: string
- **Sample**: `['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', ...]`
- **Meaning**:
  - Names of all rigid bodies/links in G1
  - Mapping for `body_positions` and `body_rotations`
  - 30 bodies = 29 joints + 1 (root/pelvis)
  - Forms kinematic tree structure
  - See "G1 Robot Structure" section below for complete listing

#### 9. **fps** - Frame Rate
- **Shape**: `(1,)`
- **Dtype**: float32
- **Value**: 30.0
- **Meaning**:
  - Temporal sampling rate: 30 frames per second
  - Downsampled from original 120 fps AMASS data
  - Time between frames: Δt = 1/30 ≈ 0.0333 seconds

---

## Key Differences Summary

| Aspect | Original AMASS (SMPL+H) | G1 Retargeted |
|--------|-------------------------|---------------|
| **Body Model** | SMPL+H (6890 vertices mesh) | G1 Robot (30 rigid bodies) |
| **Primary Data** | `poses` (156 params = 52×3) | `dof_positions` (29 joint angles) |
| **Joints/DOFs** | 52 joints (22 body + 30 hands) | 29 actuated joints (physical) |
| **Joint Types** | Virtual (SMPL+H FK) | Physical robot joints |
| **Representation** | Axis-angle rotations | Joint angles (radians) |
| **3D Positions** | Computed via SMPL+H FK | Direct `body_positions` (30×3) |
| **Hands** | Detailed (15 joints each) | No individual fingers |
| **Shape Variation** | `betas` shape params | Fixed robot geometry |
| **Frame Rate** | 120 fps (original) | 30 fps (downsampled) |
| **File Size** | 4624 frames @ 120fps | 1156 frames @ 30fps |
| **Use Case** | Human motion synthesis | Robot motion control |

---

## Retargeting Process

The conversion from AMASS (human) to G1 (robot) involves:

1. **SMPL Forward Kinematics**: poses + betas → 3D joint positions
2. **IK Solving**: Map human joints → G1 robot joints
3. **Constraint Enforcement**: Apply robot joint limits and feasibility
4. **Downsampling**: 120 fps → 30 fps for robot control
5. **Physics Validation**: Ensure dynamics are physically plausible

### Why Different Frame Counts?
- Original: 4624 frames @ 120 fps = **38.53 seconds**
- Retargeted: 1156 frames @ 30 fps = **38.53 seconds**
- Same duration! Just different sampling rates: 4624/4 = 1156 ✓

---

## Usage in MotionCLIP

### Original AMASS Pipeline
```python
# Load SMPL+H data
poses = data['poses']        # (T, 156) = (T, 52 joints × 3)
trans = data['trans']        # (T, 3)
betas = data['betas']        # (16,)

# Reshape poses to joint format
rotations = poses.reshape(T, 52, 3)  # Axis-angle per joint
# rotations[:, 0:22, :] -> Base body (22 joints)
# rotations[:, 22:37, :] -> Left hand (15 joints)
# rotations[:, 37:52, :] -> Right hand (15 joints)

# Convert to rotation representation (e.g., rot6d)
rot6d = axis_angle_to_rot6d(rotations)  # (T, 52, 6)

# SMPL+H forward kinematics for xyz
# Note: SMPL returns 24 joints, extra joints via J_regressor
joints3d = SMPLH(poses, betas, trans)  # (T, 52, 3) or (T, 23, 3)

# Network input
input = rot6d  # or joints3d for xyz representation
```

### G1 Retargeted Pipeline
```python
# Load G1 data
dof_pos = data['dof_positions']      # (T, 29)
body_pos = data['body_positions']    # (T, 30, 3)

# Option 1: Use DOF positions directly
input = dof_pos  # (T, 29) - joint angles

# Option 2: Use body positions (recommended)
input = body_pos  # (T, 30, 3) - xyz coordinates

# No SMPL forward kinematics needed!
# body_positions are already computed
```

### Why Use body_positions for G1?
- ✅ **Direct xyz data** - no FK computation needed
- ✅ **Physically grounded** - actual robot body locations
- ✅ **Same as SMPL joints3d** - compatible with existing losses
- ✅ **Easier to visualize** - 3D positions are intuitive

---

## Data Validation

### Checking Consistency

**Frame Rate Check**:
```python
# Original AMASS
duration = 4624 / 120.0 = 38.53 seconds

# G1 Retargeted
duration = 1156 / 30.0 = 38.53 seconds
# ✓ Matches!
```

**DOF vs Bodies**:
```python
# G1 has one more body than DOFs
num_dofs = 29      # Actuated joints
num_bodies = 30    # 29 links + 1 root (pelvis)
# ✓ Consistent with kinematic tree structure
```

**Quaternion Normalization**:
```python
# body_rotations should be unit quaternions
quat = body_rotations[frame, body, :]  # (4,)
norm = sqrt(w² + x² + y² + z²)
assert abs(norm - 1.0) < 1e-6  # Should be ~1.0
```

---

## Common Questions

### Q: Why 29 DOFs for G1?
**A**: G1 is a humanoid robot with:
- 6 DOFs per leg × 2 = 12 DOFs (hip pitch/roll/yaw, knee, ankle pitch/roll)
- 7 DOFs per arm × 2 = 14 DOFs (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
- 3 DOFs for waist/torso = 3 DOFs (waist yaw/roll/pitch)
- **Total: 29 actuated joints**

### Q: Why 30 bodies but 29 DOFs?
**A**: The root body (pelvis) is the base of the kinematic tree. It has 6 degrees of freedom (3 position + 3 rotation) in world space but isn't an actuated joint. Each of the 29 DOFs connects two bodies.

### Q: What's the difference between dof_positions and body_rotations?
**A**: 
- `dof_positions`: **Joint angles** (1 value per joint) - what you control
- `body_rotations`: **Body orientations** (4 values per body) - result of FK

### Q: Can I use dof_velocities for training?
**A**: Yes! You can:
- Add velocity loss: `loss = MSE(pred_velocities, true_velocities)`
- Use for data augmentation
- Enforce smooth motion dynamics

### Q: How to visualize G1 motions?
**A**: Use `body_positions` to draw a stick figure skeleton:
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

positions = body_positions[frame]  # (30, 3)
# Define bone connections based on kinematic tree
# Plot lines between connected bodies
```

---

## References

1. **AMASS**: [https://amass.is.tue.mpg.de/](https://amass.is.tue.mpg.de/)
2. **SMPL**: Loper et al., "SMPL: A Skinned Multi-Person Linear Model", 2015
3. **DMPL**: Pons-Moll et al., "DMPL: Dynamic Mesh Pose Learning", 2015
4. **Unitree G1**: Humanoid robot specifications
5. **Motion Retargeting**: Villegas et al., "Neural Kinematic Networks", 2018

---

## File Naming Conventions

### Original AMASS
```
{dataset}/{subject}/{sequence}_poses.npz
Example: ACCAD/Female1Gestures_c3d/D2 - Wait 1_poses.npz
```

### G1 Retargeted
```
{dataset}/{subject}/{sequence}_poses_{original_fps}_jpos.npz
Example: ACCAD/Female1Gestures_c3d/D2-Wait1_poses_120_jpos.npz
```
- `_120_jpos`: Indicates 120Hz source, retargeted to joint positions
- Spaces removed/replaced with hyphens for compatibility
