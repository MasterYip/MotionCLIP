# Detailed Guide: Adapting MotionCLIP for Unitree G1 Robot

## Overview

This guide provides comprehensive instructions for adapting MotionCLIP to work with Unitree G1 retargeted AMASS dataset. The key difference is that G1 data contains `dof_positions`, `dof_velocities`, `body_positions`, `body_rotations`, `body_linear_velocities`, and `body_angular_velocities`, eliminating the need for SMPL model inference.

---

## Part 1: Dataset Class Modifications

### 1.1 Create New G1 Dataset Class

**File to Create:** `src/datasets/g1_amass.py`

The current AMASS dataset class loads SMPL parameters (thetas and joints3d) [1](#0-0) . You need to create a new dataset class that:

**Key Requirements:**

1. **Inherit from the base Dataset class** [2](#0-1) 

2. **Data Loading Structure:**
   - Current AMASS loads data from `.pt` files with structure: `{'thetas': [...], 'joints3d': [...], 'clip_images': [...], ...}` [3](#0-2) 
   - Your G1 data should be structured as `.pt` or `.pkl` files with: `{'fps', 'dof_names': [...], 'body_names': [...], 'dof_positions': [...], 'dof_velocities': [...], 'body_positions': [...], 'body_rotations': [...], 'body_linear_velocities': [...], 'body_angular_velocities': [...]}`

3. **Implement Required Methods:**

   a. **`__init__` method:** Similar structure to AMASS dataset initialization [4](#0-3) , but:
      - Load your G1 `.pt` file instead
      - Store `_dof_positions`, `_body_positions`, `_body_rotations` instead of `_poses` and `_joints3d`
      - Set appropriate number of joints for G1 (G1 has different joint count than SMPL's 23 joints)
      - Still support `seq_len = 100` frame chunking [5](#0-4) 

   b. **`_load_joints3D` method:** [6](#0-5) 
      ```python
      def _load_joints3D(self, ind, frame_ix):
          # Return body_positions directly from G1 data
          joints3D = self._body_positions[ind][frame_ix]
          return joints3D
      ```

   c. **`_load_rotvec` method:** [7](#0-6) 
      - If `body_rotations` are in axis-angle format, return them directly
      - If using `dof_positions`, return those (shape: `[num_frames, num_dofs, 1]` - may need reshaping)
      - Ensure output shape matches expected format: `[num_frames, num_joints, 3]`

4. **Handle Rotation Convention:**
   - Current code uses `ROT_CONVENTION_TO_ROT_NUMBER` [8](#0-7) 
   - For G1, you need to define how many DOFs per joint (e.g., if G1 has N DOFs total and M bodies, define the mapping)

### 1.2 Update Dataset Factory

**File to Modify:** `src/datasets/get_dataset.py`

1. Add import: `from .g1_amass import G1AMASS`
2. Modify `get_dataset` function [9](#0-8) :
   ```python
   def get_dataset(name="amass"):
       if name == "g1_amass":
           return G1AMASS
       return AMASS
   ```
3. Update `get_datasets` function [10](#0-9)  to support conditional loading

---

## Part 2: Remove SMPL Dependency

### 2.1 Create G1ToXyz Class (SMPL Replacement)

**File to Create:** `src/models/g1_to_xyz.py`

The current `Rotation2xyz` class uses SMPL to convert rotations to 3D positions [11](#0-10) . For G1:

**Implementation Strategy:**

Since G1 data already contains `body_positions`, you have two options:

**Option A: Pass-through (Recommended if you already have xyz):**
```python
class G1ToXyz:
    def __init__(self, device):
        self.device = device
    
    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, **kwargs):
        # If pose_rep is already "xyz", just return x
        if pose_rep == "xyz":
            return x
        
        # If you have body_positions in the data, extract them
        # This assumes your dataset._load() returns xyz when requested
        # Otherwise, implement G1 forward kinematics here
        return x  # or implement conversion from dof_positions to body_positions
```

**Option B: Forward Kinematics (If using dof_positions):**
- Implement G1's kinematic chain
- Convert `dof_positions` → `body_positions` using G1's URDF/joint structure
- This is similar to what SMPL does but for G1 robot [12](#0-11) 

### 2.2 Modify MotionCLIP Model

**File to Modify:** `src/models/modeltype/motionclip.py`

1. **Update imports at the top:**
   - Add conditional import for G1ToXyz
   - Current: [13](#0-12) 

2. **Modify `__init__` method:** [14](#0-13) 
   - Add parameter to select between SMPL and G1 mode
   - Replace `self.rotation2xyz = Rotation2xyz(device=self.device)` with conditional:
     ```python
     if use_g1:
         from ..g1_to_xyz import G1ToXyz
         self.rotation2xyz = G1ToXyz(device=self.device)
     else:
         self.rotation2xyz = Rotation2xyz(device=self.device)
     ```

3. **No changes needed for `rot2xyz` method** [15](#0-14)  - it will use the G1ToXyz class automatically

4. **Verify loss computation** [16](#0-15)  - should work unchanged since losses operate on xyz space

### 2.3 Update Model Factory

**File to Modify:** `src/models/get_model.py`

Pass G1-specific parameters when creating the model [17](#0-16) :
- Add `use_g1` parameter to parameters dict
- Pass it to MOTIONCLIP constructor

---

## Part 3: Configuration and Training Setup

### 3.1 Update Configuration

**File to Modify:** `src/config.py`

Add G1-specific configuration [18](#0-17) :
```python
# G1 Robot Configuration
G1_NUM_DOFS = 37  # Example: adjust to actual G1 DOF count
G1_NUM_BODIES = 20  # Example: adjust to actual G1 body count
G1_JOINT_NAMES = [...]  # List of G1 joint names
ROT_CONVENTION_TO_ROT_NUMBER['g1'] = G1_NUM_DOFS
```

### 3.2 Update Dataset Parser

**File to Modify:** `src/parser/dataset.py`

Update dataset choices [19](#0-18) :
```python
group.add_argument("--dataset", required=True, 
                   help="Dataset to load", 
                   default='amass',
                   choices=['amass', 'g1_amass'])
```

### 3.3 Update Model Options Parser

**File to Modify:** `src/parser/model.py`

Add G1-specific options [20](#0-19) :
```python
group.add_argument("--use_g1", action='store_true', 
                   help="Use G1 robot instead of SMPL")
group.add_argument("--g1_num_joints", default=20, type=int,
                   help="Number of bodies/joints in G1")
```

### 3.4 Update Model and Data Loading

**File to Modify:** `src/utils/get_model_and_data.py`

The function already handles dataset and model creation [21](#0-20) . Ensure:
- G1 dataset is instantiated when `parameters['dataset'] == 'g1_amass'`
- Model receives `use_g1=True` parameter

---

## Part 4: Data Format Specifications

### 4.1 Expected G1 Data Structure

Your G1 `.pt` file should contain:
```python
{
    'dof_positions': List[np.ndarray],      # [N_sequences] each [seq_len, num_dofs]
    'dof_velocities': List[np.ndarray],     # [N_sequences] each [seq_len, num_dofs]
    'body_positions': List[np.ndarray],     # [N_sequences] each [seq_len, num_bodies, 3]
    'body_rotations': List[np.ndarray],     # [N_sequences] each [seq_len, num_bodies, 3] (axis-angle)
    'body_linear_velocities': List[np.ndarray],   # [N_sequences] each [seq_len, num_bodies, 3]
    'body_angular_velocities': List[np.ndarray],  # [N_sequences] each [seq_len, num_bodies, 3]
    # Optional CLIP-related data (same as AMASS)
    'clip_images': List[np.ndarray],        # [N_sequences] each [n_sub_seq] 
    'clip_text': List[np.ndarray],          # [N_sequences] each [seq_len]
}
```

### 4.2 Coordinate System Alignment

The current AMASS dataset centers motions by subtracting the first frame's root position [22](#0-21) . Ensure:
- G1 `body_positions` are in the same coordinate frame
- Root body (typically pelvis/base) is at index 0
- Apply same centering: `joints3D = joints3D - joints3D[0, 0, :]`

---

## Part 5: Architecture Considerations

### 5.1 Input Feature Dimensions

The encoder expects input of shape `[njoints, nfeats, nframes]` [23](#0-22) :

- **njoints**: Number of G1 bodies (not DOFs) - e.g., 20
- **nfeats**: Feature dimension per joint
  - If using `pose_rep="xyz"`: nfeats = 3 (x, y, z positions)
  - If using `pose_rep="rot6d"`: nfeats = 6 (rotation representation)
  - If using `pose_rep="rotvec"`: nfeats = 3 (axis-angle)
  - With translation: nfeats += 3 (padded)

Current code computes: `self.input_feats = self.njoints * self.nfeats` [24](#0-23) 

### 5.2 Dataset Update Parameters

The dataset must update model parameters [25](#0-24) :
- Ensure G1 dataset correctly sets `self.njoints` (number of bodies)
- Ensure G1 dataset correctly sets `self.nfeats` (features per body)

---

## Part 6: Loss Functions

### 6.1 Verify Loss Compatibility

All loss functions should work without modification:

1. **Reconstruction Loss (rc):** Operates on rotation space [26](#0-25) 
2. **XYZ Reconstruction Loss (rcxyz):** Operates on 3D positions [27](#0-26) 
3. **Velocity Loss (vel):** Operates on rotation space [28](#0-27) 
4. **XYZ Velocity Loss (velxyz):** Operates on 3D positions [29](#0-28) 

**Important:** Since G1 has different joint topology than SMPL, verify that:
- Position-based losses (rcxyz, velxyz) work correctly
- Rotation-based losses (rc, vel) handle G1's DOF structure

---

## Part 7: Training Pipeline

### 7.1 No Changes Required

The training loop should work without modification [30](#0-29)  because:
- It operates on generic batches from DataLoader
- All data processing happens in dataset and model

### 7.2 Collate Function

The collate function should work as-is [31](#0-30) , but verify:
- Batch dimension handling for G1's joint count
- CLIP text/image handling (if using)

---

## Part 8: Testing and Validation

### 8.1 Unit Tests to Create

1. **Test G1 dataset loading:**
   - Verify data shapes: `[batch, njoints, nfeats, nframes]`
   - Verify frame sampling works correctly
   - Verify masking for variable-length sequences

2. **Test G1ToXyz:**
   - If pass-through: verify input == output for xyz
   - If forward kinematics: verify joint positions are reasonable

3. **Test model forward pass:**
   - Run small batch through encoder → decoder
   - Verify output shapes match input shapes
   - Verify loss computation runs without errors

### 8.2 Visualization

To visualize G1 motions, you'll need to create G1-specific visualization code (the current visualization uses SMPL mesh). Consider:
- Stick figure visualization using `body_positions`
- 3D skeleton animation
- Comparison with ground truth G1 data

---

## Part 9: Step-by-Step Implementation Checklist

### Phase 1: Dataset (Priority: HIGH)
- [ ] Create `src/datasets/g1_amass.py` with G1AMASS class
- [ ] Implement `__init__`, `load_db`, `_load_joints3D`, `_load_rotvec`
- [ ] Update `src/datasets/get_dataset.py` to include G1AMASS
- [ ] Test dataset loading standalone

### Phase 2: Model (Priority: HIGH)
- [ ] Create `src/models/g1_to_xyz.py` with G1ToXyz class
- [ ] Update `src/models/modeltype/motionclip.py` to conditionally use G1ToXyz
- [ ] Update `src/models/get_model.py` to pass G1 parameters
- [ ] Test model forward pass with dummy data

### Phase 3: Configuration (Priority: MEDIUM)
- [ ] Add G1 constants to `src/config.py`
- [ ] Update `src/parser/dataset.py` for G1 dataset option
- [ ] Update `src/parser/model.py` for G1 model options
- [ ] Test argument parsing

### Phase 4: Training (Priority: HIGH)
- [ ] Verify training loop works with G1 data
- [ ] Monitor loss curves (should be similar to SMPL training)
- [ ] Verify checkpoint saving/loading

### Phase 5: Validation (Priority: MEDIUM)
- [ ] Create G1-specific visualization tools
- [ ] Validate generated motions look reasonable
- [ ] Compare with ground truth G1 motions

---

## Part 10: Key Differences Summary

| Component | SMPL Version | G1 Version |
|-----------|-------------|------------|
| **Input Data** | `thetas` (SMPL params) + `joints3d` | `dof_positions`, `body_positions`, `body_rotations` |
| **Number of Joints** | 23 (SMPL skeleton) | Variable (e.g., 20 for G1) |
| **Rotation Representation** | Axis-angle for 23 joints | Can use axis-angle or DOF positions |
| **XYZ Conversion** | SMPL forward kinematics | Direct from `body_positions` or G1 FK |
| **Model Path** | Uses SMPL `.pkl` files [32](#0-31)  | No model files needed |
| **Joint Regressor** | Extra joints via regressor [33](#0-32)  | Not needed |

---

## Notes

1. **Pose Representation Choice**: For G1, `pose_rep="xyz"` is simplest since you already have `body_positions`. This avoids rotation conversion complexity [34](#0-33) .

2. **Translation Flag**: The `translation` parameter adds root translation to the representation [35](#0-34) . For G1, keep it enabled if your data has global position drift.

3. **Global Orientation**: SMPL uses `glob` parameter for global orientation [36](#0-35) . G1 should handle base orientation similarly.

4. **CLIP Integration**: CLIP text/image training works independently of motion representation, so all CLIP-related code should work unchanged [37](#0-36) .

5. **Batch Processing**: Ensure G1 data files are split by train/val/test following AMASS convention [38](#0-37) .

### Citations

**File:** src/datasets/amass.py (L136-280)
```python
class AMASS(Dataset):
    dataname = "amass"

    def __init__(self, datapath="data/amass/amass_30fps_legacy_db.pt", split="train", use_z=1, **kwargs):
        assert '_db.pt' in datapath
        self.datapath = datapath.replace('_db.pt', '_{}.pt'.format(split))
        assert os.path.exists(self.datapath)
        print('datapath used by amass is [{}]'.format(self.datapath))
        super().__init__(**kwargs)

        self.dataname = "amass"

        # FIXME - hardcoded:
        self.rot_convention = 'legacy'
        self.use_betas = False
        self.use_gender = False
        self.use_body_features = False
        if 'clip_preprocess' in kwargs.keys():
            self.clip_preprocess = kwargs['clip_preprocess']

        self.use_z = (use_z != 0)

        # keep_actions = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
        dummy_class = [0]
        genders = config.GENDERS
        self.num_classes = len(dummy_class)

        self.db = self.load_db()
        self._joints3d = []
        self._poses = []
        self._num_frames_in_video = []
        self._actions = []
        self._betas = []
        self._genders = []
        self._heights = []
        self._masses = []
        self._clip_images = []
        self._clip_texts = []
        self._clip_pathes = []
        self._actions_cat = []
        self.clip_label_text = "text_raw_labels"  # "text_proc_labels"

        seq_len = 100
        n_sequences = len(self.db['thetas'])
        # split sequences
        for seq_idx in range(n_sequences):
            n_sub_seq = self.db['thetas'][seq_idx].shape[0] // seq_len
            if n_sub_seq == 0: continue
            n_frames_in_use = n_sub_seq * seq_len
            joints3d = np.split(self.db['joints3d'][seq_idx][:n_frames_in_use], n_sub_seq)
            poses = np.split(self.db['thetas'][seq_idx][:n_frames_in_use], n_sub_seq)
            self._joints3d.extend(joints3d)
            self._poses.extend(poses)
            self._num_frames_in_video.extend([seq_len] * n_sub_seq)

            if 'action_cat' in self.db:
                self._actions_cat.extend(np.split(self.db['action_cat'][seq_idx][:n_frames_in_use], n_sub_seq))

            if self.use_betas:
                self._betas.extend(np.split(self.db['betas'][seq_idx][:n_frames_in_use], n_sub_seq))
            if self.use_gender:
                self._genders.extend([str(self.db['genders'][seq_idx]).replace("b'female'", "female").replace("b'male'",
                                                                                                              "male")] * n_sub_seq)
            if self.use_body_features:
                self._heights.extend([self.db['heights'][seq_idx]] * n_sub_seq)
                self._masses.extend([self.db['masses'][seq_idx]] * n_sub_seq)
            if 'clip_images' in self.db.keys():
                images = [np.squeeze(e) for e in np.split(self.db['clip_images'][seq_idx][:n_sub_seq], n_sub_seq)]
                processed_images = [self.clip_preprocess(Image.fromarray(img)) for img in images]
                self._clip_images.extend(processed_images)
            if self.clip_label_text in self.db:
                self._clip_texts.extend(np.split(self.db[self.clip_label_text][seq_idx][:n_frames_in_use], n_sub_seq))
            if 'clip_pathes' in self.db:
                self._clip_pathes.extend(np.split(self.db['clip_pathes'][seq_idx][:n_sub_seq], n_sub_seq))
            if 'clip_images_emb' in self.db.keys():
                self._clip_images_emb.extend(np.split(self.db['clip_images_emb'][seq_idx][:n_sub_seq], n_sub_seq))



            actions = [0] * n_sub_seq
            self._actions.extend(actions)

        assert len(self._num_frames_in_video) == len(self._poses) == len(self._joints3d) == len(self._actions)
        if self.use_betas:
            assert len(self._poses) == len(self._betas)
        if self.use_gender:
            assert len(self._poses) == len(self._genders)
        if 'clip_images' in self.db.keys():
            assert len(self._poses) == len(self._clip_images)

        self._actions = np.array(self._actions)
        self._num_frames_in_video = np.array(self._num_frames_in_video)

        N = len(self._poses)
        # same set for training and testing
        self._train = np.arange(N)
        self._test = np.arange(N)

        self._action_to_label = {x: i for i, x in enumerate(dummy_class)}
        self._label_to_action = {i: x for i, x in enumerate(dummy_class)}

        self._gender_to_label = {x: i for i, x in enumerate(genders)}
        self._label_to_gender = {i: x for i, x in enumerate(genders)}

        self._action_classes = idx_to_action_label

    def load_db(self):
        # Load amass dataset encoded to a .db file
        # The loaded data is structured:
        # {
        #     'theta': [data_size, 82] (float64) (structured [pose(72), betas(10)])
        #     'vid_name': [data_size] (str)
        # }
        # data_size should be [16275369]
        db_file = self.datapath
        db = joblib.load(db_file)

        if 'clip_images' in db and db['clip_images'][0] is None:  # No images added
            del db['clip_images']

        return db

    def _load_joints3D(self, ind, frame_ix):
        joints3D = self._joints3d[ind][frame_ix]
        return joints3D

    def _load_rotvec(self, ind, frame_ix):
        pose = self._poses[ind][frame_ix, :].reshape(-1, ROT_CONVENTION_TO_ROT_NUMBER[self.rot_convention] + 1,
                                                     3)  # +1 for global orientation
        return pose

    def _load_betas(self, ind, frame_ix):
        betas = self._betas[ind][frame_ix].transpose((1, 0))
        return betas

    def _load_gender(self, ind, frame_ix):
        gender = self._gender_to_label[self._genders[ind]]
        return gender

    def _load_body_features(self, ind, frame_ix):
        return {'mass': float(self._masses[ind]), 'height': float(self._heights[ind])}


if __name__ == "__main__":
    dataset = AMASS()
```

**File:** src/datasets/dataset.py (L14-36)
```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=1, sampling="conseq", sampling_step=1, split="train",
                 pose_rep="rot6d", translation=True, glob=True, max_len=-1, min_len=-1, num_seq_max=-1, **kwargs):
        self.num_frames = num_frames
        self.sampling = sampling
        self.sampling_step = sampling_step
        self.split = split
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.max_len = max_len
        self.min_len = min_len
        self.num_seq_max = num_seq_max

        self.use_action_cat_as_text_labels = kwargs.get('use_action_cat_as_text_labels', False)
        self.only_60_classes = kwargs.get('only_60_classes', False)
        self.leave_out_15_classes = kwargs.get('leave_out_15_classes', False)
        self.use_only_15_classes = kwargs.get('use_only_15_classes', False)

        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"{self.split} is not a valid split")

        super().__init__()
```

**File:** src/datasets/dataset.py (L103-142)
```python
    def _load(self, ind, frame_ix):
        pose_rep = self.pose_rep
        if pose_rep == "xyz" or self.translation:
            if getattr(self, "_load_joints3D", None) is not None:
                # Locate the root joint of initial pose at origin
                joints3D = self._load_joints3D(ind, frame_ix)
                joints3D = joints3D - joints3D[0, 0, :]
                ret = to_torch(joints3D)
                if self.translation:
                    ret_tr = ret[:, 0, :]
            else:
                if pose_rep == "xyz":
                    raise ValueError("This representation is not possible.")
                if getattr(self, "_load_translation") is None:
                    raise ValueError("Can't extract translations.")
                ret_tr = self._load_translation(ind, frame_ix)
                ret_tr = to_torch(ret_tr - ret_tr[0])

        if pose_rep != "xyz":
            if getattr(self, "_load_rotvec", None) is None:
                raise ValueError("This representation is not possible.")
            else:
                pose = self._load_rotvec(ind, frame_ix)
                if not self.glob:
                    pose = pose[:, 1:, :]
                pose = to_torch(pose)
                if pose_rep == "rotvec":
                    ret = pose
                elif pose_rep == "rotmat":
                    ret = geometry.axis_angle_to_matrix(pose).view(*pose.shape[:2], 9)
                elif pose_rep == "rotquat":
                    ret = geometry.axis_angle_to_quaternion(pose)
                elif pose_rep == "rot6d":
                    ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
        if pose_rep != "xyz" and self.translation:
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
            padded_tr[:, :3] = ret_tr
            ret = torch.cat((ret, padded_tr[:, None]), 1)
        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float()
```

**File:** src/datasets/dataset.py (L365-372)
```python
    def update_parameters(self, parameters):
        for i in range(len(self)):
            if self[i] is not None:
                self.njoints, self.nfeats, _ = self[i]['inp'].shape
                break
        parameters["num_classes"] = self.num_classes
        parameters["nfeats"] = self.nfeats
        parameters["njoints"] = self.njoints
```

**File:** src/config.py (L1-30)
```python
import os

SMPL_DATA_PATH = "./models/smpl"
SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')

SMPLH_AMASS_PATH = './models/smplh'
SMPLH_AMASS_MODEL_PATH = os.path.join(SMPLH_AMASS_PATH, "neutral/model.npz")
SMPLH_AMASS_MALE_MODEL_PATH = os.path.join(SMPLH_AMASS_PATH, "male/model.npz")
SMPLH_AMASS_FEMALE_MODEL_PATH = os.path.join(SMPLH_AMASS_PATH, "female/model.npz")

SMPLX_DATA_PATH = "models/smplx/"
SMPLX_MODEL_PATH = os.path.join(SMPLX_DATA_PATH, "SMPLX_NEUTRAL.pkl")
SMPLX_MALE_MODEL_PATH = os.path.join(SMPLX_DATA_PATH, "SMPLX_MALE.pkl")
SMPLX_FEMALE_MODEL_PATH = os.path.join(SMPLX_DATA_PATH, "SMPLX_FEMALE.pkl")

JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')

JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')

ROT_CONVENTION_TO_ROT_NUMBER = {
    'legacy': 23,
    'no_hands': 21,
    'full_hands': 51,
    'mitten_hands': 33,
}

GENDERS = ['neutral', 'male', 'female']
NUM_BETAS = 10
```

**File:** src/datasets/get_dataset.py (L3-4)
```python
def get_dataset(name="amass"):
    return AMASS
```

**File:** src/datasets/get_dataset.py (L7-31)
```python
def get_datasets(parameters, clip_preprocess, split="train"):
    DATA = AMASS

    if split == 'all':
        train = DATA(split='train', clip_preprocess=clip_preprocess, **parameters)
        test = DATA(split='vald', clip_preprocess=clip_preprocess, **parameters)

        # add specific parameters from the dataset loading
        train.update_parameters(parameters)
        test.update_parameters(parameters)
    else:
        dataset = DATA(split=split, clip_preprocess=clip_preprocess, **parameters)
        train = dataset

        # test: shallow copy (share the memory) but set the other indices
        from copy import copy
        test = copy(train)
        test.split = test

        # add specific parameters from the dataset loading
        dataset.update_parameters(parameters)

    datasets = {"train": train,
                "test": test}

```

**File:** src/models/rotation2xyz.py (L8-88)
```python
class Rotation2xyz:
    def __init__(self, device):
        self.device = device
        self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, get_rotations_back=False, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]
            rotations = rotations[:, 1:]

        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
            # import ipdb; ipdb.set_trace()
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)

        # get the desirable joints
        joints = out[jointstype]

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz
```

**File:** src/models/modeltype/motionclip.py (L7-7)
```python
from ..rotation2xyz import Rotation2xyz
```

**File:** src/models/modeltype/motionclip.py (L16-54)
```python
class MOTIONCLIP(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, clip_lambdas={}, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.outputxyz = outputxyz

        self.lambdas = lambdas
        self.clip_lambdas = clip_lambdas

        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans

        self.clip_model = kwargs['clip_model']
        self.clip_training = kwargs.get('clip_training', False)
        if self.clip_training and self.clip_model:
            self.clip_model.training = True
        else:
            if self.clip_model:
                assert self.clip_model.training == False  # make sure clip is frozen

        self.losses = list(self.lambdas) + ["mixed"]

        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}
```

**File:** src/models/modeltype/motionclip.py (L56-59)
```python
    def rot2xyz(self, x, mask, get_rotations_back=False, **kwargs):
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask, get_rotations_back=get_rotations_back, **kargs)
```

**File:** src/models/modeltype/motionclip.py (L61-83)
```python
    def compute_loss(self, batch):

        # compute all losses other than clip
        mixed_loss = 0.
        losses = {}
        for ltype, lam in self.lambdas.items():
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            mixed_loss += loss * lam
            losses[ltype] = loss.item()

        # compute clip losses
        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)

        # mix and add clip losses
        mixed_loss_with_clip = mixed_loss + mixed_clip_loss  # this is the ultimate loss to optimize, combining ALL losses
        losses.update(clip_losses)
        losses["mixed_without_clip"] = mixed_loss.item()
        losses["mixed_clip_only"] = mixed_clip_loss if isinstance(mixed_clip_loss, float) else mixed_clip_loss.item()
        losses["mixed_with_clip"] = mixed_loss_with_clip if isinstance(mixed_loss_with_clip,
                                                                       float) else mixed_loss_with_clip.item()

        return mixed_loss_with_clip, losses
```

**File:** src/models/modeltype/motionclip.py (L85-162)
```python
    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}

        if self.clip_training:
            for d in self.clip_training.split('_'):
                if d == 'image':
                    features = self.clip_model.encode_image(
                        batch['clip_images']).float()  # preprocess is done in dataloader
                elif d == 'text':
                    texts = clip.tokenize(batch['clip_text']).to(self.device)
                    features = self.clip_model.encode_text(texts).float()

                # normalized features
                features_norm = features / features.norm(dim=-1, keepdim=True)
                seq_motion_features_norm = batch["z"] / batch["z"].norm(dim=-1, keepdim=True)
                logit_scale = self.clip_model.logit_scale.exp()
                logits_per_motion = logit_scale * seq_motion_features_norm @ features_norm.t()
                logits_per_d = logits_per_motion.t()

                batch_size = batch['x'].shape[0]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)

                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                mixed_clip_loss += clip_mixed_loss
        else:
            for d in self.clip_lambdas.keys():
                if len(self.clip_lambdas[d].keys()) == 0:
                    continue
                with torch.no_grad():
                    if d == 'image':
                        features = self.clip_model.encode_image(
                            batch['clip_images']).float()  # preprocess is done in dataloader
                    elif d == 'text':
                        texts = clip.tokenize(batch['clip_text']).to(self.device)
                        features = self.clip_model.encode_text(texts).float()
                    else:
                        raise ValueError(f'Invalid clip domain [{d}]')

                # normalized features
                features_norm = features / features.norm(dim=-1, keepdim=True)
                seq_motion_features_norm = batch["z"] / batch["z"].norm(dim=-1, keepdim=True)

                if 'ce' in self.clip_lambdas[d].keys():
                    logit_scale = self.clip_model.logit_scale.exp()
                    logits_per_motion = logit_scale * seq_motion_features_norm @ features_norm.t()
                    logits_per_d = logits_per_motion.t()

                    batch_size = batch['x'].shape[0]
                    ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)

                    ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                    ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                    clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                    clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                    clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                    clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                    mixed_clip_loss += clip_mixed_loss * self.clip_lambdas[d]['ce']

                if 'mse' in self.clip_lambdas[d].keys():
                    mse_clip_loss = loss_mse(features, batch["z"])
                    clip_losses[f'{d}_mse'] = mse_clip_loss.item()
                    mixed_clip_loss += mse_clip_loss * self.clip_lambdas[d]['mse']

                if 'cosine' in self.clip_lambdas[d].keys():
                    cos = cosine_sim(features_norm, seq_motion_features_norm)
                    cosine_loss = (1 - cos).mean()
                    clip_losses[f'{d}_cosine'] = cosine_loss.item()
                    mixed_clip_loss += cosine_loss * self.clip_lambdas[d]['cosine']

        return mixed_clip_loss, clip_losses
```

**File:** src/models/get_model.py (L8-12)
```python
def get_model(parameters, clip_model):
    encoder = Encoder_TRANSFORMER(**parameters)
    decoder = Decoder_TRANSFORMER(**parameters)
    parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
    return MOTIONCLIP(encoder, decoder, clip_model=clip_model, **parameters).to(parameters["device"])
```

**File:** src/parser/dataset.py (L4-6)
```python
def add_dataset_options(parser):
    group = parser.add_argument_group('Dataset options')
    group.add_argument("--dataset", required=True, help="Dataset to load", default='amass')
```

**File:** src/parser/model.py (L5-18)
```python
def add_model_options(parser):
    group = parser.add_argument_group('Model options')
    group.add_argument("--modelname", default='motionclip_transformer_rc_rcxyz_vel', help="Choice of the model, should be like motionclip_transformer_rc_rcxyz_vel")
    group.add_argument("--latent_dim", default=256, type=int, help="dimensionality of the latent space")
    group.add_argument("--lambda_rc", default=1.0, type=float, help="weight of the rc divergence loss")
    group.add_argument("--lambda_rcxyz", default=1.0, type=float, help="weight of the rc divergence loss")
    group.add_argument("--lambda_vel", default=1.0, type=float, help="weight of the vel divergence loss")
    group.add_argument("--lambda_velxyz", default=1.0, type=float, help="weight of the vel divergence loss")

    group.add_argument("--jointstype", default="vertices", choices=JOINTSTYPES, help="Jointstype for training with xyz")

    group.add_argument('--vertstrans', dest='vertstrans', action='store_true', help="Training with vertex translations in the SMPL mesh")
    group.add_argument('--no-vertstrans', dest='vertstrans', action='store_false', help="Training without vertex translations in the SMPL mesh")
    group.set_defaults(vertstrans=False)
```

**File:** src/utils/get_model_and_data.py (L5-28)
```python
def get_model_and_data(parameters, split="train"):

    # clip_model, preprocess = clip.load("ViT-B/32", device=device)  # Must set jit=False for training
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=parameters['device'], jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

    for domain in parameters.get('clip_training', '').split('_'):
        clip_num_layers = parameters.get('clip_layers', 12)
        if domain == 'text':
            clip_model.initialize_parameters()
            clip_model.transformer.resblocks = clip_model.transformer.resblocks[:clip_num_layers]
        if domain == 'image':
            clip_model.initialize_parameters()
            clip_model.visual.transformer = clip_model.transformer.resblocks[:clip_num_layers]

    # NO Clip Training ,Freeze CLIP weights
    if parameters.get('clip_training', '') == '':
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

    datasets = get_datasets(parameters, clip_preprocess, split)
    model = get_gen_model(parameters, clip_model)
    return model, datasets
```

**File:** src/models/architectures/transformer.py (L69-69)
```python
        self.input_feats = self.njoints*self.nfeats
```

**File:** src/models/architectures/transformer.py (L85-91)
```python
    def forward(self, batch):
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        # embedding of the skeleton
        x = self.skelEmbedding(x)
```

**File:** src/models/tools/losses.py (L7-17)
```python
def compute_rc_loss(model, batch, use_txt_output=False):
    x = batch["x"]
    output = batch["output"]
    mask = batch["mask"]
    if use_txt_output:
        output = batch["txt_output"]
    gtmasked = x.permute(0, 3, 1, 2)[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]

    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss
```

**File:** src/models/tools/losses.py (L20-30)
```python
def compute_rcxyz_loss(model, batch, use_txt_output=False):
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    mask = batch["mask"]
    if use_txt_output:
        output = batch["txt_output_xyz"]
    gtmasked = x.permute(0, 3, 1, 2)[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]

    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss
```

**File:** src/models/tools/losses.py (L33-47)
```python
def compute_vel_loss(model, batch, use_txt_output=False):
    x = batch["x"]
    output = batch["output"]
    if use_txt_output:
        output = batch["txt_output"]
    gtvel = (x[..., 1:] - x[..., :-1])
    outputvel = (output[..., 1:] - output[..., :-1])

    mask = batch["mask"][..., 1:]

    gtvelmasked = gtvel.permute(0, 3, 1, 2)[mask]
    outvelmasked = outputvel.permute(0, 3, 1, 2)[mask]

    loss = F.mse_loss(gtvelmasked, outvelmasked, reduction='mean')
    return loss
```

**File:** src/models/tools/losses.py (L50-64)
```python
def compute_velxyz_loss(model, batch, use_txt_output=False):
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    if use_txt_output:
        output = batch["txt_output_xyz"]
    gtvel = (x[..., 1:] - x[..., :-1])
    outputvel = (output[..., 1:] - output[..., :-1])

    mask = batch["mask"][..., 1:]

    gtvelmasked = gtvel.permute(0, 3, 1, 2)[mask]
    outvelmasked = outputvel.permute(0, 3, 1, 2)[mask]

    loss = F.mse_loss(gtvelmasked, outvelmasked, reduction='mean')
    return loss
```

**File:** src/train/trainer.py (L16-61)
```python
def train_or_test(model, optimizer, iterator, device, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    # loss of the epoch
    dict_loss = {}
    with grad_env():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):

            # Put everything in device
            # Added if is_tensor as 'clip_text' in batch is a list of strings, not a tensor!
            batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}

            if mode == "train":
                # update the gradients to zero
                optimizer.zero_grad()

            # forward pass
            batch = model(batch)

            mixed_loss, losses = model.compute_loss(batch)

            if i == 0:
                dict_loss = deepcopy(losses)
            else:
                for key in dict_loss.keys():
                    dict_loss[key] += losses[key]

            if mode == "train":
                # backward pass
                mixed_loss.backward()
                # update the weights
                if model.clip_training:
                    convert_models_to_fp32(model.clip_model)
                    optimizer.step()
                    clip.model.convert_weights(model.clip_model)
                else:
                    optimizer.step()

    return dict_loss
```

**File:** src/utils/tensors.py (L23-62)
```python
def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    if len(notnone_batches) == 0:
        out_batch = {"x": [], "y": [],
                     "mask": [], "lengths": [],
                     "clip_image": [], "clip_text": [],
                     "clip_path": [], "clip_images_emb": []
                     }
        return out_batch
    databatch = [b['inp'] for b in notnone_batches]
    labelbatch = [b['target'] for b in notnone_batches]
    lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor)


    out_batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
             # "y_action_names": actionlabelbatchTensor}
    if 'clip_image' in notnone_batches[0]:
        clip_image_batch = [torch.as_tensor(b['clip_image']) for b in notnone_batches]
        out_batch.update({'clip_images': collate_tensors(clip_image_batch)})

    if 'clip_text' in notnone_batches[0]:
        textbatch = [b['clip_text'] for b in notnone_batches]
        out_batch.update({'clip_text': textbatch})

    if 'clip_path' in notnone_batches[0]:
        textbatch = [b['clip_path'] for b in notnone_batches]
        out_batch.update({'clip_path': textbatch})

    if 'all_categories' in notnone_batches[0]:
        textbatch = [b['all_categories'] for b in notnone_batches]
        out_batch.update({'all_categories': textbatch})

    return out_batch
```

**File:** src/models/smpl.py (L70-71)
```python
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
```

**File:** src/datasets/amass_parser.py (L68-78)
```python
amass_test_split = ['Transitions_mocap', 'SSM_synced']
amass_vald_split = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh']
amass_train_split = ['BioMotionLab_NTroje', 'Eyes_Japan_Dataset', 'TotalCapture', 'KIT', 'ACCAD', 'CMU', 'MPI_Limits',
                     'TCD_handMocap', 'EKUT']
# Source - https://github.com/nghorbani/amass/blob/08ca36ce9b37969f72d7251eb61564a7fd421e15/src/amass/data/prepare_data.py#L235
amass_splits = {
    'test': amass_test_split,
    'vald': amass_vald_split,
    'train': amass_train_split
}
assert len(amass_splits['train'] + amass_splits['test'] + amass_splits['vald']) == len(all_sequences) == 15
```
