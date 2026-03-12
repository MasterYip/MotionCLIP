# G1 MotionCLIP Training Guide

## Overview
This guide explains how to train MotionCLIP with G1 retargeted AMASS data.

## Prerequisites

1. **Prepared G1 Dataset**: You need G1 retargeted AMASS data processed with `amass_parser_g1.py`:
   ```bash
   python MoDyeEnc/src/datasets/amass_parser_g1.py \
     --input_dir ./data/g1_retargeted_amass \
     --output_dir ./data/g1_amass_db \
     --dataset_name amass \
     --target_fps 30 \
     --babel_dir ./data/babel_v1.0_release \
     --clip_images_dir ./data/render_g1
   ```

   This creates:
   - `g1_amass_db/amass_30fps_train.pt`
   - `g1_amass_db/amass_30fps_vald.pt`
   - `g1_amass_db/amass_30fps_test.pt`

2. **Dataset Structure**: Each `.pt` file should contain:
   ```python
   {
       'thetas': List[np.ndarray],  # DOF positions [seq_len, num_dofs]
       'joints3d': List[np.ndarray],  # Body positions [seq_len, num_bodies, 3]
       'clip_images': List[np.ndarray],  # Optional CLIP images
       'text_raw_labels': List[np.ndarray],  # BABEL text labels
       'text_proc_labels': List[np.ndarray],
       'action_cat': List[np.ndarray],
       # ... other fields
   }
   ```

## Training Commands

### Basic G1 Training (with XYZ representation - Recommended)

```bash
python -m src.train.train \
  --dataset g1_amass \
  --use_g1 \
  --datapath ./data/g1_amass_db/amass_30fps_db.pt \
  --pose_rep xyz \
  --clip_text_losses cosine \
  --clip_image_losses cosine \
  --lambda_vel 100 \
  --lambda_rc 100 \
  --lambda_rcxyz 100 \
  --jointstype vertices \
  --batch_size 20 \
  --num_frames 60 \
  --num_layers 8 \
  --lr 0.0001 \
  --glob \
  --translation \
  --no-vertstrans \
  --latent_dim 512 \
  --num_epochs 100 \
  --snapshot 10 \
  --device 0 \
  --folder ./exps/g1-motionclip
```

### Key Parameters Explained

- `--dataset g1_amass`: Use G1 AMASS dataset (automatically sets `use_g1=True`)
- `--use_g1`: Enable G1 mode (use G1ToXyz instead of SMPL)
- `--pose_rep xyz`: Use XYZ positions directly (recommended for G1)
  - G1 data already has `body_positions`, so this is most straightforward
  - Alternative: `dof` would require implementing G1 forward kinematics
- `--datapath`: Path to G1 dataset (will append `_train.pt`, `_vald.pt`)
- `--g1_num_bodies 20`: Number of G1 bodies (default: 20, adjust if needed)

### Training with Different Pose Representations

**Option 1: XYZ (Recommended)**
```bash
--pose_rep xyz --translation --glob
```
- Uses `body_positions` directly from G1 data
- Simplest and most reliable for G1

**Option 2: DOF (Not Yet Implemented)**
```bash
--pose_rep dof --use_g1
```
- Would use `dof_positions` directly
- Requires implementing G1 forward kinematics in `G1ToXyz`

## Architecture Adjustments

### Input Dimensions

G1 has different dimensions than SMPL:
- **SMPL**: 23 joints × 6 features (rot6d) = 138 input features
- **G1 (xyz)**: 20 bodies × 3 features (xyz) + translation = varies
- **G1 (dof)**: 29 DOFs × 1 feature = 29 input features

The network automatically adjusts based on dataset output:
```python
# From dataset.update_parameters()
self.njoints, self.nfeats, _ = data['inp'].shape
parameters["njoints"] = self.njoints
parameters["nfeats"] = self.nfeats
```

### Model Configuration

For G1, the transformer encoder will receive:
- Input: `[batch, njoints, nfeats, nframes]`
  - njoints = 20 (number of G1 bodies)
  - nfeats = 3 (x, y, z) or more if translation is included
  - nframes = 60 (sequence length)

## Implementation Details

### What Was Changed

1. **Dataset (`amass.py`, `dataset.py`)**:
   - Added `use_g1` parameter
   - `_load_dof_positions()` method for G1 DOF data
   - Modified `_load()` to skip rotation conversion for G1
   - `_load_rotvec()` handles G1 DOF positions

2. **Model (`motionclip.py`)**:
   - Conditionally uses `G1ToXyz` instead of `Rotation2xyz`
   - Automatically selected based on `use_g1` parameter

3. **G1ToXyz (`g1_to_xyz.py`)**:
   - Replaces SMPL forward kinematics
   - Currently pass-through for xyz data
   - Placeholder for G1 FK implementation

4. **Configuration (`config.py`)**:
   - Added G1 constants: `G1_NUM_DOFS`, `G1_NUM_BODIES`
   - Added `'g1_dof'` to `ROT_CONVENTION_TO_ROT_NUMBER`

5. **Parsers**:
   - `dataset.py`: Added `g1_amass` dataset option
   - `model.py`: Added `--use_g1` and `--g1_num_bodies` flags

### What Stays the Same

- **Training loop**: No changes needed
- **Loss functions**: All work with xyz representations
- **CLIP integration**: Works identically
- **Optimizer and learning**: Same as SMPL training

## Validation

### Quick Test
```bash
# Test dataset loading
python -c "
from src.datasets.amass import AMASS
dataset = AMASS(
    datapath='./data/g1_amass_db/amass_30fps_db.pt',
    split='train',
    use_g1=True,
    num_frames=60,
    pose_rep='xyz'
)
print(f'Dataset size: {len(dataset)}')
sample = dataset[0]
print(f'Input shape: {sample[\"inp\"].shape}')
"
```

### Expected Output
```
Dataset size: XXXX
Input shape: torch.Size([20, 3, 60])  # [njoints=20, nfeats=3, nframes=60]
```

## Troubleshooting

### Error: "This representation is not possible"
- **Cause**: Trying to use a pose representation that requires conversion
- **Solution**: Use `--pose_rep xyz` for G1 data

### Error: "G1 forward kinematics not implemented"
- **Cause**: Using `--pose_rep dof` without implementing FK
- **Solution**: Either use `--pose_rep xyz` or implement G1 FK in `g1_to_xyz.py`

### Error: Dimension mismatch
- **Cause**: G1 body count doesn't match data
- **Solution**: Check actual number of bodies in your G1 data and adjust `g1_num_bodies`

### Poor Training Performance
- **Check**: Input normalization - G1 body positions may have different scales than SMPL
- **Solution**: May need to add normalization in dataset loading

## Next Steps

1. **Implement G1 Forward Kinematics** (if using DOF directly):
   - Add G1 kinematic chain to `g1_to_xyz.py`
   - Convert DOF positions → body positions

2. **Create G1 Visualization**:
   - Current visualization uses SMPL mesh
   - Need G1-specific skeleton/mesh renderer

3. **Fine-tune Hyperparameters**:
   - Learning rate may need adjustment for G1
   - Loss weights might differ from SMPL

4. **Evaluate Motion Quality**:
   - Compare generated motions with ground truth G1 data
   - Check joint limits and physical constraints

## Example Training Script

```bash
#!/bin/bash

# G1 MotionCLIP Training Script

# Set paths
DATA_PATH="./data/g1_amass_db/amass_30fps_db.pt"
BABEL_DIR="./data/babel_v1.0_release"
OUTPUT_DIR="./exps/g1-motionclip-$(date +%Y%m%d-%H%M%S)"

# Training parameters
BATCH_SIZE=20
NUM_FRAMES=60
LATENT_DIM=512
NUM_EPOCHS=100
LR=0.0001

# Run training
python -m src.train.train \
  --dataset g1_amass \
  --use_g1 \
  --datapath $DATA_PATH \
  --pose_rep xyz \
  --clip_text_losses cosine \
  --clip_image_losses cosine \
  --lambda_vel 100 \
  --lambda_rc 100 \
  --lambda_rcxyz 100 \
  --jointstype vertices \
  --batch_size $BATCH_SIZE \
  --num_frames $NUM_FRAMES \
  --num_layers 8 \
  --lr $LR \
  --glob \
  --translation \
  --no-vertstrans \
  --latent_dim $LATENT_DIM \
  --num_epochs $NUM_EPOCHS \
  --snapshot 10 \
  --device 0 \
  --folder $OUTPUT_DIR

echo "Training completed. Model saved to: $OUTPUT_DIR"
```

## References

- Original MotionCLIP paper
- G1 robot specifications
- AMASS dataset documentation
- BABEL labels documentation
