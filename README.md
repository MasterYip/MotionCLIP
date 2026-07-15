# MoDyeEnc

Motion encoder for the [PegasusMoDye](../README.md) project.
Maps motion sequences and natural-language descriptions into a shared **512-dim CLIP latent space**, providing semantic conditioning for the PDPlanner diffusion policy.

Adapted from [MotionCLIP](https://guytevet.github.io/motionclip-page/) and modified for the **Unitree G1** robot — replacing SMPL-based kinematics with G1 retargeted body positions and joint angles.

---

## Role in PegasusMoDye

```
Text prompt ("walk forward")
        │
        ▼
 ┌─────────────┐
 │  MoDyeEnc   │  Encodes text / motion clips → 512-dim latent z
 └──────┬──────┘
        │  cond z
        ▼
 ┌────────────────────┐
 │  PDPlanner         │  Co-diffuses state + action trajectories conditioned on z
 └────────────────────┘
        │  joint PD targets (29 DoF)
        ▼
     Unitree G1
```

At inference time, a single CLIP text encoder call produces `z`; the same latent can also be extracted from a reference motion clip for imitation-style conditioning.

---

## Installation

Requires Python 3.10.

```bash
pip install -e .
```

> **China mirror users:** `pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/`

### SMPL body models (needed for data pre-processing)

```bash
bash prepare/download_smpl_files.sh          # SMPL neutral model
# Download SMPL+H from https://mano.is.tue.mpg.de/ → place in ./models/smplh
```

These model files are also available from the project's Hugging Face repository (see [Pretrained Checkpoint](#pretrained-checkpoint)).

---

## Dataset Construction for Unitree G1

<!-- TODO: add G1 retargeted dataset url -->

MoDyeEnc trains on [G1-retargeted AMASS](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1) motion-capture data.
The raw AMASS sequences are retargeted onto the G1 skeleton (producing `_jpos.npz` files), then parsed into a unified `.pt` database.

### 1. Obtain G1-retargeted AMASS data

Retargeted files follow the naming `<sequence>_jpos.npz` and contain:

| Field | Shape | Description |
|---|---|---|
| `dof_positions` | `(T, 29)` | Joint angles (rad) |
| `dof_velocities` | `(T, 29)` | Joint velocities (rad/s) |
| `body_positions` | `(T, 20, 3)` | Body link positions in world frame |
| `body_rotations` | `(T, 20, 3)` | Body orientations (axis-angle) |
| `body_linear_velocities` | `(T, 20, 3)` | Body linear velocities |
| `body_angular_velocities` | `(T, 20, 3)` | Body angular velocities |

Place the retargeted dataset under `./data/g1_retargeted_amass/`.

The parser also requires two external resources:

- **BABEL labels** — download from [babel.is.tue.mpg.de](https://babel.is.tue.mpg.de/) → place in `./data/babel_v1.0_release/`
- **Rendered AMASS images** — download from [Google Drive](https://drive.google.com/file/d/1F8VLY4AC2XPaV3DqKZefQJNWn4KY2z_c/view?usp=sharing) → extract to `./data/render/`

### 2. Parse into training database

Run from the `MoDyeEnc/` directory:

```bash
python src/datasets/g1_amass_parser.py \
    --input_dir  ./data/g1_retargeted_amass \
    --output_dir ./data/g1_amass_db \
    --dataset_name amass \
    --babel_dir ./data/babel_v1.0_release \
    --target_fps 30
```

This produces:
```
data/g1_amass_db/
  amass_30fps_train.pt
  amass_30fps_vald.pt
  amass_30fps_test.pt
```

The train / validation / test split follows the standard AMASS partition:

| Split | Sequences |
|---|---|
| train | BioMotionLab_NTroje, Eyes_Japan_Dataset, TotalCapture, KIT, ACCAD, CMU, MPI_Limits, TCD_handMocap, EKUT |
| vald | HumanEva, MPI_HDM05, SFU, MPI_mosh |
| test | Transitions_mocap, SSM_synced |

**Quick sanity check:**

```bash
python -c "
from src.datasets.amass import AMASS
ds = AMASS(datapath='./data/g1_amass_db/amass_30fps_db.pt',
           split='train', use_g1=True, num_frames=60, pose_rep='xyz')
print('Dataset size:', len(ds))
print('Sample shape:', ds[0]['inp'].shape)   # expected: [20, 3, 60]
"
```

---

## Training

All commands are run from inside the `MoDyeEnc/` directory.

### G1 model — XYZ representation (recommended)

This is the configuration used to produce the released `g1-model-xyz-clip` checkpoint:

```bash
python -m src.train.train --modelname motionclip_transformer_rc_vel \
--clip_text_losses cosine --pose_rep xyz \
--clip_lambda_cosine 5.0 \
--clip_training text \
--lambda_vel 100 --lambda_rc 100 --lambda_rcxyz 100 \
--jointstype vertices --batch_size 20 --num_frames 60 --num_layers 8 \
--lr 0.0001 --glob --translation --no-vertstrans --latent_dim 512 --num_epochs 100 --snapshot 10 \
--device 0 \
--dataset g1_amass \
--datapath ./data/g1_amass_db/amass_30fps_db.pt \
--folder ./exps/g1-model-xyz-clip \
--use_g1
```

Key flags specific to G1:

| Flag | Value | Notes |
|---|---|---|
| `--dataset g1_amass` | — | Loads G1AMASS dataset class |
| `--use_g1` | — | Replaces SMPL FK with G1 pass-through |
| `--pose_rep xyz` | — | Uses `body_positions` directly; no rotation conversion needed |
| `--latent_dim 512` | — | Must match CLIP ViT-B/32 embedding dimension |

### Convenience script

```bash
#!/bin/bash
DATA_PATH="./data/g1_amass_db/amass_30fps_db.pt"
OUTPUT_DIR="./exps/g1-model-xyz-clip-$(date +%Y%m%d-%H%M%S)"

python -m src.train.train \
  --clip_text_losses cosine --clip_image_losses cosine \
  --pose_rep xyz \
  --lambda_vel 100 --lambda_rc 100 --lambda_rcxyz 100 \
  --jointstype vertices --batch_size 20 --num_frames 60 --num_layers 8 \
  --lr 0.0001 --glob --translation --no-vertstrans \
  --latent_dim 512 --num_epochs 100 --snapshot 10 \
  --device 0 \
  --dataset g1_amass --datapath $DATA_PATH \
  --folder $OUTPUT_DIR --use_g1

echo "Done → $OUTPUT_DIR"
```

Checkpoints are saved every 10 epochs as `checkpoint_XXXX.pth.tar` inside `--folder`.

---

## Pretrained Checkpoint

A pretrained checkpoint is available from the project's Hugging Face repository:

```bash
pip install huggingface_hub

# Download MoDyeEnc checkpoint only
python scripts/hf_download.py --filter checkpoints/modyeenc
```

The checkpoint is stored at `checkpoints/modyeenc/` in the HF repo
(local mirror: `MoDyeEnc/exps/g1-model-xyz-clip/`).

SMPL body model files required for data preprocessing are also available:

```bash
python scripts/hf_download.py --filter assets/smpl_models
```

---

## Demos

### Text-to-latent (inference)

Encode a text prompt and retrieve the closest motion from the training set:

```bash
python -m src.visualize.text2motion \
  ./exps/g1-model-xyz-clip/checkpoint_0100.pth.tar \
  --input_file assets/paper_texts.txt
```

Create `assets/paper_texts.txt` with one prompt per line, e.g.:
```
walk forward
wave left hand
jump in place
turn around
```

### Motion-to-text retrieval

Given a reference motion, retrieve the most likely text descriptions via CLIP:

```bash
python -m src.visualize.motion2text \
  ./exps/g1-model-xyz-clip/checkpoint_0100.pth.tar \
  --input_file assets/paper_motion2text.csv
```

### Latent interpolation

Smoothly interpolate between two motion latents:

```bash
python -m src.visualize.motion_interpolation \
  ./exps/g1-model-xyz-clip/checkpoint_0100.pth.tar \
  --input_file assets/paper_interps.csv
```

---

## Architecture Notes

MoDyeEnc is a **transformer autoencoder** aligned to CLIP space via contrastive losses:

- **Encoder**: causal transformer over `[njoints=20, nfeats=3, nframes=60]` G1 body-position sequences → 512-dim latent `z`
- **Decoder**: mirrors encoder; reconstructed motions supervised by reconstruction + velocity + xyz losses
- **CLIP alignment**: cosine similarity loss between `z` and CLIP text/image embeddings (frozen ViT-B/32)

G1-specific changes versus the original MotionCLIP:

| Component | Original | G1 modification |
|---|---|---|
| Data | SMPL parameters (`thetas`) | G1 retargeted `body_positions` (`_jpos.npz`) |
| FK | `Rotation2xyz` (SMPL model) | `G1ToXyz` pass-through for `pose_rep=xyz` |
| Joints | 23 SMPL joints | 20 G1 body links |
| Input features | 138 (23 × rot6d) | 60 (20 × xyz) |
| Body model files | SMPL `.pkl` required | Not needed at training time |

---

## Acknowledgement

This code is adapted from [MotionCLIP](https://github.com/GuyTevet/motion-clip) by Tevet et al. (ECCV 2022).
The transformer architecture originates from [ACTOR](https://github.com/Mathux/ACTOR).

```bibtex
@article{tevet2022motionclip,
  title   = {MotionCLIP: Exposing Human Motion Generation to CLIP Space},
  author  = {Tevet, Guy and Gordon, Brian and Hertz, Amir and Bermano, Amit H and Cohen-Or, Daniel},
  journal = {arXiv preprint arXiv:2203.08063},
  year    = {2022}
}
```

## License

Distributed under the [MIT License](LICENSE), following the original MotionCLIP.
Dependencies (CLIP, SMPL, PyTorch3D) carry their own respective licenses.
