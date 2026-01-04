# Dataset Motion Visualization

This script visualizes motions from the AMASS or G1AMASS dataset based on a CSV file containing motion text labels.

## Usage

```bash
python src/test/amass/vis_dataset.py \
    <checkpoint_path> \
    --input_file <csv_file>
```

```bash
python src/test/amass/vis_dataset.py \
    exps/g1-model3/checkpoint_0100.pth.tar \
    --input_file assets/dataset_vis.csv
```

## CSV File Format

The CSV file should contain a column named `motion_text` with the motion labels you want to visualize.

### Basic Format (required)
```csv
motion_text
walk
sit down and sitting
Jumping
run forward
kick
```

### Extended Format (optional - with images)
```csv
motion_text,image_path
walk,/path/to/walk_image.png
sit down,/path/to/sit_image.jpg
run forward,/path/to/run_image.png
```

## Examples

### Example 1: Basic visualization (SMPL AMASS)
```bash
python src/test/amass/vis_dataset.py \
    exps/smpl-model/checkpoint_0100.pth.tar \
    --input_file assets/paper_motion2text.csv
```

### Example 2: G1 robot visualization
```bash
python src/test/amass/vis_dataset.py \
    exps/g1-model3/checkpoint_0100.pth.tar \
    --input_file assets/g1_motions.csv
```

## Output

The script generates:

1. **Video file**: `dataset_vis_<filename>_<epoch>.gif`
   - Animated visualization of all requested motions
   - Each motion is displayed with its text label
   - For G1: Shows 30 body positions with kinematic chains
   - For SMPL: Shows full humanoid skeleton

2. **Motion list**: `dataset_vis_<filename>_motions.txt`
   - Complete list of visualized motions
   - List of motions not found in dataset (if any)

## Features

- ✅ Supports both AMASS (SMPL) and G1AMASS datasets
- ✅ Automatically detects dataset type based on model checkpoint
- ✅ Retrieves actual motion sequences from the dataset
- ✅ Optional image display alongside motions
- ✅ Clear error reporting for missing motions
- ✅ Summary statistics and logging

## Motion Label Requirements

Motion labels must match the text annotations in the dataset. To find available motion labels:

1. Check the generated `_text_labels.txt` file in your data directory
2. Or use common motion descriptions like:
   - "walk", "run", "jump", "sit", "stand"
   - "kick", "punch", "throw"
   - "sit down", "stand up", "kneel down"
   - etc.

## Comparison with motion2text.py

- **motion2text.py**: Retrieves motions and predicts text labels using CLIP
- **vis_dataset.py**: Directly visualizes motions from dataset using provided text labels

Both scripts share similar CSV format and visualization pipeline.
