import os
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import torch
import csv
import numpy as np
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.visualize.visualize import (
    viz_motion2text, 
    get_gpu_device,
    get_motion_text_mapping,
    retrieve_motions,
    generate_by_video,
    generate_by_video_g1
)
from src.utils.misc import load_model_wo_clip
import imageio

import src.utils.fixseed  # noqa

plt.switch_backend('agg')


def viz_dataset_motions(model, datasets, motion_csv, epoch, params, folder, image_pathes=None):
    """
    Visualize motions from the dataset based on a CSV file.
    
    Args:
        model: The MotionCLIP model
        datasets: Dictionary of datasets (train/val/test)
        motion_csv: List of dictionaries from CSV file with 'motion_text' field
        epoch: Current epoch number
        params: Parameters dictionary
        folder: Output folder for saving visualizations
        image_pathes: Optional list of image paths to display alongside motions
    """
    # visualize with joints3D
    model.outputxyz = True
    
    print(f"Dataset Motion Visualization (epoch {epoch})")
    figname = params["figname"].format(epoch)
    
    motion_collection = get_motion_text_mapping(datasets)
    device = params['device']
    
    print("\n" + "="*80)
    print("DATASET MOTION VISUALIZATION")
    print("="*80 + "\n")
    
    all_motions = []
    all_labels = []
    all_lengths = []
    not_found = []
    
    for idx, line in enumerate(motion_csv):
        motion_text = line['motion_text']
        
        print(f"[{idx+1}/{len(motion_csv)}] Processing: '{motion_text}'")
        
        # Retrieve motion from dataset
        if motion_text not in motion_collection:
            print(f"  ⚠ Motion '{motion_text}' not found in dataset, skipping...")
            not_found.append(motion_text)
            print()
            continue
        
        motions = retrieve_motions(datasets, motion_collection, [motion_text], device)
        
        # Get actual motion length (full length, not clipped)
        # Motion shape is typically (batch, features, frames) or (batch, frames, features)
        # We need to determine the actual sequence length
        if len(motions.shape) == 3:
            # Shape: (batch, seq_len, features) or (batch, features, seq_len)
            # Assuming (batch, seq_len, features) based on typical motion representation
            actual_length = motions.shape[1]
        elif len(motions.shape) == 2:
            # Shape: (batch, features) - single frame or flattened
            actual_length = 1
        else:
            # Fallback to params if shape is unexpected
            actual_length = params['num_frames']
        
        # Store motion for visualization
        all_motions.append(motions)
        all_labels.append(motion_text)
        all_lengths.append(torch.tensor([actual_length], device=device))
        
        print(f"  ✓ Motion retrieved successfully (shape: {motions.shape}, length: {actual_length} frames)")
        print()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total motions requested: {len(motion_csv)}")
    print(f"Successfully retrieved: {len(all_motions)}")
    print(f"Not found: {len(not_found)}")
    if not_found:
        print(f"Missing motions: {', '.join(not_found)}")
    print("="*80 + "\n")
    
    # Generate visualization if we have motions
    if len(all_motions) == 0:
        print("No motions to visualize. Exiting.")
        return
    
    print("Generating motion visualizations...")
    
    # Concatenate all motions
    all_motions_tensor = torch.cat(all_motions, dim=0)
    all_lengths_tensor = torch.cat(all_lengths, dim=0)
    
    # Convert motions to xyz coordinates
    model.eval()
    with torch.no_grad():
        mask = model.lengths_to_mask(all_lengths_tensor)
        output_xyz = model.rot2xyz(all_motions_tensor, mask)
    
    # Prepare visualization data structure
    # Layout: each row is a different motion
    h = len(all_motions)
    w = 1
    
    visualization = {
        'output_xyz': output_xyz.reshape(h, w, *output_xyz.shape[1:]),
        'lengths': all_lengths_tensor.reshape(h, w),
        'y': np.array(all_labels).reshape(h, w)
    }
    
    # Setup paths
    f_name = os.path.basename(params['input_file']).replace('.csv', '')
    finalpath = os.path.join(folder, f'dataset_vis_{f_name}_' + figname + ".gif")
    tmp_path = os.path.join(folder, f"dataset_vis_subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)
    
    # Generate videos
    print("Rendering video frames...")
    use_g1 = params.get('use_g1', False)
    if use_g1:
        frames = generate_by_video_g1({}, {}, visualization,
                                      lambda x: str(x), params, w, h, tmp_path, 
                                      image_pathes=image_pathes, mode='dataset')
    else:
        frames = generate_by_video({}, {}, visualization,
                                   lambda x: str(x), params, w, h, tmp_path, 
                                   image_pathes=image_pathes, mode='dataset')
    
    print(f"Writing video [{finalpath}]")
    imageio.mimsave(finalpath, frames, fps=params["fps"])
    print(f"✓ Visualization saved to: {finalpath}")
    
    # Save motion list to file
    list_path = os.path.join(folder, f'dataset_vis_{f_name}_motions.txt')
    with open(list_path, 'w') as f:
        f.write("DATASET MOTION VISUALIZATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total motions visualized: {len(all_motions)}\n\n")
        f.write("Motion list:\n")
        for i, label in enumerate(all_labels, 1):
            f.write(f"{i}. {label}\n")
        if not_found:
            f.write("\nMotions not found in dataset:\n")
            for motion in not_found:
                f.write(f"  - {motion}\n")
    
    print(f"✓ Motion list saved to: {list_path}")


def main():
    # parse options
    parameters, folder, checkpointname, epoch = parser()
    gpu_device = get_gpu_device()
    parameters["device"] = f"cuda:{gpu_device}"
    model, datasets = get_model_and_data(parameters, split='all')

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)

    assert os.path.isfile(parameters['input_file']), f"Input file not found: {parameters['input_file']}"
    
    # Read CSV file
    with open(parameters['input_file'], 'r') as fr:
        motion_csv = list(csv.DictReader(fr))
    
    print(f"Loaded {len(motion_csv)} motion entries from {parameters['input_file']}")
    
    # Check if image paths are provided (optional)
    image_pathes = None
    if 'image_path' in motion_csv[0]:
        image_pathes = [line.get('image_path', '') for line in motion_csv]
        image_pathes = [path for path in image_pathes if path and os.path.isfile(path)]
        if image_pathes:
            print(f"Found {len(image_pathes)} valid image paths")
        else:
            image_pathes = None
    
    # Visualize dataset motions
    viz_dataset_motions(model, datasets, motion_csv, epoch, parameters, folder, image_pathes)


if __name__ == '__main__':
    main()
