"""
Motion-to-Text script using clean configuration loading.

This script demonstrates motion-to-text conversion without the messy argparse setup.
It uses the new get_motion_clip function for clean model loading.
"""

import sys
import os
sys.path.append('.')

import torch
import csv
import argparse
from src.utils.get_model_and_data import get_motion_clip
from src.datasets.get_dataset import get_datasets
from src.visualize.visualize import viz_motion2text
import clip


def main():
    """
    Run motion-to-text conversion.
    
    Usage:
        python -m src.scripts.motion2text <checkpoint_path> --input_file <csv_file> [--device cuda]
    
    Example:
        python -m src.scripts.motion2text ./exps/paper-model/checkpoint_0100.pth.tar \\
            --input_file assets/paper_motion2text.csv
    """
    parser = argparse.ArgumentParser(description='Motion-to-Text conversion')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--input_file', type=str, default='assets/paper_motion2text.csv',
                        help='Path to input CSV file with motion labels')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional path to config YAML file')
    
    args = parser.parse_args()
    
    # Load model with new clean method
    print("Loading MotionCLIP model...")
    model, cfg = get_motion_clip(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    print("Model loaded successfully!")
    
    # Get folder and epoch from checkpoint path
    folder = os.path.dirname(args.checkpoint)
    checkpoint_name = os.path.basename(args.checkpoint)
    epoch = int(checkpoint_name.split("_")[-1].split('.')[0])
    
    # Load datasets (needed for motion retrieval)
    print("Loading datasets for motion retrieval...")
    
    # Create parameters dict for dataset loading (backward compatibility)
    # We need to convert the clean config back to the old format for dataset loading
    from omegaconf import OmegaConf
    
    # If custom config provided, load it, otherwise use checkpoint folder's opt.yaml
    if args.config:
        dataset_cfg_path = args.config
    else:
        dataset_cfg_path = os.path.join(folder, 'opt.yaml')
    
    if os.path.exists(dataset_cfg_path):
        dataset_params = OmegaConf.to_container(OmegaConf.load(dataset_cfg_path), resolve=True)
    else:
        # Use minimal defaults if no config file
        dataset_params = {
            'datapath': './data/amass_db/amass_30fps_db.pt',
            'dataset': 'amass',
            'num_frames': cfg.model.num_frames,
            'pose_rep': cfg.model.pose_rep,
            'glob': cfg.model.glob,
            'glob_rot': cfg.model.glob_rot,
            'translation': cfg.model.translation,
            'jointstype': cfg.model.jointstype,
            'vertstrans': cfg.model.vertstrans,
        }
    
    dataset_params['device'] = args.device
    
    # Load CLIP for preprocessing
    _, clip_preprocess = clip.load("ViT-B/32", device=args.device, jit=False)
    
    datasets = get_datasets(dataset_params, clip_preprocess, split='all')
    print(f"Datasets loaded: {list(datasets.keys())}")
    
    # Read motion CSV
    assert os.path.isfile(args.input_file), f"Input file not found: {args.input_file}"
    with open(args.input_file, 'r') as fr:
        motion_csv = list(csv.DictReader(fr))
    
    print(f"Loaded {len(motion_csv)} motions from {args.input_file}")
    
    # Create parameters dict for visualization
    params = {
        'device': args.device,
        'num_frames': cfg.model.num_frames,
        'pose_rep': cfg.model.pose_rep,
        'fps': 30,
        'figname': 'motion2text_{:03d}',  # Format string for epoch
        'input_file': args.input_file,
        'appearance_mode': 'motionclip',
        'vertstrans': cfg.model.vertstrans,
        'use_g1': cfg.model.get('use_g1', False),  # Add G1 flag for proper visualization
    }
    
    # Run motion-to-text conversion
    print("\nRunning motion-to-text conversion...")
    viz_motion2text(model, datasets, motion_csv, epoch, params, folder=folder)
    
    print("\nMotion-to-text conversion completed!")


if __name__ == '__main__':
    main()
