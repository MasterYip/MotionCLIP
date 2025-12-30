"""
Minimal example of loading MotionCLIP model.

This script demonstrates how to load the MotionCLIP model using the clean
Hydra-based configuration approach, without the messy argparse setup.
"""

import sys
import os
sys.path.append('.')

import torch
import clip
from src.utils.get_model_and_data import get_motion_clip


def example_1_load_with_defaults():
    """Load model with default configuration (no checkpoint)."""
    print("="*80)
    print("Example 1: Load model with default configuration")
    print("="*80)
    
    model, cfg = get_motion_clip(device='cuda')
    
    print(f"Model loaded successfully!")
    print(f"Model type: {cfg.model.modeltype}")
    print(f"Architecture: {cfg.model.archiname}")
    print(f"Latent dim: {cfg.model.latent_dim}")
    print(f"Num layers: {cfg.model.num_layers}")
    print(f"Pose representation: {cfg.model.pose_rep}")
    print()


def example_2_load_with_checkpoint():
    """Load pre-trained model from checkpoint."""
    print("="*80)
    print("Example 2: Load pre-trained model from checkpoint")
    print("="*80)
    
    checkpoint_path = './exps/paper-model/checkpoint_0100.pth.tar'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please download the model first or provide a valid checkpoint path.")
        return
    
    model, cfg = get_motion_clip(
        checkpoint_path=checkpoint_path,
        device='cuda'
    )
    
    print(f"Pre-trained model loaded successfully!")
    print(f"Checkpoint: {checkpoint_path}")
    print()


def example_3_generate_motion_from_text():
    """Generate motion from text using the loaded model."""
    print("="*80)
    print("Example 3: Generate motion from text description")
    print("="*80)
    
    checkpoint_path = './exps/paper-model/checkpoint_0100.pth.tar'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    # Load model
    model, cfg = get_motion_clip(
        checkpoint_path=checkpoint_path,
        device='cuda'
    )
    
    # Prepare text input
    text_descriptions = ["a person walks forward", "a person jumps high", "a person sits down"]
    print(f"Text inputs: {text_descriptions}")
    
    # Encode text with CLIP
    device = 'cuda'
    text_tokens = clip.tokenize(text_descriptions).to(device)
    clip_features = model.clip_model.encode_text(text_tokens).float().unsqueeze(0)
    
    # Generate motions
    num_frames = cfg.model.num_frames
    durations = torch.ones((len(text_descriptions), 1), dtype=int) * num_frames
    
    model.eval()
    with torch.no_grad():
        output_dict = model.generate(
            clip_features,
            durations,
            is_amass=True,
            is_clip_features=True
        )
    
    # Get generated motions in XYZ format
    output_xyz = output_dict['output_xyz']
    print(f"Generated motion shape: {output_xyz.shape}")
    print(f"  - Batch size: {output_xyz.shape[0]}")
    print(f"  - Num joints: {output_xyz.shape[1]}")
    print(f"  - Coordinates: {output_xyz.shape[2]} (x, y, z)")
    print(f"  - Num frames: {output_xyz.shape[3]}")
    print()


def example_4_encode_motion():
    """Encode motion sequence to latent space."""
    print("="*80)
    print("Example 4: Encode motion sequence to latent space")
    print("="*80)
    
    checkpoint_path = './exps/paper-model/checkpoint_0100.pth.tar'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    # Load model
    model, cfg = get_motion_clip(
        checkpoint_path=checkpoint_path,
        device='cuda'
    )
    
    device = 'cuda'
    
    # Create dummy motion input (normally you'd load from dataset)
    # Shape: [batch_size, num_joints, num_features, num_frames]
    batch_size = 2
    num_joints = 24 if cfg.model.jointstype == 'vertices' else 18
    num_features = 6 if cfg.model.pose_rep == 'rot6d' else 9  # rot6d or rotmat
    num_frames = cfg.model.num_frames
    
    dummy_motion = torch.randn(batch_size, num_joints, num_features, num_frames).to(device)
    
    # Encode to latent space
    model.eval()
    with torch.no_grad():
        mask = model.lengths_to_mask(torch.ones(batch_size, dtype=int, device=device) * num_frames)
        encoded = model.encoder({
            'x': dummy_motion,
            'y': torch.zeros(batch_size, dtype=int, device=device),
            'mask': mask
        })
        latent_features = encoded['mu']
    
    print(f"Encoded latent features shape: {latent_features.shape}")
    print(f"  - Batch size: {latent_features.shape[0]}")
    print(f"  - Latent dimension: {latent_features.shape[1]}")
    print()


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("MotionCLIP Minimal Loading Examples")
    print("="*80 + "\n")
    
    # Example 1: Load with defaults
    example_1_load_with_defaults()
    
    # Example 2: Load with checkpoint
    example_2_load_with_checkpoint()
    
    # Example 3: Generate motion from text
    example_3_generate_motion_from_text()
    
    # Example 4: Encode motion
    example_4_encode_motion()
    
    print("="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == '__main__':
    main()
