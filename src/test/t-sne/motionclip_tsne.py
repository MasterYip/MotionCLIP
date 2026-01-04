import os
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
from sklearn.manifold import TSNE
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.visualize.visualize import get_motion_text_mapping, get_gpu_device
from src.utils.misc import load_model_wo_clip

import src.utils.fixseed  # noqa

plt.switch_backend('agg')


def create_vocabulary_categories():
    """
    Create a categorized vocabulary for t-SNE analysis.
    Returns a dictionary where keys are category names and values are lists of text descriptions.
    """
    vocabulary_categories = {
        'Walking': [
            'a person walks forward',
            'a person walks backward',
            'a person walks to the left',
            'a person walks to the right',
            'a person is walking slowly',
            'a person is walking fast',
            'someone walks in a circle',
            'a person walks and turns around',
        ],
        'Running': [
            'a person runs forward',
            'a person is running fast',
            'someone is jogging',
            'a person runs in place',
            'a person sprints forward',
            'someone runs backward',
        ],
        'Jumping': [
            'a person jumps up',
            'a person jumps forward',
            'someone does a jumping jack',
            'a person hops on one leg',
            'a person does a vertical jump',
            'someone jumps backward',
        ],
        'Dancing': [
            'a person is dancing',
            'someone dances to music',
            'a person does a dance move',
            'a person is breakdancing',
            'someone is ballet dancing',
            'a person dances freely',
        ],
        'Arm Movements': [
            'a person waves their hand',
            'a person raises their arms',
            'someone claps their hands',
            'a person reaches up high',
            'a person stretches their arms',
            'someone crosses their arms',
        ],
        'Sitting': [
            'a person sits down',
            'a person sits on a chair',
            'someone sits on the ground',
            'a person sits cross-legged',
            'a person sits and stands',
            'someone is sitting',
        ],
        'Kicking': [
            'a person kicks forward',
            'someone kicks a ball',
            'a person does a high kick',
            'a person kicks to the side',
            'someone kicks backward',
            'a person does martial arts kicks',
        ],
        'Punching': [
            'a person punches forward',
            'someone throws a punch',
            'a person does boxing moves',
            'a person punches the air',
            'someone does martial arts punches',
            'a person jabs forward',
        ],
    }
    return vocabulary_categories


def create_motion_categories():
    """
    Create a categorized list of motions for t-SNE analysis.
    Returns a dictionary where keys are category names and values are lists of motion text labels.
    """
    motion_categories = {
        'Walk': [
            'walk',
            'walking',
            'walks forward',
            'walk forward',
            'walking forward',
        ],
        'Run': [
            'run',
            'running',
            'jog',
            'jogging',
            'sprint',
        ],
        'Jump': [
            'jump',
            'jumping',
            'hop',
            'leap',
            'jumps',
        ],
        'Dance': [
            'dance',
            'dancing',
            'dances',
        ],
        'Sit': [
            'sit',
            'sitting',
            'sits',
            'sit down',
        ],
        'Kick': [
            'kick',
            'kicking',
            'kicks',
        ],
        'Punch': [
            'punch',
            'punching',
            'punches',
        ],
        'Wave': [
            'wave',
            'waving',
            'waves',
        ],
    }
    return motion_categories


def visualize_vocabulary_tsne(model, params, folder, epoch):
    """
    Perform t-SNE analysis on vocabulary embeddings in CLIP space.
    """
    print("\n" + "="*80)
    print("VOCABULARY t-SNE ANALYSIS")
    print("="*80 + "\n")
    
    device = params['device']
    vocabulary_categories = create_vocabulary_categories()
    
    # Prepare data
    all_texts = []
    all_labels = []
    all_colors = []
    category_names = []
    
    # Define colors for each category
    colors = plt.cm.tab10(np.linspace(0, 1, len(vocabulary_categories)))
    
    for idx, (category, texts) in enumerate(vocabulary_categories.items()):
        all_texts.extend(texts)
        all_labels.extend([category] * len(texts))
        all_colors.extend([colors[idx]] * len(texts))
        category_names.append(category)
    
    print(f"Total vocabulary items: {len(all_texts)}")
    print(f"Categories: {category_names}")
    
    # Encode texts with CLIP
    print("Encoding texts with CLIP...")
    text_tokens = clip.tokenize(all_texts).to(device)
    with torch.no_grad():
        text_features = model.clip_model.encode_text(text_tokens).float()
        text_features = text_features.cpu().numpy()
    
    # Perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_texts) - 1))
    text_tsne = tsne.fit_transform(text_features)
    
    # Visualize
    plt.figure(figsize=(12, 10))
    
    # Plot points
    for idx, category in enumerate(vocabulary_categories.keys()):
        mask = np.array(all_labels) == category
        plt.scatter(text_tsne[mask, 0], text_tsne[mask, 1], 
                   c=[colors[idx]], label=category, s=100, alpha=0.7)
    
    plt.title(f'Vocabulary t-SNE in CLIP Space (Epoch {epoch})', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    figname = params["figname"].format(epoch)
    save_path = os.path.join(folder, f'vocabulary_tsne_{figname}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Vocabulary t-SNE visualization saved to: {save_path}")
    print("="*80 + "\n")
    
    return text_tsne, all_labels


def visualize_motion_tsne(model, datasets, params, folder, epoch):
    """
    Perform t-SNE analysis on motion embeddings in CLIP space.
    """
    print("\n" + "="*80)
    print("MOTION t-SNE ANALYSIS")
    print("="*80 + "\n")
    
    device = params['device']
    motion_categories = create_motion_categories()
    motion_collection = get_motion_text_mapping(datasets)
    
    # Prepare data
    all_motions = []
    all_labels = []
    all_colors = []
    category_names = []
    
    # Define colors for each category
    colors = plt.cm.tab10(np.linspace(0, 1, len(motion_categories)))
    
    print("Retrieving motions from dataset...")
    for idx, (category, motion_texts) in enumerate(motion_categories.items()):
        category_names.append(category)
        for motion_text in motion_texts:
            # Try to find motion in collection
            found = False
            for key, value in motion_collection.items():
                # Handle value as list, tuple, or string
                if isinstance(value, (list, tuple)):
                    value_str = str(value[0]) if len(value) > 0 else ""
                else:
                    value_str = str(value)
                if motion_text.lower() in value_str.lower() or value_str.lower() in motion_text.lower():
                    # Retrieve motion
                    split_name, mot_id = key.split('_##_')
                    motion = datasets[split_name].get_item_by_id(int(mot_id))
                    all_motions.append(motion['x'])
                    all_labels.append(category)
                    all_colors.append(colors[idx])
                    found = True
                    break
            if found:
                break  # Only take one example per motion text
    
    if len(all_motions) == 0:
        print("Warning: No motions found in dataset. Skipping motion t-SNE.")
        return None, None
    
    print(f"Total motion samples: {len(all_motions)}")
    print(f"Categories: {category_names}")
    
    # Stack motions
    motions = torch.stack(all_motions, dim=0).to(device)
    
    # Encode motions with the model
    print("Encoding motions...")
    model.eval()
    with torch.no_grad():
        # Create dummy labels and mask
        dummy_labels = torch.zeros(motions.shape[0], dtype=int, device=device)
        lengths = torch.ones(motions.shape[0], dtype=int, device=device) * 60
        mask = model.lengths_to_mask(lengths)
        
        # Encode motions to latent space
        motion_latents = model.encoder({
            'x': motions,
            'y': dummy_labels,
            'mask': mask
        })["mu"]
        
        motion_latents = motion_latents.cpu().numpy()
    
    # Perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_motions) - 1))
    motion_tsne = tsne.fit_transform(motion_latents)
    
    # Visualize
    plt.figure(figsize=(12, 10))
    
    # Plot points
    for idx, category in enumerate(motion_categories.keys()):
        mask = np.array(all_labels) == category
        if np.any(mask):
            plt.scatter(motion_tsne[mask, 0], motion_tsne[mask, 1], 
                       c=[colors[idx]], label=category, s=100, alpha=0.7)
    
    plt.title(f'Motion t-SNE in CLIP Space (Epoch {epoch})', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    figname = params["figname"].format(epoch)
    save_path = os.path.join(folder, f'motion_tsne_{figname}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Motion t-SNE visualization saved to: {save_path}")
    print("="*80 + "\n")
    
    return motion_tsne, all_labels


def visualize_combined_tsne(model, datasets, params, folder, epoch):
    """
    Perform combined t-SNE analysis showing both vocabulary and motion embeddings.
    """
    print("\n" + "="*80)
    print("COMBINED VOCABULARY + MOTION t-SNE ANALYSIS")
    print("="*80 + "\n")
    
    device = params['device']
    
    # Get vocabulary embeddings
    vocabulary_categories = create_vocabulary_categories()
    all_texts = []
    vocab_labels = []
    
    for category, texts in vocabulary_categories.items():
        all_texts.extend(texts)
        vocab_labels.extend([f"Text: {category}"] * len(texts))
    
    print(f"Encoding {len(all_texts)} vocabulary items...")
    text_tokens = clip.tokenize(all_texts).to(device)
    with torch.no_grad():
        text_features = model.clip_model.encode_text(text_tokens).float()
        text_features = text_features.cpu().numpy()
    
    # Get motion embeddings
    motion_categories = create_motion_categories()
    motion_collection = get_motion_text_mapping(datasets)
    
    all_motions = []
    motion_labels = []
    
    print("Retrieving and encoding motions...")
    for category, motion_texts in motion_categories.items():
        for motion_text in motion_texts:
            found = False
            for key, value in motion_collection.items():
                # Handle value as list, tuple, or string
                if isinstance(value, (list, tuple)):
                    value_str = str(value[0]) if len(value) > 0 else ""
                else:
                    value_str = str(value)
                if motion_text.lower() in value_str.lower() or value_str.lower() in motion_text.lower():
                    split_name, mot_id = key.split('_##_')
                    motion = datasets[split_name].get_item_by_id(int(mot_id))
                    all_motions.append(motion['x'])
                    motion_labels.append(f"Motion: {category}")
                    found = True
                    break
            if found:
                break
    
    if len(all_motions) > 0:
        motions = torch.stack(all_motions, dim=0).to(device)
        
        model.eval()
        with torch.no_grad():
            dummy_labels = torch.zeros(motions.shape[0], dtype=int, device=device)
            lengths = torch.ones(motions.shape[0], dtype=int, device=device) * 60
            mask = model.lengths_to_mask(lengths)
            
            motion_latents = model.encoder({
                'x': motions,
                'y': dummy_labels,
                'mask': mask
            })["mu"]
            
            motion_latents = motion_latents.cpu().numpy()
        
        # Combine embeddings
        combined_embeddings = np.vstack([text_features, motion_latents])
        combined_labels = vocab_labels + motion_labels
        
        print(f"Total samples: {len(combined_labels)} ({len(vocab_labels)} vocab + {len(motion_labels)} motion)")
        
        # Perform t-SNE
        print("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_labels) - 1))
        combined_tsne = tsne.fit_transform(combined_embeddings)
        
        # Visualize
        plt.figure(figsize=(14, 10))
        
        # Get unique labels and assign colors
        unique_labels = sorted(list(set(combined_labels)))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for idx, label in enumerate(unique_labels):
            mask = np.array(combined_labels) == label
            marker = 'o' if label.startswith('Text:') else '^'
            plt.scatter(combined_tsne[mask, 0], combined_tsne[mask, 1], 
                       c=[colors[idx]], label=label, s=100, alpha=0.7, marker=marker)
        
        plt.title(f'Combined Vocabulary + Motion t-SNE in CLIP Space (Epoch {epoch})', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(loc='best', fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        figname = params["figname"].format(epoch)
        save_path = os.path.join(folder, f'combined_tsne_{figname}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined t-SNE visualization saved to: {save_path}")
        print("="*80 + "\n")
    else:
        print("Warning: No motions found. Skipping combined visualization.")


def main():
    # Parse options
    parameters, folder, checkpointname, epoch = parser()
    gpu_device = get_gpu_device()
    parameters["device"] = f"cuda:{gpu_device}"
    
    # Load model and datasets
    model, datasets = get_model_and_data(parameters, split='all')
    
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)
    
    # Perform t-SNE analyses
    visualize_vocabulary_tsne(model, parameters, folder, epoch)
    visualize_motion_tsne(model, datasets, parameters, folder, epoch)
    visualize_combined_tsne(model, datasets, parameters, folder, epoch)
    
    print("\n" + "="*80)
    print("t-SNE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
