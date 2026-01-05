import os
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
import yaml
from sklearn.manifold import TSNE
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.visualize.visualize import get_motion_text_mapping, get_gpu_device, retrieve_motions, encode_motions
from src.utils.misc import load_model_wo_clip

import src.utils.fixseed  # noqa

plt.switch_backend('agg')


def load_categories_from_yaml(yaml_path=None):
    """
    Load motion and vocabulary categories from YAML configuration file.
    Returns a tuple of (motion_categories, vocabulary_categories).
    """
    if yaml_path is None:
        # Default path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, 'motion_categories.yaml')
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Category YAML file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    motion_categories = config.get('motion_categories', {})
    vocabulary_categories = config.get('vocabulary_categories', {})
    
    print(f"Loaded categories from: {yaml_path}")
    print(f"  Motion categories: {len(motion_categories)}")
    print(f"  Vocabulary categories: {len(vocabulary_categories)}")
    
    return motion_categories, vocabulary_categories



def visualize_vocabulary_tsne(model, params, folder, epoch, vocabulary_categories=None):
    """
    Perform t-SNE analysis on vocabulary embeddings in CLIP space.
    """
    print("\n" + "="*80)
    print("VOCABULARY t-SNE ANALYSIS")
    print("="*80 + "\n")
    
    device = params['device']
    if vocabulary_categories is None:
        _, vocabulary_categories = load_categories_from_yaml()
    
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


def visualize_motion_tsne(model, datasets, params, folder, epoch, motion_categories=None):
    """
    Perform t-SNE analysis on motion embeddings in CLIP space.
    """
    print("\n" + "="*80)
    print("MOTION t-SNE ANALYSIS")
    print("="*80 + "\n")
    
    device = params['device']
    if motion_categories is None:
        motion_categories, _ = load_categories_from_yaml()
    motion_collection = get_motion_text_mapping(datasets)
    
    # Prepare data
    all_labels = []
    all_colors = []
    category_names = []
    
    # Define colors for each category
    colors = plt.cm.tab10(np.linspace(0, 1, len(motion_categories)))
    
    print("Retrieving motions from dataset...")
    motion_texts_to_retrieve = []
    for idx, (category, motion_texts) in enumerate(motion_categories.items()):
        category_names.append(category)
        for motion_text in motion_texts:
            # Try to find motion in collection
            found = False
            if motion_text in motion_collection:
                motion_texts_to_retrieve.append(motion_text)
                all_labels.append(category)
                all_colors.append(colors[idx])
                found = True
            else:
                # Fallback to fuzzy matching
                for key in motion_collection.keys():
                    if motion_text.lower() in key.lower() or key.lower() in motion_text.lower():
                        motion_texts_to_retrieve.append(key)
                        all_labels.append(category)
                        all_colors.append(colors[idx])
                        found = True
                        break
            if found:
                break  # Only take one example per motion text
    
    if len(motion_texts_to_retrieve) == 0:
        print("Warning: No motions found in dataset. Skipping motion t-SNE.")
        return None, None
    
    print(f"Total motion samples: {len(motion_texts_to_retrieve)}")
    print(f"Categories: {category_names}")
    
    # Retrieve motions using the helper function
    motions = retrieve_motions(datasets, motion_collection, motion_texts_to_retrieve, device)
    
    # Encode motions with the model using the helper function
    print("Encoding motions...")
    model.eval()
    with torch.no_grad():
        motion_latents = encode_motions(model, motions, device)
        motion_latents = motion_latents.cpu().numpy()
    
    # Perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(motion_latents) - 1))
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


def visualize_combined_tsne(model, datasets, params, folder, epoch, motion_categories=None, vocabulary_categories=None):
    """
    Perform combined t-SNE analysis showing both vocabulary and motion embeddings.
    """
    print("\n" + "="*80)
    print("COMBINED VOCABULARY + MOTION t-SNE ANALYSIS")
    print("="*80 + "\n")
    
    device = params['device']
    
    # Load categories if not provided
    if motion_categories is None or vocabulary_categories is None:
        motion_categories, vocabulary_categories = load_categories_from_yaml()
    
    # Get vocabulary embeddings
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
    motion_collection = get_motion_text_mapping(datasets)
    
    motion_labels = []
    
    print("Retrieving and encoding motions...")
    motion_texts_to_retrieve = []
    for category, motion_texts in motion_categories.items():
        for motion_text in motion_texts:
            found = False
            # Try exact match first, then fuzzy match
            if motion_text in motion_collection:
                print(f"  Found motion for text '{motion_text}' in category '{category}'")
                motion_texts_to_retrieve.append(motion_text)
                motion_labels.append(f"Motion: {category}")
                found = True
            else:
                # Fallback to fuzzy matching if exact match fails
                for key in motion_collection.keys():
                    if motion_text.lower() in key.lower() or key.lower() in motion_text.lower():
                        print(f"  Found motion for text '{motion_text}' (matched '{key}') in category '{category}'")
                        motion_texts_to_retrieve.append(key)
                        motion_labels.append(f"Motion: {category}")
                        found = True
                        break
            if found:
                break
    
    if len(motion_texts_to_retrieve) > 0:
        # Retrieve motions using the helper function
        motions = retrieve_motions(datasets, motion_collection, motion_texts_to_retrieve, device)
        
        model.eval()
        with torch.no_grad():
            # Encode motions using the helper function
            motion_latents = encode_motions(model, motions, device)
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
    
    # Load categories from YAML
    motion_categories, vocabulary_categories = load_categories_from_yaml()
    
    # Perform t-SNE analyses
    visualize_vocabulary_tsne(model, parameters, folder, epoch, vocabulary_categories)
    visualize_motion_tsne(model, datasets, parameters, folder, epoch, motion_categories)
    visualize_combined_tsne(model, datasets, parameters, folder, epoch, motion_categories, vocabulary_categories)
    
    print("\n" + "="*80)
    print("t-SNE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
