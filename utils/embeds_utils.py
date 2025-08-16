import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np


def tensor_to_pil(tensor_img):
    """Convert a tensor image [C, H, W] in range [0, 1] to PIL Image"""
    np_img = (tensor_img.cpu().numpy() * 255).astype(np.uint8)
    np_img = np_img.transpose(1, 2, 0)  # [H, W, C]
    return Image.fromarray(np_img)


def load_gt_embeddings(embeddings_path, train_cameras):
    """
    Load pre-computed ground truth DINO embeddings from saved files.
    
    Args:
        embeddings_path: Path to directory containing saved embedding files
        train_cameras: List of camera objects from the scene
        
    Returns:
        Dictionary mapping camera indices to their embeddings
    """
    import csv
    from pathlib import Path
    
    embeddings_path = Path(embeddings_path)
    gt_embeddings = {}
    
    # Load the index.csv file to map stems to filenames
    index_file = embeddings_path / "index.csv"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found at {index_file}")
    
    # Create mapping from image stem to embedding file
    stem_to_embedding = {}
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row['stem']
            embedding_file = embeddings_path / f"{stem}_dino.pt"
            if embedding_file.exists():
                stem_to_embedding[stem] = embedding_file
    
    for idx, camera in enumerate(train_cameras):
        image_path = Path(camera.image_name)
        image_stem = image_path.stem
        
        if image_stem in stem_to_embedding:
            embedding_path = stem_to_embedding[image_stem]
            try:
                embedding = torch.load(embedding_path, map_location='cuda')
                gt_embeddings[idx] = embedding
            except Exception as e:
                print(f"Warning: Failed to load embedding for {image_stem}: {e}")
        else:
            print(f"Warning: No embedding found for camera {idx} with image {image_stem}")
    
    return gt_embeddings
