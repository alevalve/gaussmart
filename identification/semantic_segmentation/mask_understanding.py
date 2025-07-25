import torch
from transformers import CLIPProcessor, CLIPModel
import argparse
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from itertools import combinations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import shutil
import json
import torch.nn.functional as F
from feature_extraction import MultiHeadAttention
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss, MatryoshkaLoss
import torch 
from feature_extraction import Projected

class SemanticFeatureExtractor:
    def __init__(self,
                 clip_model_name="openai/clip-vit-large-patch14",
                 dino_model_name="facebook/dinov2-large",
                 device='cuda'):
    
        """Initialize both CLIP and DINO models"""

        self.device = device

        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.eval()

        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        self.dino_model = AutoModel.from_pretrained(dino_model_name).to(device)
        self.dino_model.eval()

    def extract_clip_features(self, image, mask):
        """
        Extract semantic information from image region
        """ 
        if isinstance(image, torch.Tensor):
            if image.max() <= 1.0:
                image = (image * 255).byte()
            image = Image.fromarray(image.numpy())
        elif isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            elif isinstance(mask, np.ndarray):
                mask_np = mask
            else:
                raise TypeError("Mask must be a torch.Tensor or numpy.ndarray")

            if mask_np.ndim == 2:
                img_array = np.array(image)
                img_array = img_array * mask_np[:, :, np.newaxis]
                image = Image.fromarray(img_array.astype(np.uint8))

        inputs = self.clip_processor(images=image, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)

        return image_features


    def extract_dino_features(self, image, mask):
        """Extract visual similarity from original image with masks"""

        if isinstance(image, torch.Tensor):
            if image.max() <= 1.0:
                image = (image * 255).byte()
            image = Image.fromarray(image.numpy())
        elif isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Apply mask if provided
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            elif isinstance(mask, np.ndarray):
                mask_np = mask
            else:
                raise TypeError("Mask must be a torch.Tensor or numpy.ndarray")

            if mask_np.ndim == 2:
                img_array = np.array(image)
                img_array = img_array * mask_np[:, :, np.newaxis]
                image = Image.fromarray(img_array.astype(np.uint8))

        # Use DINO processor
        inputs = self.dino_processor(images=image, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # DINO features
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                dino_features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                dino_features = outputs.last_hidden_state[:, 0, :]
            else:
                dino_features = outputs[0][:, 0, :]

            dino_features = F.normalize(dino_features, p=2, dim=-1)
        
        return dino_features

def linear_projection(dino_feat, clip_feat, dim=512, device='cuda'):

    dino_projector = Projected(dino_feat.shape[-1], dim).to(device)
    clip_projector = Projected(clip_feat.shape[-1], dim).to(device)

    if not isinstance(dino_feat, torch.Tensor):
        dino_feat = torch.tensor(dino_feat).float().to(device)
    
    if not isinstance(clip_feat, torch.Tensor):
        clip_feat = torch.tensor(clip_feat).float().to(device)

    dino_embedding = dino_projector(torch.tensor(dino_feat).float())
    clip_embedding = clip_projector(torch.tensor(clip_feat).float())
    return dino_embedding, clip_embedding


def test_feature_extraction(image_paths_and_masks, device="cuda"):
    """Process multiple images in a single batch on GPU"""
    feature_extractor = SemanticFeatureExtractor(device=device)
    attention = MultiHeadAttention(d_model=512, num_heads=8).to(device)
    
    results = {}
    
    for image_path, mask_path, idx in image_paths_and_masks:
        # Load and resize the input image to (224, 224)
        dummy_image = Image.open(image_path)
        resized_image = dummy_image.resize((224, 224), resample=Image.Resampling.NEAREST)

        # Load all masks from .npz file
        npz = np.load(mask_path)
        masks = npz["masks"]  # shape: (N, H, W)

        attention_embeds = []
        reshaped_clip = []

        # Process all masks for this image
        for i in range(masks.shape[0]):
            mask_i = masks[i]  # shape: (H, W)

            # Resize mask to (224, 224)
            mask_img = Image.fromarray(mask_i.astype(np.uint8))
            resized_mask = mask_img.resize((224, 224), resample=Image.Resampling.NEAREST)
            resized_mask_np = np.array(resized_mask)

            # Extract features on GPU
            feat_clip = feature_extractor.extract_clip_features(resized_image, resized_mask_np)
            feat_dino = feature_extractor.extract_dino_features(resized_image, resized_mask_np)

            # Linear projection on GPU
            dino_embed, clip_embed = linear_projection(feat_dino, feat_clip, 512, device)
            reshaped_clip.append(clip_embed.cpu())  

            # Attention on GPU
            result = attention(query=dino_embed, key=clip_embed, value=clip_embed)
            attention_embeds.append(result.cpu())  

        results[idx] = (attention_embeds, reshaped_clip)
    
    return results

def embeddings_gpu_optimized(images_dir, masks_dir, batch_size=4, num_workers=1):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    
    
    embeddings_dir = Path("embeddings")
    shutil.rmtree(embeddings_dir, ignore_errors=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    image_tasks = []

    for image_file in sorted(images_dir.glob("*.png")):
        idx = image_file.stem.split("_")[-1]
        mask_file = masks_dir / f"segments_{idx}.npz"

        if not mask_file.exists():
            print("No mask found")
            continue

        image_tasks.append((str(image_file), str(mask_file), idx))
    
    batches = [image_tasks[i:i + batch_size] for i in range(0, len(image_tasks), batch_size)]

    # Process each batch

    def process_batch(batch, batch_idx):
        device = "cuda"
        results = test_feature_extraction(batch, device=device)
        for idx, (attention_embeds, reshaped_clips) in results.items():
            torch.save(attention_embeds, embeddings_dir / f"seg_attention_{idx}.pt")
            torch.save(reshaped_clips, embeddings_dir / f"reshaped_clip_{idx}.pt")
        return f"Processed batch {batch_idx + 1} with {len(batch)} images"
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch, i): i
            for i, batch in enumerate(batches)
        }

        completed = 0
        for future in as_completed(future_to_batch):
            result = future.result()
            completed += 1
            print(f"[{completed} / {len(batches)}]")
            torch.cuda.empty_cache()

            

if __name__ == "__main__":
    embeddings_gpu_optimized(
        images_dir="/data3/alex/gaussmart/identification/results/segments/segmented_images",
        masks_dir="/data3/alex/gaussmart/identification/results/segments/masks",
        batch_size=4,
        num_workers=1
    )