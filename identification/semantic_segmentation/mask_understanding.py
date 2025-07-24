import torch
from transformers import CLIPProcessor, CLIPModel
import argparse
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from itertools import combinations
from pathlib import Path
import numpy as np
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
                 dino_model_name="facebook/dinov2-large"):
    
        """Initialize both CLIP and DINO models"""

        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.eval()

        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        self.dino_model = AutoModel.from_pretrained(dino_model_name)
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

def linear_projection(dino_feat, clip_feat, dim=512):

    dino_projector = Projected(dino_feat.shape[-1], dim)
    clip_projector = Projected(clip_feat.shape[-1], dim)
    dino_embedding = dino_projector(torch.tensor(dino_feat).float())
    clip_embedding = clip_projector(torch.tensor(clip_feat).float())
    return dino_embedding, clip_embedding


def test_feature_extraction(image_path, mask_path):
    feature_extractor = SemanticFeatureExtractor()

    # Load and resize the input image to (224, 224)
    dummy_image = Image.open(image_path)
    resized_image = dummy_image.resize((224, 224), resample=Image.Resampling.NEAREST)

    # Load all masks from .npz file
    npz = np.load(mask_path)
    masks = npz["masks"]  # shape: (N, H, W)

    features_clip = []
    features_dino = []

    reshaped_clip = []
    reshaped_dino = []

    attention_embeds = []

    attention = MultiHeadAttention(d_model=512, num_heads=8)

    # Loop through each instance mask
    for i in range(masks.shape[0]):
        mask_i = masks[i]  # shape: (H, W)

        # Resize mask to (224, 224)
        mask_img = Image.fromarray(mask_i.astype(np.uint8))
        resized_mask = mask_img.resize((224, 224), resample=Image.Resampling.NEAREST)
        resized_mask_np = np.array(resized_mask)

        # Extract CLIP features
        feat_clip = feature_extractor.extract_clip_features(resized_image, resized_mask_np)
        features_clip.append(feat_clip)

        # Extract features
        feat_dino = feature_extractor.extract_dino_features(resized_image, resized_mask_np)
        features_dino.append(feat_dino)

        # Obtained reshape vectors

        dino_embed, clip_embed = linear_projection(feat_dino, feat_clip, 512) 
        reshaped_dino.append(dino_embed)
        reshaped_clip.append(clip_embed)

        # Attention vectors 

        result = attention(query=dino_embed, key=clip_embed, value=clip_embed)
        attention_embeds.append(result)


    return attention_embeds


def attention_embeddings(images_dir, masks_dir):

        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        
        for image_file in sorted(images_dir.glob("*.png")):

            idx = image_file.stem.split("_")[-1]
            mask_file = masks_dir / f"segments_{idx}.npz"

            if not mask_file.exists():
                print(f"Warning: No mask found for {image_file.name}, skipping.")
                continue
            
        
            attention_embeds = test_feature_extraction(str(image_file), str(mask_file))
    
            all_embeds = torch.cat(attention_embeds, dim=0)
            torch.save(all_embeds, f"seg_attention_{idx}.pt")
            all_embeds = torch.load(f"seg_attention_{idx}.pt", map_location="cpu")

      
if __name__ == "__main__":
    attention_embeddings(
        images_dir="/data3/alex/gaussmart/identification/results/segments/segmented_images",
        masks_dir="/data3/alex/gaussmart/identification/results/segments/masks"
    )