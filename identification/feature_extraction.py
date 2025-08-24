import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoImageProcessor, AutoModel
from typing import Optional, Tuple
from typing import List
from PIL import Image
from huggingface_hub import login



login(token="")

class DINOImageEncoder(nn.Module):
    def __init__(self, pretrained_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        
        # Move to CUDA and convert to FP16
        self.model = self.model.cuda().half()
        self.model.eval()
        
        # Convert normalization values to FP16
        self.mean = torch.tensor(self.processor.image_mean, device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
        
        # Disable gradients for all parameters
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def encode_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process single tensor
        """
        image_batch = image_tensor.half().unsqueeze(0)  
        
        image_normalized = (image_batch - self.mean) / self.std
        
        outputs = self.model(image_normalized)
        return outputs.pooler_output.squeeze(0)
