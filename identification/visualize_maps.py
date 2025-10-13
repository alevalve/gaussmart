import math
from typing import Union
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.functional as TF

class DINOImageEncoder(torch.nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str = "cuda",
        fp16: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.model = self.model.to(self.device)
        if fp16 and self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()

        # keep mean/std in case you want manual normalization for raw tensors
        self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std  = torch.tensor(self.processor.image_std,  device=self.device).view(1, 3, 1, 1)

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def _to_pil(self, img: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, np.ndarray):
            # HWC (uint8 or float [0,1])
            if img.ndim == 3 and img.shape[2] == 3:
                if img.dtype != np.uint8:
                    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                return Image.fromarray(img, mode="RGB")
        if isinstance(img, torch.Tensor):
            # Expect CHW in [0,1]
            if img.ndim == 3 and img.shape[0] == 3:
                img = torch.clamp(img, 0, 1).cpu()
                return TF.to_pil_image(img)
        raise ValueError("Unsupported image type/shape for input.")

    @torch.inference_mode()
    def _preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        # Use HF processor to ensure correct resize/crop/normalize for this checkpoint
        enc = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = enc["pixel_values"].to(self.device)
        # Cast to half if model is half
        if next(self.model.parameters()).dtype == torch.float16:
            pixel_values = pixel_values.half()
        return pixel_values  # (1,3,H',W')

    @torch.inference_mode()
    def encode(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Returns the pooled image embedding (CLS) as a 1D tensor.
        """
        pil = self._to_pil(image)
        pixel_values = self._preprocess(pil)
        outputs = self.model(pixel_values)
        # Most ViT-like models return last_hidden_state (B, 1+N, D) and pooler_output (B, D)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output[0]  # (D,)
        else:
            # If no pooler, fall back to CLS token from last_hidden_state
            emb = outputs.last_hidden_state[0, 0]
        return emb

    @torch.inference_mode()
    def features_and_heatmap(self, image):
        # Keep a copy of original size for upsampling
        pil = self._to_pil(image)
        W0, H0 = pil.size

        pixel_values = self._preprocess(pil)              # (1,3,h,w)
        outputs = self.model(pixel_values)
        hidden = outputs.last_hidden_state               # (1, 1+N_tokens, D)

        cls = hidden[:, 0:1, :]                          # (1,1,D)

        # Figure out patch grid from input size and model patch size
        # Try to read patch size from config; fall back to 16.
        patch = getattr(getattr(self.model, "config", object()), "patch_size", 16)
        if isinstance(patch, (list, tuple)):
            patch = patch[0]
        h, w = pixel_values.shape[-2:]
        gh, gw = h // patch, w // patch
        n_patches = gh * gw

        # Tokens after CLS (may include register tokens)
        tokens_all = hidden[:, 1:, :]                    # (1, N_after_cls, D)
        n_after_cls = tokens_all.shape[1]
        n_reg = n_after_cls - n_patches                  # how many non-patch tokens

        if n_reg < 0:
            raise RuntimeError(
                f"Computed negative register tokens (n_reg={n_reg}). "
                f"Check patch size inference: h={h}, w={w}, patch={patch}"
            )

        # Keep only the last n_patches as patch tokens (drop registers at front)
        tokens = tokens_all[:, n_reg:, :]                # (1, n_patches, D)

        # Cosine sim of each patch token to CLS
        import torch.nn.functional as F
        cls_f = F.normalize(cls.float(), dim=-1)         # (1,1,D)
        tok_f = F.normalize(tokens.float(), dim=-1)      # (1,N,D)
        sim = torch.matmul(tok_f, cls_f.transpose(-1, -2)).squeeze(-1)  # (1,N)

        sim_map = sim.reshape(1, 1, gh, gw)              # (1,1,gh,gw)

        # Normalize to [0,1]
        sim_map = sim_map - sim_map.min()
        sim_map = sim_map / (sim_map.max() + 1e-6)

        # Upsample to original image size
        heat_up = torch.nn.functional.interpolate(
            sim_map, size=(H0, W0), mode="bilinear", align_corners=False
        )[0, 0].cpu().float().clamp(0, 1)

        # Global embedding (prefer pooler if available)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output[0].float()
        else:
            emb = hidden[0, 0].float()  # CLS

        return emb, heat_up

    
    @torch.inference_mode()
    def overlay_heatmap(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor], heatmap: torch.Tensor, alpha=0.45) -> Image.Image:
        """
        Returns a PIL image overlay: original RGB with grayscale heatmap overlay.
        (feel free to swap for a colormap if you prefer)
        """
        pil = self._to_pil(image)
        heat = heatmap.numpy()
        # make a 3-channel heat (grayscale)
        heat_rgb = np.stack([heat, heat, heat], axis=-1)   # H,W,3 in [0,1]
        heat_rgb = (heat_rgb * 255.0).astype(np.uint8)
        heat_img = Image.fromarray(heat_rgb, mode="RGB")
        # blend: original * (1-alpha) + heat * alpha
        return Image.blend(pil.convert("RGB"), heat_img, alpha=alpha)


encoder = DINOImageEncoder()
image_path = '/data1/alex/datasets/tanks_templates/tanksandtemples/courthouse/images/000001.jpg'
emb = encoder.encode(image_path)  # (D,)
emb, heat = encoder.features_and_heatmap(image_path)
vis = encoder.overlay_heatmap(image_path, heat, alpha=0.5)
vis.save("courthouse_dinov3_heatmap.png")
