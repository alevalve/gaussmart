# GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering

[![Paper](https://img.shields.io/badge/arXiv-2510.14270-b31b1b.svg)](https://arxiv.org/abs/2510.14270)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

GauSSmart is a hybrid method that bridges 2D foundational models and 3D Gaussian Splatting reconstruction. Our approach integrates established 2D computer vision techniques, including convex filtering and semantic feature supervision from foundational models such as DINO, to enhance Gaussian-based scene reconstruction.

### Key Features

- **Convex-guided outlier removal** for cleaner point clouds
- **Segment-aware point cloud densification** using SAM-derived masks
- **Embedding-aligned training** with DINOv3 features
- **Superior performance** on DTU, Mip-NeRF 360, and Tanks & Temples datasets

## Installation

### Prerequisites
- CUDA 11.6 or higher
- Python 3.8+
- Conda/Miniconda

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/gaussmart.git
cd gaussmart
```

2. **Create and activate conda environment:**
```bash
conda env create -f environment.yml
conda activate gaussmart
```

3. **Install PyTorch with CUDA support:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Install additional requirements:**
```bash
pip install -r requirements.txt
```

5. **Install custom CUDA modules:**
```bash
pip install submodules/simple-knn
pip install submodules/diff-surfel-rasterization
```

6. **Download SAM weights:**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P identification/weights/
```

7. **Set up Hugging Face token (for DINOv3):**
```bash
export HUGGINGFACE_HUB_TOKEN="hf_YOURTOKEN"
```


## Training

### Basic Training
```bash
python train.py --config configs/gaussmart.yaml --dataset_path data/dtu/scan24
```

### With Custom Parameters
```bash
python train.py \
    --config configs/gaussmart.yaml \
    --dataset_path data/dtu/scan24 \
    --iterations 30000 \
    --lambda_dino 0.05 \
    --use_convex_hull \
    --densify_segments
```

### Training Options
- `--iterations`: Number of training iterations (default: 30000)
- `--lambda_dino`: Weight for DINO loss (default: 0.05)
- `--use_convex_hull`: Enable convex hull outlier removal
- `--densify_segments`: Enable segment-aware densification
- `--min_segment_points`: Minimum points per segment (default: 5)

### Render novel views:
```bash
python render.py --checkpoint_path outputs/checkpoint.pth --output_dir renders/
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{valverde2025gaussmartenhanced3dreconstruction,
      title={GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering}, 
      author={Alexander Valverde and Brian Xu and Yuyin Zhou and Meng Xu and Hongyun Wang},
      year={2025},
      eprint={2510.14270},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14270}, 
}
```

## Acknowledgments

This work builds upon:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)
- [DINOv3](https://github.com/facebookresearch/dinov2)
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
