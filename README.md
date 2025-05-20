# GauSSmart: Semantic-Guided Gaussian Splatting for 3D Reconstruction

This repo contains the official implementation for the paper "GauSSmart: Semantic-Guided Gaussian Splatting for 3D Reconstruction". Our work enhances 2D Gaussian Splatting by integrating 2D segmentation masks for each object in the scene. By leveraging segmentation information, we improve the point cloud and in addition, we guide the Gaussian splat generation process to improve coverage in underrepresented regions, leading to accurate scene and mesh reconstructions.

![Teaser_pdf_pages-to-jpg-0001](https://github.com/user-attachments/assets/8a14f70c-7133-4f55-9d5d-1979bff232dc)


## New Features
- Semantic-guided reconstruction that preserves fine object details
- Camera clustering for efficient viewpoint selection
- Point cloud enhancement for underrepresented areas
- Segment consistency loss for improved structural coherence
- Support for Tanks and Temples, DTU, and Mip-NeRF 360 datasets

## Installation

```bash
# download
git clone https://github.com/agvalver/gaussmart.git --recursive

# if you have an environment used for 3dgs, use it
# if not, create a new environment
conda env create --file environment.yml
conda activate surfel_splatting
```

## Training
To train a scene with GauSSmart, use:

```bash
python train.py \
  -s <path to dataset> \
  --run_segmentation \
  --dataset_type <dtu/tyt/nerf> \
  --lambda_segment 0.05
```

## Tips for parameter tuning

- lambda_segment=0.05 works well for most scenes
- For unbounded/large scenes, use depth_ratio=0 for fewer "disk-aliasing" artifacts
- The segmentation loss is most effective when applied during the first 7,000 iterations

## Mesh extraction

To export a mesh with GauSSmart's improved geometric accuracy:

```bash
python render.py -m <path to pre-trained model> -s <path to dataset>
```

## Quick example

```bash
python train.py \
  -s /path/to/tanksandtemples/caterpillar \
  --run_segmentation \
  --dataset_type tyt \
  --lambda_segment 0.05

python render.py \
  -m output/caterpillar \
  -s /path/to/tanksandtemples/caterpillar
```
## Acknowledgements
Our work builds on 2D Gaussian Splatting and integrates segmentation from Segment Anything. We thank the authors of these works for their contributions.




