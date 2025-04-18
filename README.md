```markdown
# Defect-GAN Training Framework

This repository provides a training framework for Defect-GAN, a generative adversarial network designed for defect image generation. The framework includes dataset preparation, model training, and image generation capabilities.

## Features

- **Custom Dataset Loader**: Handles defect and normal images with corresponding masks.
- **GAN Model Implementation**: Includes a Generator and Discriminator optimized for defect image generation.
- **Training Utilities**: Supports mixed precision training, learning rate scheduling, and loss visualization.
- **Sample Image Generation**: Periodically generates sample images for qualitative evaluation.
- **Batch Inference Tool**: A `sample.py` script for generating synthesized defect images from a folder of normal images and masks.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (if training on GPU)
- PyTorch 1.10+ with torchvision
- Required Python packages: listed in `requirements.txt`

## Reference

This project is based on concepts from the paper **"Defect-GAN: High-Fidelity Defect Synthesis for Automated Defect Inspection"**. You can access the paper [here](https://arxiv.org/pdf/2103.15158).

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/defect-gan.git
   cd defect-gan
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Dataset Preparation

- Organize your dataset as follows:
  ```
  dataset/
  ├── train/
  │   ├── normal_images/
  │   ├── defect_images/
  │   └── masks/
  ```

- Ensure that defect images and their corresponding masks have matching filenames.

### Training

Run the training script using the following command:

```bash
python main.py --dataset /path/to/dataset --batch_size 32 --epoch 2000 --size 128
```

#### Key Arguments:
- `--batch_size`: Batch size for training (default: 32)
- `--epoch`: Number of epochs (default: 2000)
- `--device`: Device to use ('cuda' or 'cpu', default: 'cuda')
- `--dataset`: Path to the dataset
- `--size`: Image size for training (default: 128)
- Other arguments are configurable in `main.py`.

### Inference (sample.py)

You can use `sample.py` to synthesize defect images from a folder of normal images and corresponding grayscale masks.

#### Input Folder Format

```
your_input_dir/
├── img001.png
├── img001_mask.png
├── img002.png
├── img002_mask.png
```

#### Run Inference

```bash
python sample.py \
  --checkpoint ./save_model/gen_epoch200.pth \
  --input_dir ./your_input_dir \
  --output_dir ./results \
  --size 128 128 \
  --device cuda
```

#### Output

- The synthesized defect images will be saved to the specified `--output_dir`.

---

## Directory Structure

```
.
├── dataset.py        # Dataset loader for defect images and masks
├── main.py           # Entry point for training
├── trainer.py        # Training loop and utilities
├── sample.py         # Batch inference script to generate defect images
├── model/            # GAN model implementations
├── samples/          # Generated sample images (auto-created)
├── save_model/       # Saved model checkpoints (auto-created)
└── requirements.txt  # Python dependencies
```

---

## Changelog

### 2025/04/18
- ✨ Added `sample.py` for batch inference
