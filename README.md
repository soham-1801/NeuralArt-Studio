# Neural Style Transfer (AdaIN)

A PyTorch implementation of Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (AdaIN) with a Flask web interface for easy style transfer.

## Project Overview

This project implements neural style transfer using Adaptive Instance Normalization, allowing you to apply artistic styles from one image to the content of another. The implementation includes both a training script and a web application for interactive style transfer.

## Features

- **AdaIN-based Style Transfer**: Fast and high-quality style transfer using adaptive instance normalization
- **Web Interface**: User-friendly Flask application for real-time style transfer
- **GPU Support**: CUDA support for faster processing (falls back to CPU if unavailable)
- **Adjustable Parameters**: Control over style transfer strength via alpha parameter
- **Pre-trained Models**: Support for loading pre-trained encoder and decoder models

## Project Structure

```
NST_code/
├── app.py                 # Flask web application
├── train.py              # Training script for the decoder
├── utils/
│   ├── models.py         # VGG encoder and decoder model definitions
│   └── utils.py          # Utility functions (AdaIN, transforms, etc.)
├── templates/            # HTML templates for Flask app
├── content_data/         # Directory for content images
├── style_data/           # Directory for style images
└── static/uploads/       # Directory for uploaded and generated images
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- Torchvision
- Flask
- Pillow (PIL)

### Setup

1. Clone or download the repository
2. Install required dependencies:
```bash
pip install torch torchvision flask flask-wtf flask-bootstrap werkzeug pillow
```

3. Download the pre-trained VGG model:
   - Download `vgg_normalised.pth` from the appropriate source and place it in the project root

4. Place pre-trained decoder model:
   - Download or train a decoder model and place it at the specified path

## Usage

### Web Application

Run the Flask application for interactive style transfer:

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

**Steps:**
1. Upload a content image (PNG, JPG, JPEG)
2. Upload a style image
3. Adjust the alpha value to control style transfer strength (0-1)
4. Click "Transfer Style" to generate the stylized image

### Training

Train your own decoder model using the training script:

```bash
python train.py \
    --content_dir /path/to/content/images \
    --style_dir /path/to/style/images \
    --vgg /path/to/vgg_normalised.pth \
    --experiment experiment_name \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4
```

**Training Parameters:**
- `--content_dir`: Path to content images directory
- `--style_dir`: Path to style images directory
- `--vgg`: Path to pre-trained VGG model
- `--experiment`: Name of the experiment (creates output directory)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--lr_decay`: Learning rate decay (default: 5e-5)
- `--epochs`: Number of epochs (default: 1)
- `--content_weight`: Weight for content loss (default: 1.0)
- `--style_weight`: Weight for style loss (default: 5.0)
- `--resume`: Resume training from checkpoint
- `--decoder_path`: Path to decoder checkpoint for resuming
- `--optimizer_path`: Path to optimizer checkpoint for resuming

## Model Components

### VGGEncoder
Extracts feature maps from images using a pre-trained VGG19 network (normalized).

### Decoder
Neural network that reconstructs images from feature maps. Trained end-to-end with the content and style losses.

### Adaptive Instance Normalization (AdaIN)
The core of the style transfer, aligns the mean and variance of content features with style features.

## Loss Functions

The model is trained using two main loss components:

1. **Content Loss**: Ensures the output preserves the content structure
   - MSE loss between output and AdaIN features

2. **Style Loss**: Ensures the output adopts the style characteristics
   - MSE loss on mean and standard deviation across feature layers

## Configuration

### App Configuration (app.py)
- Upload folder: `static/uploads`
- Allowed extensions: PNG, JPG, JPEG
- Image resize size: 512x512
- Alpha range: 0.0 to 1.0 (controls style transfer strength)

### Training Configuration (train.py)
- Default image sizes: 512x512
- Default batch size: 4
- Default learning rate: 1e-4
- Output saved in `experiment/` directory

## Output

### Training Output
The training script saves the following to the experiment directory:
- Decoder checkpoints: `decoder_*.pth`
- Optimizer checkpoints: `optimizer_*.pth`
- Sample outputs: `output_*.png`
- Training arguments: `args.txt`

### Web Application Output
Generated stylized images are saved to `static/uploads/` directory with the prefix `stylized_`

## System Requirements

- NVIDIA GPU (optional but recommended) - with CUDA support for faster processing
- Minimum 4GB RAM
- 2GB disk space for models and data

## Troubleshooting

- **Out of Memory**: Reduce batch size or image size
- **Slow Processing**: Ensure GPU is being used (check device in code)
- **File Upload Issues**: Verify file permissions in `static/uploads/` directory
- **Model Not Found**: Ensure all pre-trained models are in the correct paths

## References

This implementation is based on the paper:
- Huang et al., "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" (2017)

## License

This project is provided as-is for educational and research purposes.

## Notes

- Adjust the alpha parameter to balance between content and style
- Larger alpha values (closer to 1) produce more stylized outputs
- Smaller alpha values (closer to 0) preserve more of the original content
- Image quality depends on input image size and model quality
