#!/usr/bin/env python
"""
NST Project Setup Script
- Downloads VGG model
- Trains decoder or creates dummy model
- Creates necessary directories
"""

import os
import torch
from pathlib import Path
import urllib.request

def download_vgg_model():
    """Download VGG normalized model"""
    print("\n" + "="*60)
    print("STEP 1: Downloading VGG19 Normalized Model")
    print("="*60)
    
    vgg_path = Path('vgg_normalised.pth')
    
    if vgg_path.exists():
        size_bytes = vgg_path.stat().st_size
        size = size_bytes / (1024**2)
        if size_bytes == 80109653:
            print("[WARNING] Detected mock VGG model weights (from create_vgg.py), deleting to download real pre-trained weights...")
            vgg_path.unlink()
        elif size > 50:
            print(f"[OK] VGG model already exists ({size:.1f} MB)")
            return True
        else:
            print(f"[WARNING] Existing file too small ({size:.1f} MB), redownloading...")
            vgg_path.unlink()
    
    try:
        print("Downloading from GitHub Releases...")
        url = 'https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth'
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(vgg_path, 'wb') as f:
                f.write(response.read())
        size = vgg_path.stat().st_size / (1024**2)
        print(f"[OK] Downloaded VGG model ({size:.1f} MB)")
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n" + "="*60)
    print("STEP 2: Creating Directories")
    print("="*60)
    
    dirs = [
        'static/uploads',
        'content_data',
        'style_data',
        'experiment',
        'examples',
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created/verified: {dir_path}")

def download_decoder_model():
    """Download pretrained Decoder model"""
    print("\n" + "="*60)
    print("STEP 3: Downloading Pre-trained Decoder Model")
    print("="*60)
    
    # Use decoder.pth matching the URL filename
    decoder_path = Path('decoder.pth')
    
    if decoder_path.exists():
        size_bytes = decoder_path.stat().st_size
        size = size_bytes / (1024**2)
        if size_bytes == 14027957:
            print("[WARNING] Detected mock/dummy decoder model, deleting to download real pre-trained decoder...")
            decoder_path.unlink()
        elif size > 5:
            print(f"[OK] Decoder model already exists ({size:.1f} MB)")
            return True
            
    try:
        print("Downloading from GitHub Releases...")
        url = 'https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth'
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(decoder_path, 'wb') as f:
                f.write(response.read())
        size = decoder_path.stat().st_size / (1024**2)
        print(f"[OK] Downloaded Decoder model ({size:.1f} MB)")
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False

def verify_setup():
    """Verify all components are in place"""
    print("\n" + "="*60)
    print("STEP 4: Verifying Setup")
    print("="*60)
    
    checks = {
        'vgg_normalised.pth': 'VGG Model',
        'decoder.pth': 'Decoder Model',
        'static/uploads': 'Upload Directory',
        'content_data': 'Content Data Directory',
        'style_data': 'Style Data Directory',
        'utils/models.py': 'Models Module',
        'utils/utils.py': 'Utils Module',
        'app.py': 'Flask App',
        'train.py': 'Training Script',
        'templates/index.html': 'HTML Template'
    }
    
    all_ok = True
    for path, name in checks.items():
        if Path(path).exists():
            print(f"[OK] {name:25} ({path})")
        else:
            print(f"[MISSING] {name:25} (MISSING: {path})")
            all_ok = False
    
    return all_ok

def main():
    print("\n")
    print("+" + "="*58 + "+")
    print("|" + " "*15 + "Neural Style Transfer Setup" + " "*15 + "|")
    print("+" + "="*58 + "+")
    
    # Step 1: Download VGG
    if not download_vgg_model():
        print("[WARNING] Could not download VGG. You'll need to download manually.")
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Download decoder
    if not download_decoder_model():
        print("[WARNING] Could not download decoder model.")
    
    # Step 4: Verify
    if verify_setup():
        print("\n" + "="*60)
        print("[OK] SETUP COMPLETE!")
        print("="*60)
        print("\nYou can now run:")
        print("  python app.py")
        print("\nOr train a decoder:")
        print("  python train.py --content_dir content_data --style_dir style_data")
        print("="*60 + "\n")
    else:
        print("\n[WARNING] Setup incomplete. Check the errors above.")

if __name__ == '__main__':
    main()
