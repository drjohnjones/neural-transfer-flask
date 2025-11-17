"""
Neural Style Transfer using VGG19
Based on "A Neural Algorithm of Artistic Style" paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple
import os
from pathlib import Path


class StyleTransfer:
    """
    Neural Style Transfer using pre-trained VGG19
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        image_size: int = 512
    ):
        """
        Initialize style transfer
        
        Args:
            device: Device to use ('cuda', 'mps', 'cpu')
            image_size: Target image size
        """
        self.device = self._get_device(device)
        self.image_size = image_size
        
        print(f"🔧 Initializing Neural Style Transfer...")
        print(f"   Device: {self.device}")
        print(f"   Image size: {image_size}")
        
        # Load VGG19 model
        self.model = None
        self._load_model()
        
        # Define layers for style and content
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    
    def _get_device(self, device: Optional[str] = None) -> str:
        """Determine the best available device"""
        if device:
            return device
        
        # Check for CUDA
        if torch.cuda.is_available():
            return 'cuda'
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        
        return 'cpu'
    
    def _load_model(self):
        """Load pre-trained VGG19 model"""
        try:
            print("\n📥 Loading VGG19 model...")
            
            # Load pre-trained VGG19
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
            
            # Freeze all parameters
            for param in vgg.parameters():
                param.requires_grad_(False)
            
            # Move to device
            self.model = vgg.to(self.device).eval()
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        
        # Resize maintaining aspect ratio
        if max(image.size) > self.image_size:
            scale = self.image_size / max(image.size)
            new_size = tuple(int(dim * scale) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Transform to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def get_features(self, image: torch.Tensor) -> dict:
        """
        Extract features from different layers
        
        Args:
            image: Input image tensor
            
        Returns:
            Dictionary of layer features
        """
        features = {}
        x = image
        
        # Layer names in VGG19
        layer_names = {
            '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
            '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
        }
        
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layer_names:
                features[layer_names[name]] = x
        
        return features
    
    def gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate Gram matrix for style representation
        
        Args:
            tensor: Feature tensor
            
        Returns:
            Gram matrix
        """
        batch, channels, height, width = tensor.size()
        tensor = tensor.view(batch * channels, height * width)
        gram = torch.mm(tensor, tensor.t())
        return gram
    
    def transfer_style(
        self,
        content_path: str,
        style_path: str,
        output_path: Optional[str] = None,
        num_steps: int = 300,
        style_weight: float = 1e6,
        content_weight: float = 1,
        learning_rate: float = 0.003,
        show_every: int = 50
    ) -> str:
        """
        Apply style transfer
        
        Args:
            content_path: Path to content image
            style_path: Path to style image
            output_path: Where to save result
            num_steps: Number of optimization steps
            style_weight: Weight for style loss
            content_weight: Weight for content loss
            learning_rate: Learning rate for optimizer
            show_every: Show progress every N steps
            
        Returns:
            Path to output image
        """
        print(f"\n🎨 Starting style transfer...")
        print(f"   Content: {os.path.basename(content_path)}")
        print(f"   Style: {os.path.basename(style_path)}")
        print(f"   Steps: {num_steps}")
        
        # Load images
        content = self.load_image(content_path)
        style = self.load_image(style_path)
        
        # Get features
        content_features = self.get_features(content)
        style_features = self.get_features(style)
        
        # Calculate style gram matrices
        style_grams = {
            layer: self.gram_matrix(style_features[layer])
            for layer in self.style_layers
        }
        
        # Initialize target image (start from content)
        target = content.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = optim.Adam([target], lr=learning_rate)
        
        # Optimization loop
        for step in range(1, num_steps + 1):
            # Get features
            target_features = self.get_features(target)
            
            # Content loss
            content_loss = torch.mean(
                (target_features['conv4_2'] - content_features['conv4_2']) ** 2
            )
            
            # Style loss
            style_loss = 0
            for layer in self.style_layers:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                style_gram = style_grams[layer]
                
                layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
                
                # Weight by layer size
                _, d, h, w = target_feature.shape
                layer_style_loss /= (d * h * w)
                style_loss += layer_style_loss
            
            style_loss /= len(self.style_layers)
            
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Show progress
            if step % show_every == 0:
                print(f"   Step {step}/{num_steps} | "
                      f"Total: {total_loss.item():.2f} | "
                      f"Content: {content_loss.item():.2f} | "
                      f"Style: {style_loss.item():.2f}")
        
        print("✅ Style transfer complete!")
        
        # Save result
        if output_path is None:
            output_dir = Path("data/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"styled_{int(torch.rand(1).item() * 10000)}.jpg")
        
        self._save_image(target, output_path)
        print(f"💾 Saved: {output_path}")
        
        return output_path
    
    def _save_image(self, tensor: torch.Tensor, filepath: str):
        """Save tensor as image"""
        # Denormalize
        image = tensor.cpu().clone().detach().squeeze(0)
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image = torch.clamp(image, 0, 1)
        
        # Convert to PIL
        image = transforms.ToPILImage()(image)
        
        # Save
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        image.save(filepath, quality=95)
    
    def is_available(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None


if __name__ == '__main__':
    # Demo
    print("\n🎨 Neural Style Transfer Demo")
    print("="*60)
    
    # Initialize
    style_transfer = StyleTransfer(image_size=512)
    
    if style_transfer.is_available():
        print("\n✅ Style transfer ready!")
        print("\nPlace your images in:")
        print("   - data/content/ (photos to stylize)")
        print("   - data/style/ (artistic styles)")
    else:
        print("\n⚠️  Style transfer not available")
