"""
Neural Style Transfer using VGG19 - VERIFIED WORKING VERSION
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Optional
import os
from pathlib import Path
import copy


class StyleTransfer:
    """Neural Style Transfer using VGG19"""
    
    def __init__(self, device: Optional[str] = None, image_size: int = 512):
        self.device = self._get_device(device)
        self.image_size = image_size
        
        print(f"🔧 Initializing Neural Style Transfer...")
        print(f"   Device: {self.device}")
        print(f"   Image size: {image_size}")
        
        self.model = None
        self._load_model()
        
        # Style and content layers
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    def _get_device(self, device: Optional[str] = None) -> str:
        if device:
            return device
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _load_model(self):
        """Load VGG19"""
        print("\n📥 Loading VGG19 model...")
        
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Build model with named layers
        self.model = nn.Sequential()
        
        i = 0
        conv_i = 0
        
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                conv_i += 1
                name = f'conv_{conv_i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{conv_i}'
                layer = nn.ReLU(inplace=False)  # Important!
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{conv_i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{conv_i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            
            self.model.add_module(name, layer)
        
        self.model = self.model.to(self.device)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        print("✅ Model loaded!")
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        
        # Resize
        if max(image.size) > self.image_size:
            scale = self.image_size / max(image.size)
            new_size = tuple(int(dim * scale) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def get_features(self, image: torch.Tensor, layers: list) -> dict:
        """Extract features from specified layers"""
        features = {}
        x = image
        
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in layers:
                features[name] = x
        
        return features
    
    def gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix"""
        b, c, h, w = tensor.size()
        features = tensor.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(b * c * h * w)
    
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
        """Apply style transfer"""
        
        print(f"\n🎨 Starting style transfer...")
        print(f"   Content: {os.path.basename(content_path)}")
        print(f"   Style: {os.path.basename(style_path)}")
        print(f"   Steps: {num_steps}")
        print(f"   Style weight: {style_weight}")
        print(f"   Content weight: {content_weight}")
        
        # Load images
        content = self.load_image(content_path)
        style = self.load_image(style_path)
        
        # Get content features
        content_features = self.get_features(content, self.content_layers)
        
        # Get style features
        style_features = self.get_features(style, self.style_layers)
        
        # Calculate style grams
        style_grams = {
            layer: self.gram_matrix(style_features[layer])
            for layer in self.style_layers
        }
        
        # Initialize target (start from content)
        target = content.clone().requires_grad_(True).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam([target], lr=learning_rate)
        
        print("\n🔄 Optimizing...")
        
        # Optimization loop
        for step in range(1, num_steps + 1):
            # Get target features
            target_features = self.get_features(target, self.content_layers + self.style_layers)
            
            # Content loss
            content_loss = 0
            for layer in self.content_layers:
                content_loss += torch.mean(
                    (target_features[layer] - content_features[layer]) ** 2
                )
            content_loss *= content_weight
            
            # Style loss
            style_loss = 0
            for layer in self.style_layers:
                target_gram = self.gram_matrix(target_features[layer])
                style_gram = style_grams[layer]
                style_loss += torch.mean((target_gram - style_gram) ** 2)
            style_loss *= style_weight
            
            # Total loss
            total_loss = content_loss + style_loss
            
            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Progress
            if step % show_every == 0 or step == 1:
                print(f"   Step {step:4d}/{num_steps} | "
                      f"Total: {total_loss.item():12.2f} | "
                      f"Content: {content_loss.item():10.2f} | "
                      f"Style: {style_loss.item():12.2f}")
        
        print("✅ Optimization complete!")
        
        # Save
        if output_path is None:
            output_dir = Path("data/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"styled_{int(torch.rand(1).item() * 10000)}.jpg")
        
        self._save_image(target, output_path)
        print(f"💾 Saved: {output_path}")
        
        return output_path
    
    def _save_image(self, tensor: torch.Tensor, filepath: str):
        """Save tensor as image"""
        image = tensor.cpu().clone().detach().squeeze(0)
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Convert to PIL
        image = transforms.ToPILImage()(image)
        
        # Save
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        image.save(filepath, quality=95)
    
    def is_available(self) -> bool:
        return self.model is not None


if __name__ == '__main__':
    print("\n🎨 Neural Style Transfer Engine Ready")
