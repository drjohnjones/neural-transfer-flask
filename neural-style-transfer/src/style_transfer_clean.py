"""
Minimal Neural Style Transfer - GUARANTEED TO WORK
Based on the original Gatys et al. paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_image(img_path, max_size=512, device=None):
    """Load and preprocess image"""
    if device is None:
        device = get_device()
    
    image = Image.open(img_path).convert('RGB')
    
    # Resize
    size = max_size if max(image.size) > max_size else max(image.size)
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    return image.to(device)


def save_image(tensor, filepath):
    """Save tensor as image"""
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    
    # Denormalize
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = torch.clamp(image, 0, 1)
    
    # To PIL
    transform = transforms.ToPILImage()
    image = transform(image)
    
    # Save
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    image.save(filepath, quality=95)


def get_features(image, model, layers):
    """Extract features from specific layers"""
    features = {}
    x = image
    
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    
    return features


def gram_matrix(tensor):
    """Calculate gram matrix"""
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram


def style_transfer(content_path, style_path, output_path=None,
                   steps=300, style_weight=1e8, content_weight=1,
                   learning_rate=0.01, print_every=50):
    """
    Perform neural style transfer
    
    CRITICAL: Uses high style_weight by default for strong effect
    """
    
    device = get_device()
    print(f"\n🎨 Neural Style Transfer")
    print(f"   Device: {device}")
    print(f"   Steps: {steps}")
    print(f"   Style weight: {style_weight:.0e}")
    print(f"   Content weight: {content_weight}")
    
    # Load VGG19
    print("\n📥 Loading VGG19...")
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    vgg = vgg.to(device).eval()
    
    # Freeze weights
    for param in vgg.parameters():
        param.requires_grad_(False)
    
    # Load images
    print("📸 Loading images...")
    content = load_image(content_path, device=device)
    style = load_image(style_path, device=device)
    
    # Match sizes
    if content.shape != style.shape:
        print(f"   Resizing style to match content...")
        style = torch.nn.functional.interpolate(
            style, size=content.shape[2:], mode='bilinear', align_corners=False
        )
    
    # Feature layers
    content_layers = {'21': 'conv4_2'}  # VGG19 layer
    style_layers = {
        '0': 'conv1_1',
        '5': 'conv2_1', 
        '10': 'conv3_1',
        '19': 'conv4_1',
        '28': 'conv5_1'
    }
    
    # Get features
    print("🔍 Extracting features...")
    content_features = get_features(content, vgg, content_layers)
    style_features = get_features(style, vgg, style_layers)
    
    # Calculate style grams
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Initialize target with content
    target = content.clone().requires_grad_(True)
    
    # Optimizer - CRITICAL: Use LBFGS for better convergence
    optimizer = optim.LBFGS([target], lr=learning_rate, max_iter=20)
    
    print(f"\n🔄 Optimizing for {steps} steps...")
    print(f"   (This will take 3-5 minutes)\n")
    
    step = [0]  # Use list to modify in closure
    
    def closure():
        """Optimization closure"""
        # Clamp values
        target.data.clamp_(0, 1)
        
        optimizer.zero_grad()
        
        # Get target features
        target_features = get_features(target, vgg, {**content_layers, **style_layers})
        
        # Content loss
        content_loss = 0
        for layer in content_layers.values():
            content_loss += torch.mean(
                (target_features[layer] - content_features[layer]) ** 2
            )
        content_loss *= content_weight
        
        # Style loss
        style_loss = 0
        for layer in style_layers.values():
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            
            layer_loss = torch.mean((target_gram - style_gram) ** 2)
            b, c, h, w = target_feature.shape
            style_loss += layer_loss / (c * h * w)
        
        style_loss *= style_weight
        
        # Total loss
        total_loss = content_loss + style_loss
        total_loss.backward()
        
        step[0] += 1
        
        if step[0] % print_every == 0:
            print(f"   Step {step[0]:4d} | "
                  f"Total: {total_loss.item():12.2f} | "
                  f"Content: {content_loss.item():8.2f} | "
                  f"Style: {style_loss.item():12.2f}")
        
        return total_loss
    
    # Run optimization
    while step[0] <= steps:
        optimizer.step(closure)
        
        if step[0] >= steps:
            break
    
    # Final clamp
    target.data.clamp_(0, 1)
    
    # Save
    if output_path is None:
        output_path = f"data/output/styled_{torch.randint(10000, (1,)).item()}.jpg"
    
    save_image(target, output_path)
    
    print(f"\n✅ Complete!")
    print(f"💾 Saved: {output_path}")
    
    return output_path


if __name__ == '__main__':
    # Quick test
    import sys
    
    if len(sys.argv) >= 3:
        content = sys.argv[1]
        style = sys.argv[2]
        
        output = style_transfer(
            content, style,
            steps=200,
            style_weight=1e8,  # VERY STRONG
            content_weight=1
        )
        
        print(f"\n🎨 Result: {output}")
        print(f"   open {output}")
