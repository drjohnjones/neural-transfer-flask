"""
Shared Neural Style Transfer Engine
Used by both Flask and Gradio versions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np


class StyleTransferEngine:
    """Shared style transfer logic"""
    
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # Load VGG19
        print("📥 Loading VGG19...")
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = self.vgg.to(self.device).eval()
        print("✅ VGG19 loaded!")
        
        # Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
    
    def image_loader(self, image, target_size=None, max_size=512):
        """
        Load and preprocess image
        
        Args:
            image: PIL Image, numpy array, or file path
            target_size: Tuple (height, width) to resize to, or None for auto
            max_size: Maximum dimension if target_size is None
        
        Returns:
            Tensor of preprocessed image
        """
        # Convert to PIL if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        if target_size is not None:
            # Resize to exact target size (height, width)
            loader = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])
        else:
            # Auto-resize maintaining aspect ratio
            orig_width, orig_height = image.size
            
            if orig_width > orig_height:
                new_width = max_size
                new_height = int(max_size * orig_height / orig_width)
            else:
                new_height = max_size
                new_width = int(max_size * orig_width / orig_height)
            
            loader = transforms.Compose([
                transforms.Resize((new_height, new_width)),
                transforms.ToTensor()
            ])
        
        image_tensor = loader(image).unsqueeze(0)
        return image_tensor.to(self.device, torch.float)
    
    @staticmethod
    def gram_matrix(input):
        """Calculate gram matrix"""
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
    
    def get_style_model_and_losses(self, style_img, content_img):
        """Build model with losses"""
        
        class ContentLoss(nn.Module):
            def __init__(self, target):
                super(ContentLoss, self).__init__()
                self.target = target.detach()
            
            def forward(self, input):
                self.loss = F.mse_loss(input, self.target)
                return input
        
        class StyleLoss(nn.Module):
            def __init__(self, target_feature):
                super(StyleLoss, self).__init__()
                self.target = StyleTransferEngine.gram_matrix(target_feature).detach()
            
            def forward(self, input):
                G = StyleTransferEngine.gram_matrix(input)
                self.loss = F.mse_loss(G, self.target)
                return input
        
        class Normalization(nn.Module):
            def __init__(self, mean, std, device):
                super(Normalization, self).__init__()
                self.mean = mean.clone().detach().view(-1, 1, 1)
                self.std = std.clone().detach().view(-1, 1, 1)
            
            def forward(self, img):
                return (img - self.mean) / self.std
        
        normalization = Normalization(self.mean, self.std, self.device).to(self.device)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)
        
        i = 0
        for layer in self.vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError('Unrecognized layer')
            
            model.add_module(name, layer)
            
            if name == 'conv_4':
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)
            
            if name in ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)
        
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        
        model = model[:(i + 1)]
        return model, style_losses, content_losses
    
    def transfer_style(self, content_img, style_img, num_steps=200, 
                      style_weight=1e8, content_weight=1, max_size=512):
        """
        Apply style transfer - output matches content image size
        
        Args:
            content_img: PIL Image, numpy array, or file path
            style_img: PIL Image, numpy array, or file path
            num_steps: Number of optimization steps
            style_weight: Weight for style loss
            content_weight: Weight for content loss
            max_size: Maximum dimension for processing
        
        Returns:
            PIL Image with same size as content image
        """
        print(f"🎨 Starting style transfer ({num_steps} steps)...")
        
        # Get original content size
        if isinstance(content_img, str):
            orig_content = Image.open(content_img).convert('RGB')
        elif isinstance(content_img, np.ndarray):
            orig_content = Image.fromarray(content_img).convert('RGB')
        else:
            orig_content = content_img
        
        original_size = orig_content.size  # (width, height)
        print(f"📐 Original content size: {original_size[0]}x{original_size[1]}")
        
        # Load content with aspect ratio preserved
        content = self.image_loader(content_img, target_size=None, max_size=max_size)
        
        # Get content tensor size (batch, channels, height, width)
        content_h = content.shape[2]
        content_w = content.shape[3]
        print(f"📐 Processing at: {content_w}x{content_h}")
        
        # Load style and resize to MATCH content processing size
        style = self.image_loader(style_img, target_size=(content_h, content_w))
        
        print(f"✅ Content tensor: {content.shape}")
        print(f"✅ Style tensor: {style.shape}")
        
        # Build model
        model, style_losses, content_losses = self.get_style_model_and_losses(style, content)
        
        # Initialize
        input_img = content.clone()
        input_img.requires_grad_(True)
        model.requires_grad_(False)
        
        optimizer = optim.LBFGS([input_img])
        
        run = [0]
        
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            
            optimizer.zero_grad()
            model(input_img)
            
            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            
            loss = style_score + content_score
            loss.backward()
            
            run[0] += 1
            if run[0] % 50 == 0 or run[0] == 1:
                print(f"   Step {run[0]:4d}/{num_steps} | Loss: {loss.item():.2f}")
            
            return loss
        
        # Optimize
        while run[0] <= num_steps:
            optimizer.step(closure)
            if run[0] >= num_steps:
                break
        
        # Final clamp
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        # Convert to PIL
        output = input_img.cpu().clone().squeeze(0)
        output = transforms.ToPILImage()(output)
        
        # Resize back to original content dimensions
        output = output.resize(original_size, Image.Resampling.LANCZOS)
        print(f"✅ Output resized to original: {original_size[0]}x{original_size[1]}")
        
        print("✅ Style transfer complete!")
        return output


# Global instance (loaded once)
_engine = None

def get_engine():
    """Get or create global engine instance"""
    global _engine
    if _engine is None:
        _engine = StyleTransferEngine()
    return _engine
