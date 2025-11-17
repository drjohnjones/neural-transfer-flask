"""
Image processing utilities for style transfer
"""

from PIL import Image
import numpy as np
from typing import Tuple, List
import os


class ImageUtils:
    """Image processing utilities"""
    
    @staticmethod
    def resize_image(
        image_path: str,
        max_size: int = 512,
        output_path: str = None
    ) -> str:
        """
        Resize image maintaining aspect ratio
        
        Args:
            image_path: Input image path
            max_size: Maximum dimension
            output_path: Where to save
            
        Returns:
            Path to resized image
        """
        image = Image.open(image_path)
        
        # Calculate new size
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        
        # Resize
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save
        if output_path is None:
            output_path = image_path.replace('.', '_resized.')
        
        resized.save(output_path, quality=95)
        return output_path
    
    @staticmethod
    def create_comparison(
        content_path: str,
        style_path: str,
        output_path: str,
        save_path: str
    ) -> str:
        """
        Create side-by-side comparison
        
        Args:
            content_path: Original image
            style_path: Style reference
            output_path: Styled result
            save_path: Where to save comparison
            
        Returns:
            Path to comparison image
        """
        # Load images
        content = Image.open(content_path)
        style = Image.open(style_path)
        output = Image.open(output_path)
        
        # Resize to same height
        target_height = 400
        
        def resize_to_height(img, height):
            ratio = height / img.size[1]
            new_size = (int(img.size[0] * ratio), height)
            return img.resize(new_size, Image.Resampling.LANCZOS)
        
        content = resize_to_height(content, target_height)
        style = resize_to_height(style, target_height)
        output = resize_to_height(output, target_height)
        
        # Create comparison
        total_width = content.width + style.width + output.width + 40
        comparison = Image.new('RGB', (total_width, target_height + 60), 'white')
        
        # Paste images
        comparison.paste(content, (10, 40))
        comparison.paste(style, (content.width + 20, 40))
        comparison.paste(output, (content.width + style.width + 30, 40))
        
        # Add labels (you'd need PIL ImageDraw for text)
        comparison.save(save_path, quality=95)
        return save_path
    
    @staticmethod
    def list_images(directory: str, extensions: List[str] = None) -> List[str]:
        """
        List all images in directory
        
        Args:
            directory: Directory to search
            extensions: Image extensions to include
            
        Returns:
            List of image paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        images = []
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in extensions):
                images.append(os.path.join(directory, file))
        
        return sorted(images)


if __name__ == '__main__':
    print("✅ Image utils ready")
