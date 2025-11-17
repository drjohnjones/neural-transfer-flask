import gradio as gr
from src.style_transfer_engine import get_engine

# Load engine
engine = get_engine()

def process_images(content_img, style_img, style_strength, steps):
    """Process style transfer"""
    if content_img is None or style_img is None:
        return None
    
    strength_map = {
        "Light": 1e6,
        "Medium": 1e7,
        "Strong": 1e8,
        "Very Strong": 5e8,
        "Extreme": 1e9
    }
    
    result = engine.transfer_style(
        content_img,
        style_img,
        int(steps),
        strength_map[style_strength]
    )
    
    return result

# Simple interface - avoids Gradio Blocks bug
demo = gr.Interface(
    fn=process_images,
    inputs=[
        gr.Image(type="numpy", label="📸 Your Photo"),
        gr.Image(type="numpy", label="🎨 Art Style"),
        gr.Radio(
            ["Light", "Medium", "Strong", "Very Strong", "Extreme"],
            value="Strong",
            label="Style Strength"
        ),
        gr.Slider(100, 500, value=200, step=50, label="Quality")
    ],
    outputs=gr.Image(type="pil", label="✨ Result"),
    title="🎨 Neural Style Transfer",
    description="Transform your photos into artistic masterpieces using deep learning!",
    article="""
    ### 💡 Tips
    - Use clear, well-lit photos for best results
    - Try famous artworks as style (Van Gogh, Monet, Kandinsky)
    - Processing takes 2-3 minutes
    - Your photo's aspect ratio is preserved!
    
    **Created by Dr. John Jones** | Project #21 of 52 AI/ML Projects Challenge
    """
)

if __name__ == "__main__":
    demo.launch()