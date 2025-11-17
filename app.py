"""
Neural Style Transfer - Flask Version (Production)
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import uuid
from datetime import datetime
from PIL import Image
import re

# Import shared engine
from src.style_transfer_engine import get_engine

app = Flask(__name__)
CORS(app)  # Enable CORS
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'data/uploads'
STYLE_FOLDER = 'data/style'
OUTPUT_FOLDER = 'data/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure folders exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(STYLE_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Load engine once at startup
print("🔧 Initializing Neural Style Transfer Engine...")
engine = get_engine()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_filename(filename):
    """Extract clean name from filename"""
    name = Path(filename).stem
    name = re.sub(r'^[a-f0-9]{20,}_', '', name)
    name = re.sub(r'[^a-zA-Z0-9-]', '', name)
    name = name[:20]
    return name if name else 'image'


def create_output_filename(content_name, style_name, style_weight, num_steps):
    """Create descriptive output filename"""
    content_base = clean_filename(content_name)
    style_base = clean_filename(style_name)
    
    strength_map = {
        1e6: "Light",
        1e7: "Medium",
        1e8: "Strong",
        5e8: "VeryStrong",
        1e9: "Extreme"
    }
    
    strength = "Custom"
    for weight, name in strength_map.items():
        if abs(style_weight - weight) < weight * 0.1:
            strength = name
            break
    
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    
    filename = f"{content_base}_{style_base}_{strength}_Q{num_steps}_{date_str}_{time_str}.jpg"
    
    return filename


@app.route('/')
def index():
    """Main page"""
    styles = []
    if os.path.exists(STYLE_FOLDER):
        for file in os.listdir(STYLE_FOLDER):
            if allowed_file(file):
                styles.append({
                    'name': file,
                    'path': f'/style/{file}'
                })
    return render_template('index.html', styles=styles)


@app.route('/style/<filename>')
def get_style(filename):
    """Serve style image"""
    filepath = os.path.join(STYLE_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'Style not found'}), 404


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload content image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        img = Image.open(filepath)
        width, height = img.size
        
        return jsonify({
            'success': True,
            'filename': filename,
            'width': width,
            'height': height
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/transfer', methods=['POST'])
def transfer():
    """Apply style transfer"""
    try:
        data = request.get_json()
        
        content_filename = data.get('content')
        style_filename = data.get('style')
        num_steps = int(data.get('steps', 300))
        style_weight = float(data.get('style_weight', 1e6))
        content_weight = float(data.get('content_weight', 1))
        
        if not content_filename or not style_filename:
            return jsonify({'error': 'Missing content or style image'}), 400
        
        content_path = os.path.join(UPLOAD_FOLDER, content_filename)
        style_path = os.path.join(STYLE_FOLDER, style_filename)
        
        if not os.path.exists(content_path):
            return jsonify({'error': 'Content image not found'}), 404
        
        if not os.path.exists(style_path):
            return jsonify({'error': 'Style image not found'}), 404
        
        print(f"\n🎨 Processing style transfer...")
        print(f"   Content: {content_filename}")
        print(f"   Style: {style_filename}")
        print(f"   Steps: {num_steps}, Style Weight: {style_weight}")
        
        output_filename = create_output_filename(
            content_filename,
            style_filename,
            style_weight,
            num_steps
        )
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        print(f"   Output: {output_filename}")
        
        result = engine.transfer_style(
            content_img=content_path,
            style_img=style_path,
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight
        )
        
        result.save(output_path, quality=95)
        
        print(f"✅ Saved: {output_path}")
        
        return jsonify({
            'success': True,
            'output': output_filename
        })
        
    except Exception as e:
        print(f"❌ Transfer error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/output/<filename>')
def get_output(filename):
    """Serve output image"""
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'Output not found'}), 404


@app.route('/download/<filename>')
def download(filename):
    """Download output image"""
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'version': 'flask-production',
        'device': str(engine.device)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5090))
    
    print("\n" + "="*60)
    print("🎨 Neural Style Transfer - Production")
    print("="*60)
    print(f"\n📱 Running on port: {port}")
    print(f"🔍 Health check: /health")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
