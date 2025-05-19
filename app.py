import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Configuration - Use absolute paths
base_dir = Path(__file__).parent.absolute()
app.config['UPLOAD_FOLDER'] = base_dir / 'uploads'
app.config['DATABASE_FOLDER'] = base_dir / 'database'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create required directories
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['DATABASE_FOLDER'].mkdir(exist_ok=True)

def init_database():
    database = {}
    db_path = app.config['DATABASE_FOLDER']
    
    for img_path in db_path.glob('*'):
        if img_path.suffix.lower()[1:] in app.config['ALLOWED_EXTENSIONS']:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Convert to grayscale and resize
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = resize(img_gray, (300, 300), preserve_range=True)  # Keep original range
            database[img_path.name] = img_resized
    return database
def calculate_ssim(query_img, db_img):
    try:
        # Ensure both images have the same dimensions
        query_resized = resize(query_img, db_img.shape)
        
        # Convert to uint8 with proper scaling
        query_uint8 = (query_resized * 255).astype(np.uint8)
        db_uint8 = (db_img * 255).astype(np.uint8)
        
        # Calculate SSIM with explicit data range
        score, _ = ssim(query_uint8, db_uint8, 
                       full=True,
                       data_range=255,  # For uint8 images
                       win_size=3,      # Smaller window for better performance
                       channel_axis=None)
        return max(0, round(score * 100, 2))
    except Exception as e:
        app.logger.error(f"SSIM calculation error: {str(e)}")
        return 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    database = init_database()
    if not database:
        return jsonify({'error': 'Empty database'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save and process query image
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        query_img = cv2.imread(filepath)
        if query_img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Preprocess query image
        query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        query_processed = resize(query_gray, (300, 300), preserve_range=True)

        results = []
        for filename, db_img in database.items():
            similarity = calculate_ssim(query_processed, db_img)
            results.append({
                'filename': filename,
                'similarity': similarity
            })

        # Sort results by similarity score
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return jsonify({'results': results[:10]})

    except Exception as e:
        app.logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/database-status')
def database_status():
    database = init_database()
    return jsonify({'count': len(database)})
if __name__ == '__main__':
    app.run(debug=True)