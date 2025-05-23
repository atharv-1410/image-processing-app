from flask import Flask, render_template, request, jsonify
import os
from PIL import Image, ExifTags
from flask_cors import CORS
import urllib.request
import json
import requests
import base64
from io import BytesIO
import math

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Use lightweight Hugging Face API for image classification
HF_API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "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")  # Set this in your environment variables

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare')
def compare_page():
    return render_template('compare.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return render_template('index.html', filename=file.filename)

@app.route('/process', methods=['POST'])
def process():
    filename = request.form.get('filename')
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found"
    
    action = request.form.get('action')
    output_path = os.path.join(RESULT_FOLDER, f'processed_{filename}')
    
    if action == 'background_removal':
        try:
            # Using Remove.bg API for better background removal
            removebg_api_key = os.environ.get("REMOVEBG_API_KEY", "")
            if not removebg_api_key:
                return render_template('index.html', filename=filename, 
                                      error="Background removal requires REMOVEBG_API_KEY environment variable")
            
            with open(file_path, 'rb') as img_file:
                img_data = img_file.read()
            
            # API request to remove.bg
            response = requests.post(
                'https://api.remove.bg/v1.0/removebg',
                data={
                    'size': 'auto',
                    'format': 'auto'
                },
                files={'image_file': img_data},
                headers={'X-Api-Key': removebg_api_key},
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as out:
                    out.write(response.content)
            else:
                return render_template('index.html', filename=filename, 
                                      error=f"Background removal API error: {response.status_code}")
        except Exception as e:
            return render_template('index.html', filename=filename, 
                                  error=f"Background removal failed: {str(e)}")
    
    elif action == 'edge_detection':
        try:
            # Simple edge detection using PIL
            from PIL import ImageFilter
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            img = img.filter(ImageFilter.FIND_EDGES)
            img.save(output_path)
        except Exception as e:
            return render_template('index.html', filename=filename, error=f"Edge detection failed: {str(e)}")

    elif action == 'histogram':
        try:
            # Calculate histogram without numpy
            img = Image.open(file_path)
            r, g, b = img.convert('RGB').split()
            # Get histograms
            r_hist = r.histogram()
            g_hist = g.histogram()
            b_hist = b.histogram()
            
            # Save histogram data as JSON for client-side rendering
            hist_data = {
                'r': r_hist,
                'g': g_hist,
                'b': b_hist
            }
            json_path = os.path.join(RESULT_FOLDER, f'histogram_{filename}.json')
            with open(json_path, 'w') as f:
                json.dump(hist_data, f)
            
            # Return JSON path for client-side rendering
            return render_template('index.html', filename=filename, histogram_data=os.path.basename(json_path))
        except Exception as e:
            return render_template('index.html', filename=filename, error=f"Histogram generation failed: {str(e)}")
    
    elif action == 'classify':
        try:
            # Use Hugging Face API for classification
            predictions = classify_image_api(file_path)
            return render_template('index.html', filename=filename, predictions=predictions)
        except Exception as e:
            return render_template('index.html', filename=filename, error=f"Classification failed: {str(e)}")
    
    elif action == 'metadata':
        metadata = extract_metadata(file_path)
        return render_template('index.html', filename=filename, metadata=metadata)
    
    return render_template('index.html', filename=filename, output_file=os.path.basename(output_path))

@app.route('/compare_images', methods=['POST'])
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return "Please upload both images."
    
    img1 = request.files['image1']
    img2 = request.files['image2']
    
    if img1.filename == '' or img2.filename == '':
        return "Both images must be selected."
    
    img1_path = os.path.join(UPLOAD_FOLDER, img1.filename)
    img2_path = os.path.join(UPLOAD_FOLDER, img2.filename)
    img1.save(img1_path)
    img2.save(img2_path)

    pixel_similarity = compare_pixels(img1_path, img2_path)
    metadata_similarity = compare_metadata(img1_path, img2_path)
    
    try:
        # Simplified comparison using perceptual hash
        visual_similarity = compare_visual(img1_path, img2_path)
    except Exception as e:
        visual_similarity = f"Visual comparison unavailable: {str(e)}"

    return render_template('compare.html', 
                          image1=img1.filename,
                          image2=img2.filename,
                          pixel_similarity=pixel_similarity, 
                          metadata_similarity=metadata_similarity, 
                          visual_similarity=visual_similarity)

def classify_image_api(image_path):
    """Use Hugging Face API instead of loading models locally"""
    if not HF_API_KEY:
        return [("API_KEY_MISSING", "Please set HUGGINGFACE_API_KEY environment variable", 1.0)]
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }
        
        req = urllib.request.Request(
            HF_API_URL,
            data=image_bytes,
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            # Format like VGG output for compatibility
            return [(item["label"], item["label"], item["score"]) for item in result[:3]]
    except Exception as e:
        return [("ERROR", str(e), 0.0)]

def extract_metadata(image_path):
    try:
        img = Image.open(image_path)
        metadata = {}

        # Check if EXIF data exists
        exif_data = img._getexif() if hasattr(img, '_getexif') else None

        if exif_data:
            metadata = {ExifTags.TAGS.get(tag, tag): str(value) for tag, value in exif_data.items() 
                       if tag in ExifTags.TAGS}  # Filter to known tags and stringify values
        else:
            metadata["Info"] = "No EXIF metadata found"

        # Include format and mode
        metadata["Format"] = img.format
        metadata["Mode"] = img.mode
        metadata["Size"] = f"{img.width}x{img.height} pixels"

        return metadata

    except Exception as e:
        return {"Error": str(e)}

def compare_pixels(img1_path, img2_path):
    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Check dimensions
        if img1.size != img2.size:
            return "Images have different dimensions"
        
        # Convert to same mode for comparison
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')
        
        # Simple pixel comparison without numpy
        width, height = img1.size
        total_pixels = width * height * 3  # RGB channels
        diff_count = 0
        
        # Sample pixels (checking every 10th pixel to avoid memory issues)
        for x in range(0, width, 10):
            for y in range(0, height, 10):
                r1, g1, b1 = img1.getpixel((x, y))
                r2, g2, b2 = img2.getpixel((x, y))
                diff_count += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
        
        # Scale up the difference count since we sampled
        diff_count = diff_count * 100 / (width * height / 10 / 10 * 3)
        
        # Convert to similarity percentage (max possible difference is 255*3 per pixel)
        similarity = 100 - (diff_count / (255 * 3) * 100)
        return f"Pixel similarity: {similarity:.2f}%"
    except Exception as e:
        return f"Comparison error: {str(e)}"

def compare_metadata(img1_path, img2_path):
    meta1 = extract_metadata(img1_path)
    meta2 = extract_metadata(img2_path)
    
    # Count matching keys
    common_keys = set(meta1.keys()) & set(meta2.keys())
    matches = sum(1 for k in common_keys if meta1[k] == meta2[k])
    
    if len(common_keys) == 0:
        return "No common metadata found"
    
    similarity = (matches / len(common_keys)) * 100
    return f"Metadata similarity: {similarity:.2f}%"

def compare_visual(img1_path, img2_path):
    """Compare images using a perceptual hash"""
    try:
        def dhash(image, hash_size=8):
            # Convert to grayscale and resize
            image = image.convert('L').resize((hash_size + 1, hash_size))
            pixels = []
            
            # Calculate pixel values
            for row in range(hash_size):
                for col in range(hash_size):
                    pixels.append(image.getpixel((col, row)))
            
            # Calculate differences between adjacent pixels
            difference = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = image.getpixel((col, row))
                    pixel_right = image.getpixel((col + 1, row))
                    difference.append(pixel_left > pixel_right)
            
            # Convert to a 64-bit integer
            decimal_value = 0
            for index, value in enumerate(difference):
                if value:
                    decimal_value += 2 ** index
            
            # Convert to hex string
            hex_value = hex(decimal_value)[2:].zfill(16)
            return hex_value
        
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        hash1 = dhash(img1)
        hash2 = dhash(img2)
        
        # Calculate Hamming distance (number of different bits)
        distance = 0
        for i in range(len(hash1)):
            b1 = bin(int(hash1[i], 16))[2:].zfill(4)
            b2 = bin(int(hash2[i], 16))[2:].zfill(4)
            for j in range(len(b1)):
                if b1[j] != b2[j]:
                    distance += 1
        
        # Convert to similarity percentage (64 bits total in our hash)
        similarity = 100 - (distance / 64) * 100
        return f"Visual similarity: {similarity:.2f}%"
    except Exception as e:
        return f"Visual comparison error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False in production
