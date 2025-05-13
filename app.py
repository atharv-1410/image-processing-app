from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from skimage.filters import prewitt_h, prewitt_v
from rembg import remove
from PIL import Image, ExifTags
from flask_cors import CORS
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19, decode_predictions as decode_predictions_vgg19

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


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
    
    img = cv2.imread(file_path)
    
    if action == 'background_removal':
        input_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_image = remove(input_image_rgb)
        output_pil_image = Image.fromarray(output_image)
        output_pil_image = output_pil_image.convert("RGB")
        output_pil_image.save(output_path, format="JPEG")
    
    elif action == 'edge_detection':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges_h = prewitt_h(gray)
        edges_v = prewitt_v(gray)
        edges_combined = edges_h + edges_v

    # Save to static/results/
        output_file = f"processed_{filename}"
        output_p = os.path.join("static", "results", output_file)
        cv2.imwrite(output_p, (edges_combined * 255).astype(np.uint8))



    elif action == 'histogram':
        import matplotlib.pyplot as plt
        colors = ('b', 'g', 'r')  # Blue, Green, Red
        plt.figure()
        plt.title("Color Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])  # Compute histogram for each channel
            plt.plot(hist, color=color)  # Plot histogram with respective color
    
        plt.savefig(output_path)
        plt.close()
    
    elif action == 'classify_vgg16':
        predictions = classify_vgg16(file_path)
        return render_template('index.html', filename=filename, predictions=predictions)
    
    elif action == 'classify_vgg19':
        predictions = classify_vgg19(file_path)
        return render_template('index.html', filename=filename, predictions=predictions)
    
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
    vgg_similarity = compare_vgg(img1_path, img2_path)

    return render_template('compare.html', pixel_similarity=pixel_similarity, metadata_similarity=metadata_similarity, vgg_similarity=vgg_similarity)

def classify_vgg16(image_path):
    model_vgg16 = VGG16(weights='imagenet')
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model_vgg16.predict(img_array)
    return decode_predictions(predictions, top=3)[0]

def classify_vgg19(image_path):
    model_vgg19 = VGG19(weights='imagenet')
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input_vgg19(img_array)
    predictions = model_vgg19.predict(img_array)
    return decode_predictions_vgg19(predictions, top=3)[0]

def extract_metadata(image_path):
    try:
        img = Image.open(image_path)
        metadata = {}

        # Check if EXIF data exists
        exif_data = img._getexif() if hasattr(img, '_getexif') else None

        if exif_data:
            metadata = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items()}
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
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1.shape != img2.shape:
        return "Images are different in pixel dimensions."
    difference = np.sum(cv2.absdiff(img1, img2))
    return "All pixels matched" if difference == 0 else "Images are different"


def compare_metadata(img1_path, img2_path):
    meta1 = extract_metadata(img1_path) 
    meta2 = extract_metadata(img2_path)
    return "Metadata matches" if meta1 == meta2 else "Metadata differs"

def compare_vgg(img1_path, img2_path):
    predictions1 = classify_vgg19(img1_path)
    predictions2 = classify_vgg19(img2_path)
    return "Same classification" if predictions1[0][1] == predictions2[0][1] else "Different classifications"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT env variable provided by Render
    app.run(host='0.0.0.0', port=port)
