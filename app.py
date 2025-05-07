import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
import faiss
import spacy

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load pre-trained model and data
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
nlp = spacy.load('en_core_web_sm')
df_valid = pd.read_csv('fashion_valid.csv')
embeddings = np.load('embeddings.npy')
image_base_dir = os.path.join('dataset', 'data')

# Set up FAISS index
d = embeddings.shape[1]
indexx = faiss.IndexFlatL2(d)
indexx.add(embeddings)

# Function to extract embedding
def extract_embedding(image_path):
    try:
        img = Image.open(image_path).resize((224, 224)).convert('RGB')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        embedding = model.predict(img_array, verbose=0)
        return embedding.flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Helper function to parse and match the query
def extract_attributes(text_query):
    doc = nlp(text_query.lower())
    attributes = {
        "color": None,
        "material": None,
        "product_type": None,
        "gender": None,
        "category": None,
        "sub_category": None
    }

    # Define possible colors, product types, genders, and categories
    colors = ["white", "black", "blue", "pink", "yellow", "green", "red", "olive", "brown", "purple", "grey"]
    product_types = ["shirt", "top", "capris", "pant", "footwear", "jeans", "suits"]
    genders = ["boys", "girls", "men", "women"]
    categories = ["apparel", "footwear"]
    sub_categories = ["topwear", "bottomwear", "footwear", "dresses", "jeans"]

    # Detect colors in the text
    for color in colors:
        if color in text_query.lower():
            attributes["color"] = color

    # Detect product types
    for p_type in product_types:
        if p_type in text_query.lower():
            attributes["product_type"] = p_type

    # Detect gender
    for gender in genders:
        if gender in text_query.lower():
            attributes["gender"] = gender.capitalize()

    # Detect category
    for category in categories:
        if category in text_query.lower():
            attributes["category"] = category

    # Detect sub-category
    for sub_cat in sub_categories:
        if sub_cat in text_query.lower():
            attributes["sub_category"] = sub_cat

    return attributes

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload
        if 'image' not in request.files:
            return render_template('index.html', error="No image uploaded")
        image_file = request.files['image']
        if image_file.filename == '':
            return render_template('index.html', error="No image selected")
        
        # Save uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'query.jpg')
        image_file.save(image_path)

        # Get text query (if provided)
        text_query = request.form.get('text_query', '')

        # Process image query
        query_embedding = extract_embedding(image_path)
        if query_embedding is None:
            return render_template('index.html', error="Failed to process image")

        # Search for similar images
        distances, indices = indexx.search(np.array([query_embedding]), k=50)
        filtered_df = df_valid.iloc[indices[0]]
        print(filtered_df.head())

        if text_query:
            # Extract attributes from text query if present
            attributes = extract_attributes(text_query)
            print(f"🔍 Extracted attributes: {attributes}")

            # Filter images based on attributes
            results = []
            for idx, row in filtered_df.iterrows():
                score = 0
                # Check if the product matches the extracted attributes
                if attributes["color"] and attributes["color"].lower() in row['Colour'].lower(): score += 1
                if attributes["product_type"] and attributes["product_type"].lower() in row['ProductType'].lower(): score += 1
                if attributes["gender"] and attributes["gender"].capitalize() == row['Gender']: score += 1
                if attributes["category"] and attributes["category"].lower() == row['Category'].lower(): score += 1
                if attributes["sub_category"] and attributes["sub_category"].lower() in row['SubCategory'].lower(): score += 1
                if score > 0:
                    results.append({"ProductId": row['ProductId'], "score": score, "details": row.to_dict()})

            # Sort and get top 5
            results.sort(key=lambda x: x["score"], reverse=True)
            top_results = results[:min(5, len(results))]
            print(f"Top Results: {top_results}")
        else:
            # If no text query is provided, suggest top 5 based on image alone
            results = []
            for idx, row in filtered_df.iterrows():
                results.append({"ProductId": row['ProductId'], "score": 1, "details": row.to_dict()})
            results.sort(key=lambda x: x["score"], reverse=True)
            top_results = results[:min(5, len(results))]

        # Prepare result data
        result_data = []
        for result in top_results:
            subdir = f"{result['details']['Category']}/{result['details']['Gender']}/Images/images_with_product_ids"
            image_rel_path = os.path.join(subdir, result['details']['Image'])
            image_rel_path = image_rel_path.replace("\\", "/")  # Ensures URL-compatible paths
            image_full_path = os.path.join(image_base_dir, image_rel_path)
            print(f"Image relative path: {image_rel_path}")
            print(f"Full image path: {image_full_path}")

            if os.path.exists(image_full_path):
                result_data.append({
                    'ProductId': result['ProductId'],
                    'Title': result['details']['ProductTitle'],
                    'Score': result['score'],
                    'Image': image_rel_path  # Relative path for HTML
                })
        print(f"Final result_data: {result_data}")
        return render_template('index.html', query_image='uploads/query.jpg', results=result_data)
    
    return render_template('index.html')

# Serve static files (if needed)
# Serve images from the 'data' directory (subdirectories included)
@app.route('/dataset/data/<path:filename>')
def serve_dataset(filename):
    return send_from_directory(image_base_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
