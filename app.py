import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model (from .h5 file)
model = load_model('C:\\Users\\ARNAV\\Desktop\\Grapes-Leaf-Disease-detection-master2\\Grapes-Leaf-Disease-detection-master\\Coloured Model\\leaf_disease_coloured.h5')

# Define categories (disease names)
categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]

# Dictionary to store control methods and probable causes for each disease
control_methods = {
    "Black_rot": {
        "title": "To control black rot in grapes:",
        "probable_causes": "Black rot is caused by the fungus *Guignardia bidwellii*, often spread in warm, humid conditions.",
        "cultural": "Prune and destroy infected plant parts, improve airflow, and sanitize tools.",
        "chemical": "Apply fungicides like sulfur or copper-based products regularly.",
        "organic": "Use neem oil, baking soda solutions, or copper sulfate as natural fungicides.",
        "resistant": "Grow black rot-resistant grape varieties.",
        "watering": "Avoid overhead watering to keep leaves dry.",
        "summary": "Combining these methods will help manage the disease effectively."
    },
    "Esca_(Black_Measles)": {
        "title": "To manage black measles in grapes:",
        "probable_causes": "Black measles is caused by a complex of fungi, including *Phaeoacremonium* and *Phaeomoniella* species, which thrive in old and poorly drained vineyards.",
        "cultural": "Remove infected leaves, improve airflow, and avoid overhead watering.",
        "fungicides": "Use copper-based or systemic fungicides like fosetyl-aluminum.",
        "organic": "Apply neem oil or a baking soda solution as a natural fungicide.",
        "resistant": "Grow resistant grape varieties.",
        "monitor": "Apply fungicides before rain and during wet conditions.",
        "summary": "These methods combined will help control the disease."
    },
    "Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "title": "To manage leaf blight in grapes:",
        "probable_causes": "Leaf blight is caused by the fungus *Isariopsis clavispora*, which spreads through infected soil and wet leaves.",
        "cultural": "Prune infected leaves, improve airflow, and avoid overhead watering.",
        "fungicides": "Use copper-based or systemic fungicides like chlorothalonil.",
        "organic": "Apply neem oil or a baking soda solution.",
        "resistant": "Grow disease-resistant grape varieties.",
        "care": "Ensure balanced nutrition and avoid overwatering.",
        "summary": "Combining these methods will help control leaf blight effectively."
    },
    "Healthy": {
        "title": "Don't worry, you have a healthy crop!",
        "summary": "Keep up with regular care and maintenance to prevent future issues."
    }
}

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Resize to 256x256
    img = np.reshape(img, (-1, 256, 256, 3))  # Reshape to fit model input
    
    
    # Predict
    prediction = model.predict(img)
    
    # Get the index of the highest probability
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Get the predicted category name
    predicted_category = categories[int(predicted_class)]
    
    # Get the control methods and probable causes for the predicted category
    control_advice = control_methods.get(predicted_category, {})
    
    return predicted_category, control_advice

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_path = None
    control_advice = None
    
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Define path to save file in static/uploads folder
            upload_folder = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)  # Create folder if not exists
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Run prediction and get control advice
            result, control_advice = predict_image(file_path)
            # Use relative path for img_path to work with `url_for`
            img_path = f'uploads/{file.filename}'

    return render_template('index.html', result=result, img_path=img_path, control_advice=control_advice)

if __name__ == '__main__':
    app.run(debug=True)


