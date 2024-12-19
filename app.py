from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your models
age_model = load_model('agemodel.h5');
gender_model = load_model('genmodel.h5');

# age_model = load_model('agemodel.h5')
# gender_model = load_model('genmodel.h5')

# Ensure the upload folder exists
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def model_predict(img_path):
    # Load the image and resize to 200x200 (since the model expects this input size)
    img = image.load_img(img_path, target_size=(200, 200))

    # Convert the image to an array and normalize it
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Reshape to (1, 200, 200, 3)
    img_array /= 255.0  # Normalize the image

    # Predict age
    age_pred = age_model.predict(img_array)
    age = int(np.round(age_pred[0][0]))

    # Predict gender
    gender_pred = gender_model.predict(img_array)
    gender = 'Male' if gender_pred[0][0] < 0.5 else 'Female'  #og >

    return age, gender


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Predict
            age, gender = model_predict(filepath)

            return render_template('index.html', age=age, gender=gender, image_url=filepath)
    return render_template('index.html', age=None, gender=None, image_url=None)


if __name__ == "__main__":
    app.run(debug=True)
