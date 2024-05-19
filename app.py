from flask import Flask, request, render_template, url_for, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    image = Image.open(image)
    image = image.resize((150, 150))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 150, 150, 3)
    return image_arr

classes = ['Buildings' ,'Forest', 'Glacier' ,'Mountain' ,'Sea' ,'Street']
model = tf.keras.models.load_model("Intel_Image_Classification.h5")

@app.route('/predictImage', methods=['POST'])
def predictImage():
    try:
        if 'uploadedimage' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files['uploadedimage']
        image_arr = preprossing(image)
        result = model.predict(image_arr)
        print(result)
        index = np.argmax(result)
        prediction = classes[index]
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})

@app.route('/')
def index():
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=True)
