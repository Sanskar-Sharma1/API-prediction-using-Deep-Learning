from CombinedData.Preprocessed.app1 import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('my_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['input'])
    prediction = model.predict(input_data)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
