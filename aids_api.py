import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load your pre-trained model
model = load("decision_tree_model.joblib")

api = Flask(__name__)
CORS(api)

@api.route('/api/hiv_prediction', methods=['POST'])
def predict_hiv_status():
    try:
        # Get data from request
        data = request.json['inputs']
        input_df = pd.DataFrame(data)
        
        # Make prediction
        # This is the important part - we're using predict_proba to get probabilities
        prediction = model.predict_proba(input_df)
        class_labels = model.classes_
        
        # Format response
        response = []
        for prob in prediction:
            prob_dict = {}
            for k, v in zip(class_labels, prob):
                prob_dict[str(k)] = round(float(v)*100, 2)
            response.append(prob_dict)
        
        return jsonify({'prediction': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    api.run(host='0.0.0.0', debug=True)
