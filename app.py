from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
import os

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# --- Load the saved model and artifacts ONCE when the server starts ---
MODEL_DIR = 'saved_model'
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'best_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    with open(os.path.join(MODEL_DIR, 'features.json'), 'r') as f:
        features = json.load(f)
    with open(os.path.join(MODEL_DIR, 'ad_params.json'), 'r') as f:
        ad_params = json.load(f)  # Load AD parameters
    MODEL_LOADED = True
    print("Model and artifacts loaded successfully.")
except FileNotFoundError:
    MODEL_LOADED = False
    print("ERROR: Model files not found. Run the training script first.")

DESC_NAMES = [desc[0] for desc in Descriptors._descList]
CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)
# --------------------------------------------------------------------

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model is not loaded on the server'}), 500

    data = request.get_json()
    smiles = data.get('smiles', '').strip()

    if not smiles:
        return jsonify({'error': 'SMILES string is missing'}), 400

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({'error': 'Invalid SMILES string provided'}), 400
        
        all_descriptors = CALCULATOR.CalcDescriptors(mol)
        
        desc_dict = {name: val for name, val in zip(DESC_NAMES, all_descriptors)}
        desc_df = pd.DataFrame([desc_dict])
        
        feature_vector = desc_df.reindex(columns=features, fill_value=0)

        # --- Applicability Domain Check ---
        is_in_ad = True
        for feature in features:
            min_val = ad_params[feature]['min']
            max_val = ad_params[feature]['max']
            if not (min_val <= feature_vector[feature].iloc[0] <= max_val):
                is_in_ad = False
                break
        # -----------------------------------
        
        scaled_features = scaler.transform(feature_vector)
        prediction = model.predict(scaled_features)[0]
        
        response_data = {
            'prediction': f"{prediction:.3f}",
            'molecule_svg': Draw.MolToSVG(mol),
            'is_in_ad': is_in_ad  # Add AD status to the response
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
