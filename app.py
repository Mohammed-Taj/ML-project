# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the pipeline at startup
pipeline = joblib.load('diamond_pipeline_ready.pkl')

# Define mappings (these should match your training)
color_mapping = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_mapping = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5,
                   'VVS2': 6, 'VVS1': 7, 'IF': 8}

# Reverse mappings for display purposes
color_reverse_mapping = {v: k for k, v in color_mapping.items()}
clarity_reverse_mapping = {v: k for k, v in clarity_mapping.items()}

@app.route('/')
def index():
    return render_template('index.html', 
                          color_options=color_mapping.keys(),
                          clarity_options=clarity_mapping.keys())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = {
            'carat': float(request.form['carat']),
            'color': request.form['color'],
            'clarity': request.form['clarity'],
            'x': float(request.form['x']),
            'y': float(request.form['y']),
            'z': float(request.form['z'])
        }
        
        # Map ordinal features
        df_new = pd.DataFrame([features])
        df_new['color'] = df_new['color'].map(pipeline['mappings']['color'])
        df_new['clarity'] = df_new['clarity'].map(pipeline['mappings']['clarity'])

        # Keep only features used in training
        X_new = df_new[pipeline['feature_names']]
        X_new = pipeline['scaler'].transform(X_new)
        
        # Make prediction
        log_price_pred = pipeline['model'].predict(X_new)[0]
        price_pred = np.exp(log_price_pred)
        
        return render_template('index.html', 
                              prediction_text=f'Predicted Price: ${price_pred:,.2f}',
                              **request.form,
                              color_options=color_mapping.keys(),
                              clarity_options=clarity_mapping.keys())
    
    except Exception as e:
        return render_template('index.html', 
                              prediction_text=f'Error: {str(e)}',
                              color_options=color_mapping.keys(),
                              clarity_options=clarity_mapping.keys())

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        features = {
            'carat': float(data['carat']),
            'color': data['color'],
            'clarity': data['clarity'],
            'x': float(data['x']),
            'y': float(data['y']),
            'z': float(data['z'])
        }
        
        # Map ordinal features
        df_new = pd.DataFrame([features])
        df_new['color'] = df_new['color'].map(pipeline['mappings']['color'])
        df_new['clarity'] = df_new['clarity'].map(pipeline['mappings']['clarity'])

        # Keep only features used in training
        X_new = df_new[pipeline['feature_names']]
        X_new = pipeline['scaler'].transform(X_new)
        
        # Make prediction
        log_price_pred = pipeline['model'].predict(X_new)[0]
        price_pred = np.exp(log_price_pred)
        
        return jsonify({'predicted_price': price_pred})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)