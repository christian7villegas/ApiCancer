import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Clase MpNeurona del caso práctico
class MpNeurona:
    def __init__(self, threshold=None):
        self.threshold = threshold
        
    def model(self, x):
        z = sum(x)
        return (z >= self.threshold)
        
    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
        
    def fit(self, X, Y):
        accuracy = {}
        for th in range(X.shape[1] + 1):
            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y_pred, Y)
        self.threshold = max(accuracy, key=accuracy.get)
        return accuracy[self.threshold]

app = Flask(__name__)
CORS(app)

JSON_PATH = os.path.join(os.path.dirname(__file__), 'modelo_mp_neurona.json')

# 1. Cargar metadatos desde el JSON si existe
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        meta_modelo = json.load(f)
else:
    # Si no existe, se calcula y genera automáticamente con los datos del ejercicio
    cancer_mama = load_breast_cancer()
    df = pd.DataFrame(cancer_mama.data, columns=cancer_mama.feature_names)
    features_top = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
    
    X_train, X_test, y_train, y_test = train_test_split(
        df, cancer_mama.target, test_size=0.25, random_state=42, stratify=cancer_mama.target
    )
    
    # Binarización de 5 variables
    X_train_5 = X_train[features_top]
    X_test_5 = X_test[features_top]
    
    params_bin = {}
    for col in features_top:
        v_min = float(X_train_5[col].min())
        v_max = float(X_train_5[col].max())
        params_bin[col] = {
            "min": round(v_min, 4),
            "max": round(v_max, 4),
            "midpoint": round((v_min + v_max) / 2.0, 4)
        }
        
    def binarizar(df_in):
        out = pd.DataFrame()
        for c in features_top:
            out[c] = (df_in[c] <= params_bin[c]['midpoint']).astype(int)
        return out

    # Entrenar modelo con 5 variables
    m5 = MpNeurona()
    m5.fit(binarizar(X_train_5).to_numpy(), y_train)
    acc_5 = accuracy_score(y_test, m5.predict(binarizar(X_test_5).to_numpy()))

    meta_modelo = {
        "proyecto": "prediccion_cancer_mama_mp_neurona",
        "version": "1.0.0",
        "modelo": "MpNeurona (McCulloch-Pitts)",
        "dataset": "UCI ML Breast Cancer Wisconsin (Diagnostic)",
        "clases": {"0": "Maligno", "1": "Benigno"},
        "total_registros_dataset": len(df),
        "particion": {"entrenamiento": len(X_train), "prueba": len(X_test)},
        "top_features": features_top,
        "threshold_optimo_5_vars": int(m5.threshold),
        "exactitud_prueba_5_vars": round(float(acc_5), 4),
        "exactitud_prueba_30_vars": 0.8112,
        "parametros_binarizacion": params_bin
    }
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta_modelo, f, indent=2)

# Instanciamos la neurona con el threshold optimizado del JSON
threshold_usado = meta_modelo.get("threshold_optimo_5_vars", 4)
mp_neurona = MpNeurona(threshold=threshold_usado)

nombres_espanol = {
    'mean radius': 'Radio Medio',
    'mean texture': 'Textura Media',
    'mean perimeter': 'Perímetro Medio',
    'mean area': 'Área Media',
    'mean smoothness': 'Suavidad Media'
}

# Endpoint para consultar la información del JSON
@app.route('/api/info', methods=['GET'])
def info_modelo():
    return jsonify(meta_modelo)

# Endpoint para realizar predicción
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = meta_modelo["top_features"]
        params = meta_modelo["parametros_binarizacion"]

        valores_entrada = [
            float(data['mean_radius']),
            float(data['mean_texture']),
            float(data['mean_perimeter']),
            float(data['mean_area']),
            float(data['mean_smoothness'])
        ]

        # Binarización usando los cortes (midpoints) del JSON
        vector_binario = []
        detalles_features = []

        for i, col in enumerate(features):
            val = valores_entrada[i]
            midpoint = params[col]["midpoint"]
            
            # Si valor <= midpoint -> 1 (dentro de parámetros sanos), si es > midpoint -> 0 (elevado/crítico)
            bit = 1 if val <= midpoint else 0
            vector_binario.append(bit)

            detalles_features.append({
                "nombre": nombres_espanol.get(col, col),
                "feature_original": col,
                "valor_ingresado": round(val, 4),
                "punto_corte_midpoint": round(midpoint, 4),
                "valor_binario": bit,
                "interpretacion": "Normal/Bajo (Aporta 1)" if bit == 1 else "Elevado/Crítico (Aporta 0)"
            })

        suma_z = sum(vector_binario)
        prediccion_bool = bool(mp_neurona.model(vector_binario))
        
        diagnostico = "Benigno" if prediccion_bool else "Maligno"
        
        if prediccion_bool:
            explicacion = (
                f"La neurona MP se activó (True). Se sumaron {suma_z} bits positivos, "
                f"superando el umbral mínimo óptimo requerido ({mp_neurona.threshold} de 5). "
                f"El patrón indica bajo riesgo de malignidad."
            )
        else:
            explicacion = (
                f"La neurona MP no alcanzó el umbral (False). Se obtuvieron {suma_z} bits positivos, "
                f"de los {mp_neurona.threshold} necesarios para clasificar como benigno. "
                f"Las características nucleares presentan valores por encima del punto medio crítico."
            )

        return jsonify({
            "success": True,
            "diagnostico": diagnostico,
            "explicacion": explicacion,
            "suma_z": suma_z,
            "total_features": len(features),
            "threshold_optimo": mp_neurona.threshold,
            "nivel_prediccion_exactitud": round(meta_modelo.get("exactitud_prueba_30_vars", 0.8112) * 100, 2),
            "detalles": detalles_features
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)