from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional
#from tensorflow.keras.models import load_model
import json
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


# ============================================================
#  FUNÇÃO DE PRÉ-PROCESSAMENTO (LightGBM) – igual à sua
# ============================================================
def preprocess_for_api(input_data, bundle):
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    map_raca = bundle.get("map_raca", {"Branco": 1, "Pardo": 2, "Preto": 3})
    map_boolean = bundle.get("map_boolean", {"Sim": 1, "YES": 1, "SIM": 1, "TRUE": 1, "Nao": 1, "NAO": 0, "Não": 0, "NÃO": 0, "FALSE": 0})
    map_hist_diabetes = bundle.get("map_hist_diabetes", {"Não": 0, "NAO": 0, "NÃO": 0, "Nao": 0, "1º grau": 3, "1° GRAU": 3, "1 GRAU": 3, "2º grau": 2, "2° GRAU": 2, "2 GRAU": 2, "3º grau": 1, "3° GRAU": 1, "3 GRAU": 1})

    if "origemRacial" in df.columns:
        df["origemRacial"] = df["origemRacial"].replace(map_raca)

    if "idade" in df.columns:
        df["idade"] = pd.to_numeric(df["idade"], errors='coerce')
    
    cols_bool = ["diabetes", "fuma", "hipertensao"]
    for col in cols_bool:
        if col in df.columns:
            df[col] = df[col].replace(map_boolean).astype(float)
    
    if "historicoFamiliarDiabetes" in df.columns:
        df["historicoFamiliarDiabetes"] = (
            df["historicoFamiliarDiabetes"]
            .astype(str).str.strip()
            .replace(map_hist_diabetes).astype(float)
        )
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    input_features = bundle.get("input_features", [])
    for feature in input_features:
        if feature not in df.columns:
            df[feature] = 0
    
    df = df.fillna(0)
    
    scaler = bundle.get("scaler")
    if scaler is not None:
        df_scaled = scaler.transform(df[input_features])
        df_final = pd.DataFrame(df_scaled, columns=input_features)
    else:
        df_final = df[input_features]
    
    return df_final

# ============================================================
#  CARREGAR MODELOS
# ============================================================

print("Carregando modelos...")

# ----- LightGBM -----
try:
    bundle_path = os.path.join(MODEL_DIR, "model_lgbm_bundle.pkl")
    bundle_lgbm = joblib.load(bundle_path)
    model_lgbm = bundle_lgbm["model"]
    input_features = bundle_lgbm["input_features"]
    print("✅ LightGBM carregado")
except Exception as e:
    print(f"❌ Erro ao carregar LightGBM: {e}")
    bundle_lgbm = None
    model_lgbm = None
    input_features = []


# Rede Neural

#try:
#    nn_path = os.path.join(MODEL_DIR, "nn_preeclampsia_24f_nosmote.keras")
#    model_nn = load_model(nn_path)
#
#    norm_path = os.path.join(MODEL_DIR, "nn_norm_stats_24f.json")
#    with open(norm_path) as f:
#        norm_stats = json.load(f)
#    nn_mean = np.array(norm_stats["mean"])
#    nn_std  = np.array(norm_stats["std"]) + 1e-8

#    nn_features = input_features.copy()
#    print("✅ Rede Neural (24f, sem SMOTE) carregada")
#except Exception as e:
#    print(f"⚠ Rede Neural não carregada: {e}")
#    model_nn = None
#    nn_mean = None
#    nn_std = None
#    nn_features = []

# ============================================================
#  SCHEMAS
# ============================================================

class InputData(BaseModel):
    # aqui você pode alinhar com as mesmas 24 features do LGBM se quiser
    peso: float
    imc: float
    diabetes: int
    hipertensao: int
    pesoPrimeiroTrimestre: float
    origemRacial: str
    historicoFamiliarDiabetes: int
    mediaIP: float
    historiaObstetricaAnterior: int
    perdasGestacionais: int
    fuma: int
    idadeGestacional: float
    idadeGestacionalCorrigida: float
    pesoFetal: float
    percentilArteriaUterina: float
    percentilPeso: float
    temHipertensaoPreExistente: int
    temEndometriose: int
    temHipotireoidismo: int
    temIIC: int
    temTrombofilia: int
    temRetocolite: int
    temGastrite: int
    temOutraDoenca: int

class InputLGBM(BaseModel):
    idade: float
    peso: float
    imc: float
    diabetes: int
    hipertensao: int
    pesoPrimeiroTrimestre: float
    origemRacial: str
    historicoFamiliarDiabetes: int
    mediaIP: float
    historiaObstetricaAnterior: int
    perdasGestacionais: int
    fuma: int
    idadeGestacional: float
    idadeGestacionalCorrigida: float
    pesoFetal: float
    percentilArteriaUterina: float
    percentilPeso: float
    temHipertensaoPreExistente: int
    temEndometriose: int
    temHipotireoidismo: int
    temIIC: int
    temTrombofilia: int
    temRetocolite: int
    temGastrite: int
    temOutraDoenca: int

# ============================================================
#  ENDPOINTS
# ============================================================

@app.post("/predict_nn")
def predict_nn(data: InputData):
    if model_nn is None or nn_mean is None:
        return {"error": "Modelo de Rede Neural não disponível"}

    sample = data.dict()

    # 1) mapear origemRacial (string -> número), igual ao treino
    map_raca = {"Branco": 1, "Pardo": 2, "Preto": 3}
    if "origemRacial" in sample:
        sample["origemRacial"] = map_raca.get(sample["origemRacial"], 0)

    # 2) montar vetor na ordem das 24 features
    x = []
    for col in nn_features:
        x.append(float(sample.get(col, 0)))
    X = np.array(x).reshape(1, -1)

    # 3) normalizar com as stats salvas
    X = (X - nn_mean) / nn_std

    # 4) predição
    prob = float(model_nn.predict(X)[0, 0])

    return {
        "risco_pre_eclampsia": prob,
        "risk_level": "alto" if prob >= 0.2 else "baixo"
    }


@app.post("/predict_lgbm")
def predict_lgbm(data: InputLGBM):
    if model_lgbm is None:
        return {"error": "Modelo LightGBM não disponível"}
    
    try:
        df_input = pd.DataFrame([data.dict()])
        df_processed = preprocess_for_api(df_input, bundle_lgbm)

        for feature in input_features:
            if feature not in df_processed.columns:
                df_processed[feature] = -1
        
        df_final = df_processed[input_features].fillna(-1)
        prob = float(model_lgbm.predict_proba(df_final)[0, 1])
        
        return {
            "success": True,
            "risco_pre_eclampsia": prob,
            "risk_level": "alto" if prob > 0.5 else "baixo"
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "risco_pre_eclampsia": 0.0
        }

@app.get("/")
def health_check():
    return {
        "status": "API funcionando",
        "lightgbm_loaded": model_lgbm is not None,
        "neural_network_loaded": model_nn is not None,
        "lightgbm_features": len(input_features),
        "nn_features": len(nn_features)
    }

@app.get("/features")
def get_features():
    return {
        "features": input_features,
        "total": len(input_features)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
