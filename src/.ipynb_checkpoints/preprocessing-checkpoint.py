from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def create_preprocessor(num_cols, cat_cols, X):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
    ])
    preprocessor.fit(X)
    return preprocessor

def save_preprocessor(preprocessor, path):
    joblib.dump(preprocessor, path)
