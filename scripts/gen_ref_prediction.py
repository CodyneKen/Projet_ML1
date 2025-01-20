import pandas as pd
import joblib
import fct_model

ref_data = pd.read_csv("../data/ref_data.csv")
ref_data_report = ref_data.copy()
ref_data_report["prediction"] = None

encoder = joblib.load("../artifacts/encoder.pkl")
model_data = joblib.load("../artifacts/model.pkl")
model = model_data["model"]
scaler = joblib.load("../artifacts/scaler.pkl")

embedding_size = 162
emotion_mapping = {
    'C': 'Colère 😡​',   
    'T': 'Tristesse 😢​',
    'J': 'Joie 😁​',     
    'P': 'Peur 😨​',     
    'D': 'Dégoût ​☹️​',   
    'S': 'Surprise ​​😮​', 
    'N': 'Neutre 😐​'    
}

features_table_ref = ref_data.drop(columns=["target"])
for i in range(len(ref_data)):
    features = features_table_ref.iloc[i]
    # Scale and reshape the features for the model
    features_scaled = scaler.transform(features.values.reshape(1, -1))
    features_reshaped = features_scaled.reshape(1, embedding_size, 1)

    # Predict the class probabilities
    pred = model.predict(features_reshaped)
    predicted_class_label = encoder.inverse_transform(pred)[0][0]

    #ref_data_report["prediction"].iloc[i] = predicted_class_label
    ref_data_report.at[i, "prediction"] = predicted_class_label

ref_data_report.to_csv("../data/ref_data_report.csv", index=False)