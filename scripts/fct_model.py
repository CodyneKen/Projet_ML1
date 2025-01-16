import pandas as pd
import numpy as np
import keras
from keras import layers, models
from joblib import dump
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import dill
import librosa
import os

def preprocess_data(df):
    """
    Prétraitement des données et renvoie X, y, encoder, scaler.
    Gère le cas où le dataframe contient soit 'target', soit 'target' et 'prediction'.
    """
    if 'target' not in df.columns:
        raise ValueError("The input DataFrame must contain a 'target' column.")
    
    if 'prediction' in df.columns:
        X = df.drop(columns=['target', 'prediction']).values
    else:
        X = df.iloc[:, :-1].values
        
    y = df['target'].values

    #on encode les labels
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(y).reshape(-1,1)).toarray()

    #normalisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #on reshape les données
    X = np.expand_dims(X, axis=2)

    return X, Y, encoder, scaler


def define_model(input_dim):
    """
    Définition du modèle
    """
    model_full = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(128, kernel_size=5, strides=1, activation='relu', padding='same'),
        layers.Dropout(0.3),

        layers.Conv1D(64, kernel_size=5, strides=1, activation='relu', padding='same'),
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        
        layers.Dense(7, activation='softmax')
    ])

    model_full.summary()

    model_full.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model_full

def train_model(X_train, Y_train, X_test, Y_test, model, epochs=50, batch_size=64):
    """
    Entraîner le modèle
    """
    rlrp = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)

    history = model.fit(
        X_train, 
        Y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, Y_test),
        callbacks=[rlrp]
    )

    return model, history

def display_results(model, history, x_test, y_test, epochs=50):
    print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

    print_epochs = [i for i in range(epochs)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    fig.set_size_inches(20,6)
    ax[0].plot(print_epochs , train_loss , label = 'Training Loss')
    ax[0].plot(print_epochs , test_loss , label = 'Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(print_epochs , train_acc , label = 'Training Accuracy')
    ax[1].plot(print_epochs , test_acc , label = 'Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    plt.show()

def predict_on_test(model, encoder, x_test, y_test):
    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)

    y_test = encoder.inverse_transform(y_test)

    return y_pred, y_test

def show_predictions(y_pred, y_test):
    df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
    df['Predicted Labels'] = y_pred.flatten()
    df['Actual Labels'] = y_test.flatten()

    print(df.head(10))

def show_conf_matrix(y_pred, y_test, encoder):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (12, 10))
    cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.show()

def predict_on_audio(model, encoder, scaler, audio_data, sample_rate):
    """
    Prédire l'émotion sur un fichier audio
    """
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

    with open("../artifacts/extract_features.pkl", "rb") as f:
        extract_features_test = dill.load(f)
        extract_features_test.__globals__["np"] = np
        extract_features_test.__globals__["librosa"] = librosa

    # Extract features using the pre-loaded function
    try:
        features = extract_features_test(audio_data, sample_rate)
    except Exception as e:
        return {"error": f"Failed to extract features: {str(e)}"}

    # Scale and reshape the features for the model
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_reshaped = features_scaled.reshape(1, embedding_size, 1)

    # Predict the class probabilities
    try:
        pred = model.predict(features_reshaped)
        predicted_class_label = encoder.inverse_transform(pred)[0][0]
        predicted_emotion = emotion_mapping[predicted_class_label]
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Return the predicted label
    return {"prediction": predicted_emotion}

def train_save_model(csv_path, output_model_path, verbose=2):
    """
    Entrainer un cnn sur les données du csv et sauvegarder le model, encoder, scaler
    """
    data = pd.read_csv(csv_path)

    """ #on récup les données du csv
    X = data.iloc[:, :-1].values  
    y = data['labels'].values

    #on encode les labels
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(y).reshape(-1,1)).toarray()

    #on sauv l'encoder
    dump(encoder, f'{output_model_path}/encoder.pkl')

    #normalisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #on sauv le scaler
    dump(scaler, f'{output_model_path}/scaler.pkl')

    #on reshape les données
    X = np.expand_dims(X, axis=2) """

    X, Y, encoder, scaler = preprocess_data(data)

    #on sauv l'encoder
    dump(encoder, f'{output_model_path}/encoder.pkl')

    #on sauv le scaler
    dump(scaler, f'{output_model_path}/scaler.pkl')

    input_dim = X.shape[1]

    #model
    model_full = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(128, kernel_size=5, strides=1, activation='relu', padding='same'),
        layers.Dropout(0.3),

        layers.Conv1D(64, kernel_size=5, strides=1, activation='relu', padding='same'),
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        
        layers.Dense(7, activation='softmax')
    ])

    model_full.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    rlrp = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)


    history = model_full.fit(
        X, 
        Y, 
        epochs=50, 
        batch_size=64, 
        callbacks=[rlrp],
        verbose=verbose
    )

    #on sauvegarde le model
    dump(model_full, f'{output_model_path}/model.pkl')

    return model_full, history

def save_feedback(audio_data, sample_rate, target, prediction, output_path):
    with open("../artifacts/extract_features.pkl", "rb") as f:
        extract_features_test = dill.load(f)
        extract_features_test.__globals__["np"] = np
        extract_features_test.__globals__["librosa"] = librosa
        
    try:
        features = extract_features_test(audio_data, sample_rate)
    except Exception as e:
        return {"error": f"Failed to extract features: {str(e)}"}

    Features = pd.DataFrame(features).T
    Features['target'] = target
    Features['prediction'] = prediction
    
    file_exists_and_non_empty = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    
    Features.to_csv(output_path, mode='a',
                    header=not file_exists_and_non_empty,
                    index=False)