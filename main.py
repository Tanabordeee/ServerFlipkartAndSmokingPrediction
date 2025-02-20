from fastapi import FastAPI, UploadFile, File
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import joblib
import gdown
import os
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

url_model_svm = "https://drive.google.com/uc?export=download&id=1xylAv4t1br1R1IpoTLZF2yigOuPMKDpz"
url_model_rf = "https://drive.google.com/uc?export=download&id=1gjXvX_qi_3Xdua-1iCxlnolIS_hK7zY6"
url_data = "https://drive.google.com/uc?export=download&id=155Y7fC1jrzOzWPXa1O7bqOuomUdipr4j"
url_model_neural = "https://drive.google.com/uc?export=download&id=1cPDW-dOH8tvgcsPcSuMIfzZBmMxDdRJ0"

model_neural = None
model_rf = None
model_svm = None
df = None

@app.on_event("startup")
async def load_models_and_data():
    global model_rf, model_svm, df, model_neural
    try:
        # Download model and data files
        gdown.download(url_model_rf, 'model_rf_new.pkl', quiet=False)
        gdown.download(url_model_svm, 'model_svm.pkl', quiet=False)
        gdown.download(url_data, "cleaned_data.csv", quiet=False)
        gdown.download(url_model_neural, "smoking_detection_model.pkl", quiet=False)

        print("Files downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model or data: {str(e)}")

class InputData(BaseModel):
    price: float
    quantity: int
    customer_rating: float

@app.post("/predictsvm")
async def Svm_predictions(data: InputData):
    global model_svm
    if model_svm is None:
        model_svm = joblib.load('model_svm.pkl')  # Lazy load model when needed
    
    input_features = pd.DataFrame([[data.price, data.quantity, data.customer_rating]], columns=["Price (INR)", "Quantity Sold", "Customer Rating"])
    prediction = model_svm.predict(input_features)
    
    return {"status": "success", "message": "Prediction successful", "prediction": prediction[0]}

@app.post("/predictRF")
async def predictRF(data: InputData):
    global model_rf
    if model_rf is None:
        model_rf = joblib.load('model_rf_new.pkl')  # Lazy load model when needed
    
    input_features = pd.DataFrame([[data.price, data.quantity, data.customer_rating]], columns=["Price (INR)", "Quantity Sold", "Customer Rating"])
    prediction = model_rf.predict(input_features)
    
    return {"status": "success", "message": "Prediction successful", "prediction": prediction[0]}

@app.get("/showmodels")
async def get_models():
    global df
    if df is None:
        df = pd.read_csv('cleaned_data.csv')
    
    x = df[["Price (INR)", "Quantity Sold", "Customer Rating"]]
    y = df["Total Sales (INR)"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Lazy load models only when needed
    global model_svm, model_rf
    if model_svm is None:
        model_svm = joblib.load('model_svm.pkl')
    if model_rf is None:
        model_rf = joblib.load('model_rf_new.pkl')
    
    y_predict_svm = model_svm.predict(x_test)
    mae_svm = mean_absolute_error(y_test, y_predict_svm)

    y_predict_rf = model_rf.predict(x_test)
    mae_rf = mean_absolute_error(y_test, y_predict_rf)

    # Generate plots to avoid memory overload
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_predict_svm, color="red", alpha=0.6)
    ax.set_xlabel("True Total Sales (INR)")
    ax.set_ylabel("Predicted Total Sales (INR)")
    ax.set_title("True vs Predicted Total Sales (SVM)")
    buf_svm = BytesIO()
    plt.savefig(buf_svm, format="png")
    buf_svm.seek(0)
    plot_svm_base64 = base64.b64encode(buf_svm.getvalue()).decode("utf-8")
    buf_svm.close()
    plt.close(fig)

    fig, ay = plt.subplots(figsize=(8, 6))
    ay.scatter(y_test, y_predict_rf, color="blue", alpha=0.6)
    ay.set_xlabel("True Total Sales (INR)")
    ay.set_ylabel("Predicted Total Sales (INR)")
    ay.set_title("True vs Predicted Total Sales (RF)")
    buf_rf = BytesIO()
    plt.savefig(buf_rf, format="png")
    buf_rf.seek(0)
    plot_rf_base64 = base64.b64encode(buf_rf.getvalue()).decode("utf-8")
    buf_rf.close()
    plt.close(fig)

    return {
        "status": "success",
        "mae_svm": mae_svm,
        "mae_rf": mae_rf,
        "plot_svm_base64": plot_svm_base64,
        "plot_rf_base64": plot_rf_base64
    }

@app.post("/predictimage")
async def predict(file: UploadFile = File(...)):
    global model_neural
    if model_neural is None:
        model_neural = joblib.load('smoking_detection_model.pkl')  # Lazy load model when needed
    
    # Open and preprocess image
    img = Image.open(file.file)
    img = img.resize((224, 224))  # Resize image
    img = np.array(img) / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Perform prediction
    pred = model_neural.predict(img)
    result = "Smoking" if pred >= 0.5 else "Not Smoking"
    
    # Convert image to base64 to send in response
    buffer = BytesIO()
    img_pil = Image.fromarray((img[0] * 255).astype(np.uint8))  # Convert back to image from normalized array
    img_pil.save(buffer, format="JPEG")
    image = base64.b64encode(buffer.getvalue()).decode()
    
    # Free up memory after prediction
    del img
    del img_pil
    gc.collect()

    return {"result": result, "image": image}
