from fastapi import FastAPI
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
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://client-flipkart-prediction.vercel.app"],
    allow_methods=["GET" , "POST"], 
    allow_headers=["*"],  
)
url_model_svm = "https://drive.google.com/uc?export=download&id=1xylAv4t1br1R1IpoTLZF2yigOuPMKDpz"
url_model_rf = "https://drive.google.com/uc?export=download&id=1gjXvX_qi_3Xdua-1iCxlnolIS_hK7zY6"
url_data = "https://drive.google.com/uc?export=download&id=155Y7fC1jrzOzWPXa1O7bqOuomUdipr4j"

model_rf = None
model_svm = None
df = None

@app.on_event("startup")
async def load_models_and_data():
    global model_rf, model_svm, df
    try:
        # ดาวน์โหลดไฟล์ (คุณสามารถตรวจสอบว่ามีไฟล์อยู่แล้วหรือไม่)
        gdown.download(url_model_rf, 'model_rf_new.pkl', quiet=False)
        gdown.download(url_model_svm, 'model_svm.pkl', quiet=False)
        gdown.download(url_data, "cleaned_data.csv", quiet=False)
        
        if os.path.exists('model_rf_new.pkl') and os.path.exists('model_svm.pkl') and os.path.exists('cleaned_data.csv'):
            model_rf = joblib.load('model_rf_new.pkl')
            model_svm = joblib.load('model_svm.pkl')
            df = pd.read_csv('cleaned_data.csv')
            print("Models and data loaded successfully!")
        else:
            print("Model or data files not found.")
    except Exception as e:
        print(f"Error loading model or data: {str(e)}")
class InputData(BaseModel):
    price: float
    quantity: int
    customer_rating: float
@app.post("/predictsvm")
async def Svm_predictions(data:InputData):
    input_features = pd.DataFrame([[data.price, data.quantity, data.customer_rating]],columns=["Price (INR)", "Quantity Sold", "Customer Rating"])
    prediction = model_svm.predict(input_features)
    print(prediction)
    return {"status": "success", "message": "PASSED", "prediction": prediction[0]}
@app.post("/predictRF")
async def predictRF(data:InputData):
    input_features = pd.DataFrame([[data.price, data.quantity, data.customer_rating]],columns=["Price (INR)", "Quantity Sold", "Customer Rating"])
    prediction = model_rf.predict(input_features)
    print(prediction)
    return {"status": "success", "message": "PASSED", "prediction": prediction[0]}

@app.get("/showmodels")
async def get_models():
    global df 
    
    if df is None:
        return {"status": "error", "message": "Dataset not loaded."}
    x = df[["Price (INR)", "Quantity Sold", "Customer Rating"]]
    y = df["Total Sales (INR)"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    

    y_predict_svm = model_svm.predict(x_test)
    mae_svm = mean_absolute_error(y_test, y_predict_svm)


    y_predict_rf = model_rf.predict(x_test)
    mae_rf = mean_absolute_error(y_test, y_predict_rf)
    

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
        "message": "Model comparison and errors",
        "mae_svm": mae_svm,
        "mae_rf": mae_rf,
        "plot_svm_base64": plot_svm_base64,
        "plot_rf_base64": plot_rf_base64
    }