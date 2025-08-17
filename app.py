import os
import io
import uvicorn
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, auth
from google.cloud import firestore
from google.oauth2 import service_account
from fastapi import FastAPI, File, UploadFile, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
from dotenv import load_dotenv
import json
import random
from pydantic import BaseModel, Field
from typing import Optional

# --- Environment Variable Loading ---
load_dotenv()

# --- Firebase Setup ---
SERVICE_ACCOUNT_FILE = "firebase-adminsdk.json"
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)
    print("✅ Firebase Admin SDK initialized successfully.")
    gcp_credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    db = firestore.Client(credentials=gcp_credentials, project=gcp_credentials.project_id)
    print("✅ Firestore client initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing Firebase/Firestore: {e}")
    db = None

# --- FastAPI App Initialization ---
app = FastAPI(title="Farmer's Companion API")
templates = Jinja2Templates(directory="templates")

# --- Global Variables for Model & Data ---
model = None
class_names = []
plant_data_df = None

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHERAPI_KEY = "c7342727798e4729b39183944251608" # As requested

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables.")
if not WEATHERAPI_KEY:
    print("⚠️ WARNING: WEATHERAPI_KEY not found. Weather feature will be disabled.")


# --- Pydantic Models for Data Validation ---
class Location(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    manual_entry: Optional[str] = None

class LandData(BaseModel):
    name: str
    area: float
    unit: str
    location: Location


# --- Authentication ---
oauth2_scheme = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)):
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
    try:
        return auth.verify_id_token(credentials.credentials)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid credentials: {e}")

# --- Model and Data Loading on Startup ---
@app.on_event("startup")
async def load_model_and_data():
    global model, class_names, plant_data_df
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'fine_tuned_model', 'best_fine_tuned_model.keras')
        model = tf.keras.models.load_model(model_path)
        data_path = os.path.join(os.path.dirname(__file__), 'plant_data.csv')
        plant_data_df = pd.read_csv(data_path)
        class_names.extend(plant_data_df['class_name'].tolist())
        print("✅ Model and plant data loaded successfully!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during startup: {e}")

# --- Helper Functions ---
def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((256, 256))
    img_array = np.expand_dims(np.array(img), axis=0) / 255.0
    return img_array

def get_market_price(plant_name: str) -> float:
    prices = {'Apple': (80, 120), 'Tomato': (20, 35), 'Potato': (15, 25), 'Grape': (50, 90)}
    price_range = prices.get(plant_name, (10, 50))
    return random.uniform(price_range[0], price_range[1])

async def get_ai_recommendations(plant_name: str, disease_name: str, impact: str) -> dict:
    if not GEMINI_API_KEY: return {"error": "API key is missing."}
    prompt = f"You are a helpful agricultural expert for a farmer with a {plant_name} plant suffering from {disease_name}. The impact is '{impact}'. Provide a JSON object with three keys: 'harvest_recommendations', 'selling_strategies', and 'bargaining_tips'. Each value should be a brief, clear string."
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(apiUrl, json=payload, timeout=30)
            response.raise_for_status()
            return json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")
        return {"error": "Failed to get AI recommendations."}

# ==============================================================================
# HTML PAGE SERVING ROUTES
# FIX: Removed the `Depends(get_current_user)` dependency from these routes.
# The JavaScript inside each page will handle authentication checks.
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_root_as_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def serve_login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/onboarding", response_class=HTMLResponse)
async def serve_onboarding_page(request: Request): # <- FIX: Auth dependency removed
    return templates.TemplateResponse("onboarding.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard_page(request: Request): # <- FIX: Auth dependency removed
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/prediction", response_class=HTMLResponse)
async def serve_prediction_page(request: Request): # <- FIX: Auth dependency removed
    return templates.TemplateResponse("prediction.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def serve_history_page(request: Request): # <- FIX: Auth dependency removed
    return templates.TemplateResponse("history.html", {"request": request})

# ==============================================================================
# API ENDPOINTS (These remain protected)
# ==============================================================================

@app.get("/api/user-status")
async def get_user_status(user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    doc_ref = db.collection("predictions").document(user_uid)
    doc = doc_ref.get()
    has_onboarded = doc.exists and "lands" in doc.to_dict()
    return JSONResponse(content={"has_onboarded": has_onboarded})

@app.post("/api/onboard")
async def handle_onboarding(land_data: LandData, user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    user_doc_ref = db.collection("predictions").document(user_uid)
    try:
        user_doc_ref.set({"user_email": user.get("email", "N/A"), "lands": [land_data.dict()]}, merge=True)
        print(f"✅ Onboarding data saved for user {user_uid}.")
        return JSONResponse(content={"status": "success"}, status_code=201)
    except Exception as e:
        print(f"❌ Error saving onboarding data for {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not save farm details.")

@app.get("/api/dashboard-data")
async def get_dashboard_data(user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    doc_ref = db.collection("predictions").document(user_uid)
    doc = doc_ref.get()
    if not doc.exists or "lands" not in doc.to_dict():
        raise HTTPException(status_code=404, detail="User has not completed onboarding.")
    farm_data = doc.to_dict().get("lands", [])[0]
    location = farm_data.get("location", {})
    lat = location.get("latitude")
    lon = location.get("longitude")
    weather_data = None
    if lat and lon and WEATHERAPI_KEY:
        weather_url = f"http://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY}&q={lat},{lon}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(weather_url)
                resp.raise_for_status()
                weather_data = resp.json()
                print(f"✅ Fetched weather from WeatherAPI for {user_uid} at ({lat}, {lon}).")
        except Exception as e:
            print(f"⚠️ Could not fetch weather data from WeatherAPI: {e}")
            weather_data = {"error": "Could not retrieve weather data."}
    return JSONResponse(content={"farm_data": farm_data, "weather_data": weather_data})

@app.get("/api/history")
async def get_prediction_history(user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    doc_ref = db.collection("predictions").document(user_uid)
    doc = doc_ref.get()
    if doc.exists:
        history = doc.to_dict().get("history", [])
        for item in history:
            if 'timestamp' in item and isinstance(item['timestamp'], datetime):
                item['timestamp'] = item['timestamp'].isoformat()
        return JSONResponse(content={"history": history})
    return JSONResponse(content={"history": []})

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    if not model or plant_data_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    image_bytes = await file.read()
    preprocessed_image = preprocess_image(image_bytes)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])
    plant_data = plant_data_df[plant_data_df['class_name'] == predicted_class_name].iloc[0]
    plant_name, disease_name, impact = plant_data['plant_name'], plant_data['disease_name'], plant_data['impact']
    predicted_yield = plant_data['base_yield_kg_per_acre'] * (1 - plant_data['yield_reduction_factor'])
    market_price = get_market_price(plant_name)
    total_harvest_value = predicted_yield * market_price
    ai_recs = await get_ai_recommendations(plant_name, disease_name, impact)
    prediction_details = {
        "prediction": predicted_class_name, "plant_name": plant_name, "disease_name": disease_name,
        "confidence": confidence, "predicted_yield_kg_per_acre": predicted_yield, "market_price_per_kg": market_price,
        "total_harvest_value": total_harvest_value, "recommendations": ai_recs, "impact": impact,
        "timestamp": datetime.utcnow()
    }
    if db:
        user_uid = user["uid"]
        user_doc_ref = db.collection("predictions").document(user_uid)
        user_doc_ref.set({"history": firestore.ArrayUnion([prediction_details])}, merge=True)
        print(f"✅ Prediction saved for user {user_uid}.")
    return {
        "prediction": prediction_details['prediction'], "confidence": f"{prediction_details['confidence']:.2f}",
        "predicted_yield_kg_per_acre": f"{prediction_details['predicted_yield_kg_per_acre']:.2f}",
        "market_price_per_kg": f"{prediction_details['market_price_per_kg']:.2f}",
        "total_harvest_value": f"{prediction_details['total_harvest_value']:.2f}",
        "recommendations": prediction_details['recommendations'], "plant_name": prediction_details['plant_name'],
        "disease_name": prediction_details['disease_name'], "impact": prediction_details['impact']
    }

# --- Run App ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
