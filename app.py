import os
import io
import uvicorn
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, auth, firestore
from google.cloud import firestore as google_firestore
from google.oauth2 import service_account
from fastapi import FastAPI, File, UploadFile, Request, Depends, HTTPException, status, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
from dotenv import load_dotenv
import json
import random
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid # Used for creating unique IDs for land plots

# --- Environment Variable Loading ---
load_dotenv()

# --- Firebase Setup ---
SERVICE_ACCOUNT_FILE = "firebase-adminsdk.json"
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)
    print("✅ Firebase Admin SDK initialized successfully.")
    db = firestore.client()
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
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found in .env file.")
if not WEATHERAPI_KEY:
    print("⚠️ WARNING: WEATHERAPI_KEY not found in .env file. Weather feature will be disabled.")

# --- Pydantic Models for Data Validation ---
class Location(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    manual_entry: Optional[str] = None

class LandData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    area: float
    unit: str
    location: Location

class LandDeleteRequest(BaseModel):
    plot_id: str


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

async def get_ai_market_price_estimate(plant_name: str, location_str: str) -> float:
    """
    NEW: This function replaces the old random price generator.
    It asks the Gemini AI for a realistic market price based on the crop and location.
    """
    if not GEMINI_API_KEY:
        print("⚠️ Gemini API key missing, falling back to random price.")
        return random.uniform(10, 100) # Fallback

    prompt = (
        f"You are an agricultural market analyst. Based on current market trends, "
        f"what is a single, realistic, estimated market price per kilogram in INR for '{plant_name}' "
        f"in the region of '{location_str}'? "
        f"Your response MUST be a JSON object with a single key 'estimated_price_inr' and a numerical value. "
        f"Example: {{\"estimated_price_inr\": 35.50}}"
    )
    
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(apiUrl, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            # --- NEW: TOKEN USAGE REPORTING ---
            if 'usageMetadata' in result:
                usage = result['usageMetadata']
                print("\n--- 🧠 AI Price Estimator Token Report ---")
                print(f"   Prompt Tokens: {usage.get('promptTokenCount', 'N/A')}")
                print(f"   Response Tokens: {usage.get('candidatesTokenCount', 'N/A')}")
                print(f"   TOTAL TOKENS USED: {usage.get('totalTokenCount', 'N/A')}")
                print("-----------------------------------------\n")
            # --- END OF NEW BLOCK ---

            price_data = json.loads(result['candidates'][0]['content']['parts'][0]['text'])
            price = float(price_data.get("estimated_price_inr", random.uniform(10, 100)))
            print(f"🧠 AI Price Estimate for {plant_name} in {location_str}: ₹{price:.2f}")
            return price
    except Exception as e:
        print(f"❌ Error getting AI price estimate: {e}. Falling back to random price.")
        return random.uniform(10, 100) # Fallback on any error

async def get_ai_recommendations(plant_name: str, disease_name: str, impact: str) -> dict:
    if not GEMINI_API_KEY: return {"error": "API key is missing."}
    prompt = f"You are a helpful agricultural expert for a farmer with a {plant_name} plant suffering from {disease_name}. The impact is '{impact}'. Provide a JSON object with three keys: 'harvest_recommendations', 'selling_strategies', and 'bargaining_tips'. Each value should be a brief, clear string."
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(apiUrl, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            # --- NEW: TOKEN USAGE REPORTING ---
            if 'usageMetadata' in result:
                usage = result['usageMetadata']
                print("\n--- 🌿 AI Recommendations Token Report ---")
                print(f"   Prompt Tokens: {usage.get('promptTokenCount', 'N/A')}")
                print(f"   Response Tokens: {usage.get('candidatesTokenCount', 'N/A')}")
                print(f"   TOTAL TOKENS USED: {usage.get('totalTokenCount', 'N/A')}")
                print("-----------------------------------------\n")
            # --- END OF NEW BLOCK ---

            return json.loads(result['candidates'][0]['content']['parts'][0]['text'])
    except Exception as e:
        print(f"❌ Error calling Gemini API for recommendations: {e}")
        return {"error": "Failed to get AI recommendations."}

# --- HTML PAGE SERVING ROUTES (Unchanged) ---
@app.get("/", response_class=HTMLResponse)
async def serve_root_as_login(request: Request): return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def serve_login_page(request: Request): return templates.TemplateResponse("login.html", {"request": request})

@app.get("/onboarding", response_class=HTMLResponse)
async def serve_onboarding_page(request: Request): return templates.TemplateResponse("onboarding.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard_page(request: Request): return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/prediction", response_class=HTMLResponse)
async def serve_prediction_page(request: Request): return templates.TemplateResponse("prediction.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def serve_history_page(request: Request): return templates.TemplateResponse("history.html", {"request": request})

# --- API ENDPOINTS ---

@app.get("/api/geocode")
async def geocode_location(q: str = Query(..., min_length=1), user: dict = Depends(get_current_user)):
    if not WEATHERAPI_KEY: raise HTTPException(status_code=503, detail="Weather service is not configured.")
    search_url = f"http://api.weatherapi.com/v1/search.json?key={WEATHERAPI_KEY}&q={q}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(search_url)
            resp.raise_for_status()
            results = resp.json()
            if not results: raise HTTPException(status_code=404, detail=f"Location '{q}' not found.")
            first_result = results[0]
            full_name = f"{first_result['name']}, {first_result['region']}, {first_result['country']}"
            return JSONResponse(content={"name": full_name, "latitude": first_result['lat'], "longitude": first_result['lon']})
        except httpx.HTTPStatusError: raise HTTPException(status_code=404, detail=f"Could not find location '{q}'. Please be more specific.")
        except Exception as e:
            print(f"❌ Unexpected error during geocoding: {e}")
            raise HTTPException(status_code=500, detail="An error occurred while searching for the location.")

@app.get("/api/user-status")
async def get_user_status(user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    doc_ref = db.collection("predictions").document(user_uid)
    doc = doc_ref.get()
    has_onboarded = doc.exists and "lands" in doc.to_dict() and len(doc.to_dict()["lands"]) > 0
    return JSONResponse(content={"has_onboarded": has_onboarded})

@app.post("/api/onboard")
async def handle_onboarding(land_data: LandData, user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    user_doc_ref = db.collection("predictions").document(user_uid)
    try:
        user_doc_ref.set({"user_email": user.get("email", "N/A"), "lands": firestore.ArrayUnion([land_data.dict()])}, merge=True)
        print(f"✅ New plot added for user {user_uid}.")
        return JSONResponse(content={"status": "success", "plot_id": land_data.id}, status_code=201)
    except Exception as e:
        print(f"❌ Error saving new plot for {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not save new plot details.")

@app.get("/api/dashboard-data")
async def get_dashboard_data(user: dict = Depends(get_current_user)):
    try:
        if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
        user_uid = user["uid"]
        doc_ref = db.collection("predictions").document(user_uid)
        doc = doc_ref.get()
        if not doc.exists or "lands" not in doc.to_dict() or not doc.to_dict()["lands"]:
            raise HTTPException(status_code=404, detail="User has not completed onboarding.")
        all_plots_raw = doc.to_dict().get("lands")
        if not isinstance(all_plots_raw, list):
            print(f"⚠️ Data for user {user_uid} has a malformed 'lands' field. Type is {type(all_plots_raw)}. Treating as empty.")
            all_plots = []
        else:
            all_plots = all_plots_raw
        needs_update = False
        for plot in all_plots:
            if isinstance(plot, dict) and "id" not in plot:
                plot["id"] = str(uuid.uuid4())
                needs_update = True
                print(f"🩺 Migrating plot '{plot.get('name', 'N/A')}' for user {user_uid}, adding new ID.")
        if needs_update:
            doc_ref.update({"lands": all_plots})
            print(f"✅ Data migration complete for user {user_uid}.")
        weather_forecasts = []
        async with httpx.AsyncClient() as client:
            for plot in all_plots:
                if not isinstance(plot, dict): continue
                location = plot.get("location", {})
                lat, lon = location.get("latitude"), location.get("longitude")
                if lat and lon and WEATHERAPI_KEY:
                    weather_url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={lat},{lon}&days=3&aqi=yes&alerts=yes"
                    try:
                        resp = await client.get(weather_url)
                        resp.raise_for_status()
                        weather_forecasts.append(resp.json())
                    except Exception as e:
                        print(f"⚠️ Could not fetch weather for plot '{plot.get('name', 'N/A')}': {e}")
                        weather_forecasts.append({"error": f"Could not retrieve weather for {plot.get('name', 'N/A')}."})
                else:
                    weather_forecasts.append({"error": f"Location not set for {plot.get('name', 'N/A')}."})
        return JSONResponse(content={"all_plots": all_plots, "weather_forecasts": weather_forecasts})
    except Exception as e:
        print(f"🔥🔥🔥 CRITICAL ERROR IN DASHBOARD DATA API for user {user.get('uid')}: {e}")
        raise HTTPException(status_code=500, detail="A critical error occurred while fetching dashboard data.")

@app.post("/api/lands/delete")
async def delete_land_plot(request: LandDeleteRequest, user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    plot_id_to_delete = request.plot_id
    user_doc_ref = db.collection("predictions").document(user_uid)
    try:
        doc = user_doc_ref.get()
        if not doc.exists: raise HTTPException(status_code=404, detail="User document not found.")
        all_plots = doc.to_dict().get("lands", [])
        updated_plots = [plot for plot in all_plots if isinstance(plot, dict) and plot.get("id") != plot_id_to_delete]
        if len(updated_plots) == len(all_plots): raise HTTPException(status_code=404, detail="Plot ID not found.")
        user_doc_ref.update({"lands": updated_plots})
        print(f"✅ Plot {plot_id_to_delete} deleted for user {user_uid}.")
        return JSONResponse(content={"status": "success", "deleted_id": plot_id_to_delete})
    except Exception as e:
        print(f"❌ Error deleting plot {plot_id_to_delete} for {user_uid}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not delete plot: {e}")

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
    """
    UPGRADED: Now uses the user's location to get an AI-powered market price estimate.
    """
    if not model or plant_data_df is None: raise HTTPException(status_code=503, detail="Model not loaded.")
    
    user_uid = user["uid"]
    location_str = "India" # Default fallback
    if db:
        doc_ref = db.collection("predictions").document(user_uid)
        doc = doc_ref.get()
        if doc.exists and "lands" in doc.to_dict() and doc.to_dict()["lands"]:
            first_plot = doc.to_dict()["lands"][0]
            if isinstance(first_plot, dict):
                location_str = first_plot.get("location", {}).get("manual_entry", "India")

    image_bytes = await file.read()
    preprocessed_image = preprocess_image(image_bytes)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])
    plant_data = plant_data_df[plant_data_df['class_name'] == predicted_class_name].iloc[0]
    plant_name, disease_name, impact = plant_data['plant_name'], plant_data['disease_name'], plant_data['impact']
    predicted_yield = plant_data['base_yield_kg_per_acre'] * (1 - plant_data['yield_reduction_factor'])
    
    market_price = await get_ai_market_price_estimate(plant_name, location_str)
    total_harvest_value = predicted_yield * market_price
    
    ai_recs = await get_ai_recommendations(plant_name, disease_name, impact)
    
    prediction_details = {
        "prediction": predicted_class_name, "plant_name": plant_name, "disease_name": disease_name,
        "confidence": confidence, "predicted_yield_kg_per_acre": predicted_yield, "market_price_per_kg": market_price,
        "total_harvest_value": total_harvest_value, "recommendations": ai_recs, "impact": impact,
        "timestamp": datetime.utcnow()
    }
    if db:
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
