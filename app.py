import os
import io
import uvicorn
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, auth, firestore
from google.cloud.firestore_v1.transforms import Sentinel
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
import uuid
import asyncio

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

# --- UPGRADED Pydantic Models for Multi-Plot/Multi-Crop Structure ---
class CropData(BaseModel):
    crop_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    crop_name: str
    sowing_date: str
    ai_workflow: Optional[dict] = None
    progress_log: Optional[List[str]] = []
    price_trend_cache: Optional[list] = None
    price_trend_timestamp: Optional[datetime] = None

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
    crops: List[CropData]

class LandDataIn(BaseModel):
    name: str
    area: float
    unit: str
    location: Location
    crops: List[CropData]

class LandUpdateRequest(BaseModel):
    plot_id: str
    updated_plot_data: LandData

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

# --- UPGRADED AI GURU BRAIN (HELPER FUNCTIONS) ---

async def generate_ai_workflow(crop_name: str, location_str: str, sowing_date: str) -> dict:
    if not GEMINI_API_KEY: return {"error": "AI key not configured."}
    prompt = (
        f"You are a world-class agronomist specializing in the climate of India. A beginner farmer in '{location_str}' "
        f"has just planted '{crop_name}' on '{sowing_date}'. Generate a complete, week-by-week farming schedule from sowing to harvest. "
        "The output MUST be a valid JSON object. The keys must be week ranges (e.g., 'Week 1-2', 'Week 3-4'). "
        "Each value must be an object containing detailed, beginner-friendly instructions for 'irrigation', 'fertilizer', 'pest_control', and 'general_tasks' for that specific week. "
        "Keep the instructions concise, practical, and easy to understand."
    )
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            print(f"✅ AI Workflow generated for '{crop_name}' in '{location_str}'.")
            return json.loads(result['candidates'][0]['content']['parts'][0]['text'])
    except Exception as e:
        print(f"❌ Error generating AI workflow: {e}")
        return {"error": "Could not generate an AI farming plan for this crop."}

async def get_ai_price_trend(user_uid: str, all_plots: list, plot_id: str, crop_id: str, crop_name: str, location_str: str) -> list:
    if not GEMINI_API_KEY: return []

    target_plot = next((p for p in all_plots if p.get("id") == plot_id), None)
    target_crop = next((c for c in target_plot.get("crops", []) if c.get("crop_id") == crop_id), None) if target_plot else None

    if target_crop and target_crop.get("price_trend_cache") and target_crop.get("price_trend_timestamp"):
        cached_time = datetime.fromisoformat(target_crop["price_trend_timestamp"])
        if datetime.now() - cached_time < timedelta(hours=24):
            print(f"✅ Using cached price trend for '{crop_name}'.")
            return target_crop["price_trend_cache"]

    print(f"Fetching new AI Price Trend for '{crop_name}' from API.")
    prompt = (
        f"You are an agricultural market analyst for the region of '{location_str}'. "
        f"Generate a realistic JSON array of 12 estimated weekly market prices per kg in INR for '{crop_name}' over the last 3 months. "
        "The output MUST be a valid JSON array of objects. Each object must have two keys: 'week' (formatted as 'YYYY-Www', e.g., '2025-W30') and 'price' (a number). "
        "Ensure the price fluctuations are realistic for this crop and region."
    )
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=45)
            response.raise_for_status()
            result = response.json()
            price_data = json.loads(result['candidates'][0]['content']['parts'][0]['text'])

            if db:
                for plot in all_plots:
                    if plot.get("id") == plot_id:
                        for crop in plot.get("crops", []):
                            if crop.get("crop_id") == crop_id:
                                crop["price_trend_cache"] = price_data
                                crop["price_trend_timestamp"] = datetime.now().isoformat()
                                break
                doc_ref = db.collection("predictions").document(user_uid)
                doc_ref.update({"lands": all_plots})
                print(f"✅ Saved new price trend for '{crop_name}' to cache.")

            return price_data
    except Exception as e:
        print(f"❌ Error generating AI price trend: {e}")
        return []

# --- TRANSPLANTED PREDICTION HELPER FUNCTIONS (FROM YOUR OLD, WORKING CODE) ---
def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((256, 256))
    img_array = np.expand_dims(np.array(img), axis=0) / 255.0
    return img_array

async def get_ai_market_price_estimate(plant_name: str, location_str: str) -> float:
    if not GEMINI_API_KEY:
        print("⚠️ Gemini API key missing, falling back to random price.")
        return random.uniform(10, 100)

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

            ## PERMANENT TOKEN USAGE REPORTING
            if 'usageMetadata' in result:
                usage = result['usageMetadata']
                print("\n--- 🧠 AI Price Estimator Token Report ---")
                print(f"   Prompt Tokens: {usage.get('promptTokenCount', 'N/A')}")
                print(f"   Response Tokens: {usage.get('candidatesTokenCount', 'N/A')}")
                print(f"   TOTAL TOKENS USED: {usage.get('totalTokenCount', 'N/A')}")
                print("-----------------------------------------\n")

            price_data = json.loads(result['candidates'][0]['content']['parts'][0]['text'])
            price = float(price_data.get("estimated_price_inr", random.uniform(10, 100)))
            print(f"🧠 AI Price Estimate for {plant_name} in {location_str}: ₹{price:.2f}")
            return price
    except Exception as e:
        print(f"❌ Error getting AI price estimate: {e}. Falling back to random price.")
        return random.uniform(10, 100)

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

            ## PERMANENT TOKEN USAGE REPORTING
            if 'usageMetadata' in result:
                usage = result['usageMetadata']
                print("\n--- 🌿 AI Recommendations Token Report ---")
                print(f"   Prompt Tokens: {usage.get('promptTokenCount', 'N/A')}")
                print(f"   Response Tokens: {usage.get('candidatesTokenCount', 'N/A')}")
                print(f"   TOTAL TOKENS USED: {usage.get('totalTokenCount', 'N/A')}")
                print("-----------------------------------------\n")

            return json.loads(result['candidates'][0]['content']['parts'][0]['text'])
    except Exception as e:
        print(f"❌ Error calling Gemini API for recommendations: {e}")
        return {"error": "Failed to get AI recommendations."}

# --- HTML PAGE SERVING ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def serve_root_as_login(request: Request): return templates.TemplateResponse("index.html", {"request": request})

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

@app.get("/api/plots")
async def get_all_plots(user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    doc_ref = db.collection("predictions").document(user_uid)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("lands", [])
    return []

@app.post("/api/plots/update")
async def update_plot(request: LandUpdateRequest, user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    plot_id_to_update = request.plot_id
    updated_data = request.updated_plot_data.dict(exclude_none=True)
    
    doc_ref = db.collection("predictions").document(user_uid)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User document not found.")
        
    all_plots = doc.to_dict().get("lands", [])
    plot_found = False
    for i, plot in enumerate(all_plots):
        if plot.get("id") == plot_id_to_update:
            for j, crop in enumerate(updated_data.get("crops", [])):
                if "price_trend_cache" not in crop and len(plot["crops"]) > j:
                    crop["price_trend_cache"] = plot["crops"][j].get("price_trend_cache")
                    crop["price_trend_timestamp"] = plot["crops"][j].get("price_trend_timestamp")
            all_plots[i] = updated_data
            plot_found = True
            break
    
    if not plot_found:
        raise HTTPException(status_code=404, detail="Plot not found.")
        
    doc_ref.update({"lands": all_plots})
    print(f"✅ Plot {plot_id_to_update} updated for user {user_uid}.")
    return JSONResponse(content={"status": "success", "updated_id": plot_id_to_update})

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
        except Exception as e:
            raise HTTPException(status_code=500, detail="An error occurred while searching for the location.")

@app.get("/api/user-status")
async def get_user_status(user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    doc_ref = db.collection("predictions").document(user_uid)
    doc = doc_ref.get()
    has_onboarded = doc.exists and "lands" in doc.to_dict() and len(doc.to_dict().get("lands", [])) > 0
    return JSONResponse(content={"has_onboarded": has_onboarded})

@app.post("/api/onboard")
async def handle_onboarding(land_data: LandDataIn, user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    user_doc_ref = db.collection("predictions").document(user_uid)
    
    location_str = land_data.location.manual_entry or f"{land_data.location.latitude}, {land_data.location.longitude}"
    
    crop_tasks = [generate_ai_workflow(crop.crop_name, location_str, crop.sowing_date) for crop in land_data.crops]
    ai_workflows = await asyncio.gather(*crop_tasks)
    
    enriched_crops = []
    for i, crop in enumerate(land_data.crops):
        new_crop_data = CropData(
            crop_name=crop.crop_name, 
            sowing_date=crop.sowing_date, 
            ai_workflow=ai_workflows[i]
        )
        enriched_crops.append(new_crop_data.dict())

    new_plot_data = LandData(name=land_data.name, area=land_data.area, unit=land_data.unit, location=land_data.location, crops=enriched_crops).dict()

    try:
        user_doc_ref.set({"lands": firestore.ArrayUnion([new_plot_data])}, merge=True)
        print(f"✅ New plot with AI-powered crops saved for user {user_uid}.")
        return JSONResponse(content={"status": "success"}, status_code=201)
    except Exception as e:
        print(f"❌ Error saving new plot for {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not save new plot details.")

@app.get("/api/dashboard-data")
async def get_dashboard_data(user: dict = Depends(get_current_user), plot_id: Optional[str] = None, crop_id: Optional[str] = None):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    doc_ref = db.collection("predictions").document(user_uid)
    doc = doc_ref.get()
    if not doc.exists or "lands" not in doc.to_dict() or not doc.to_dict().get("lands"):
        raise HTTPException(status_code=404, detail="User has not completed onboarding.")
    
    all_plots = doc.to_dict().get("lands", [])
    
    target_plot = None
    if plot_id:
        target_plot = next((p for p in all_plots if p.get("id") == plot_id), None)
    if not target_plot and all_plots:
        target_plot = all_plots[0]
    
    if not target_plot:
        raise HTTPException(status_code=404, detail="No plot data found.")

    target_plot_id = target_plot.get("id")
    location_str = target_plot.get("location", {}).get("manual_entry", "India")
    
    price_trend_data = []
    target_crop = None
    if target_plot.get("crops"):
        if crop_id:
            target_crop = next((c for c in target_plot["crops"] if c.get("crop_id") == crop_id), None)
        if not target_crop and target_plot["crops"]:
            target_crop = target_plot["crops"][0]
        
        if target_crop:
            target_crop_id = target_crop.get("crop_id")
            target_crop_name = target_crop.get("crop_name")
            price_trend_data = await get_ai_price_trend(user_uid, all_plots, target_plot_id, target_crop_id, target_crop_name, location_str)

    temp_trend_data, weather_forecast = [], None
    lat = target_plot.get("location", {}).get("latitude")
    lon = target_plot.get("location", {}).get("longitude")
    if lat and lon and WEATHERAPI_KEY:
        try:
            async with httpx.AsyncClient() as client:
                forecast_url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={lat},{lon}&days=4"
                resp_forecast = await client.get(forecast_url)
                resp_forecast.raise_for_status()
                weather_forecast = resp_forecast.json()

                history_tasks = []
                for i in range(3, 0, -1):
                    past_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                    history_url = f"http://api.weatherapi.com/v1/history.json?key={WEATHERAPI_KEY}&q={lat},{lon}&dt={past_date}"
                    history_tasks.append(client.get(history_url))
                
                history_responses = await asyncio.gather(*history_tasks)
                
                past_temps = []
                for resp_history in history_responses:
                    resp_history.raise_for_status()
                    day_data = resp_history.json()["forecast"]["forecastday"][0]
                    past_temps.append({"date": day_data["date"], "max_temp": day_data["day"]["maxtemp_c"], "min_temp": day_data["day"]["mintemp_c"]})

                future_temps = [
                    {"date": day["date"], "max_temp": day["day"]["maxtemp_c"], "min_temp": day["day"]["mintemp_c"]}
                    for day in weather_forecast.get("forecast", {}).get("forecastday", [])
                ]
                temp_trend_data = past_temps + future_temps
        except Exception as e:
            print(f"⚠️ Could not fetch weather for dashboard graph: {e}")
            weather_forecast = {"error": "Could not retrieve weather."}

    return JSONResponse(content={
        "all_plots": all_plots,
        "weather_forecast": weather_forecast,
        "price_trend": price_trend_data,
        "temp_trend": temp_trend_data
    })


@app.post("/api/plots/delete")
async def delete_land_plot(request: LandDeleteRequest, user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    plot_id_to_delete = request.plot_id
    user_doc_ref = db.collection("predictions").document(user_uid)
    try:
        doc = user_doc_ref.get()
        if not doc.exists: raise HTTPException(status_code=404, detail="User document not found.")
        all_plots = doc.to_dict().get("lands", [])
        original_length = len(all_plots)
        updated_plots = [p for p in all_plots if p.get("id") != plot_id_to_delete]
        if len(updated_plots) == original_length: raise HTTPException(status_code=404, detail="Plot ID not found.")
        
        user_doc_ref.update({"lands": updated_plots})
        return JSONResponse(content={"status": "success", "deleted_id": plot_id_to_delete})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete plot: {e}")


@app.get("/api/history")
async def get_prediction_history(user: dict = Depends(get_current_user)):
    if not db: raise HTTPException(status_code=503, detail="Database service unavailable.")
    user_uid = user["uid"]
    doc = db.collection("predictions").document(user_uid).get()
    if doc.exists:
        history = doc.to_dict().get("history", [])
        ## MISSION FIX: This is the "Translation" fix for the history page crash.
        # It translates the fancy Firestore timestamp into a simple string.
        for item in history:
            if 'timestamp' in item and isinstance(item['timestamp'], datetime):
                item['timestamp'] = item['timestamp'].isoformat()
        return JSONResponse(content={"history": history})
    return JSONResponse(content={"history": []})


# --- TRANSPLANTED PREDICT ENDPOINT (FROM YOUR OLD, WORKING CODE) ---
@app.post("/predict")
async def predict_image(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
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
    
    ## MISSION FIX: This is the "Polite Pause" to prevent the 429 error.
    print("Taking a polite 2-second pause before the next AI call...")
    await asyncio.sleep(2) 
    
    ai_recs = await get_ai_recommendations(plant_name, disease_name, impact)
    
    prediction_details = {
        "prediction": predicted_class_name, "plant_name": plant_name, "disease_name": disease_name,
        "confidence": confidence, "predicted_yield_kg_per_acre": predicted_yield, "market_price_per_kg": market_price,
        "total_harvest_value": predicted_yield * market_price, "recommendations": ai_recs, "impact": impact,
        "timestamp": datetime.utcnow() # Use a standard datetime object
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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
