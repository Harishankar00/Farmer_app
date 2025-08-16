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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
from dotenv import load_dotenv
import json
import random

# Load environment variables from .env file
load_dotenv()

# ----------------------- Firebase Setup ----------------------- #

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

# ----------------------- FastAPI App ----------------------- #

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = None
class_names = []
plant_data_df = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in environment variables.")

oauth2_scheme = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ----------------------- Model and Data Loading ----------------------- #

@app.on_event("startup")
async def load_model_and_data():
    global model, class_names, plant_data_df
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'fine_tuned_model', 'best_fine_tuned_model.keras')
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")

        data_path = os.path.join(os.path.dirname(__file__), 'plant_data.csv')
        plant_data_df = pd.read_csv(data_path)
        print("✅ Plant data loaded successfully!")

        class_names.extend(plant_data_df['class_name'].tolist())

    except Exception as e:
        print(f"❌ Error during startup: {e}")
        model = None
        plant_data_df = None

def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_market_price(plant_name: str) -> float:
    prices = {
        'Apple': (80, 120),
        'Tomato': (20, 35),
        'Potato': (15, 25),
        'Grape': (50, 90),
        'Corn': (10, 15),
        'Blueberry': (200, 300),
        'Cherry': (150, 250),
        'Orange': (40, 60),
        'Peach': (80, 110),
        'Pepper': (30, 50),
        'Raspberry': (250, 400),
        'Soybean': (50, 80),
        'Squash': (15, 20),
        'Strawberry': (180, 280),
    }
    
    price_range = prices.get(plant_name, (10, 50))
    return random.uniform(price_range[0], price_range[1])

async def get_ai_recommendations(plant_name: str, disease_name: str, impact: str) -> dict:
    if not GEMINI_API_KEY:
        return {"error": "API key is missing."}

    prompt = (
        f"You are a helpful agricultural expert. A farmer has a {plant_name} plant with {disease_name}. "
        f"The disease impact is: '{impact}'. "
        "Provide a comprehensive set of recommendations for the farmer. The response must be a JSON object "
        "with three keys: 'harvest_recommendations', 'selling_strategies', and 'bargaining_tips'. "
        "Each key should have a brief, clear string value. The total response should be concise."
    )
    
    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    
    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 512,
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "harvest_recommendations": { "type": "STRING" },
                    "selling_strategies": { "type": "STRING" },
                    "bargaining_tips": { "type": "STRING" }
                }
            }
        }
    }
    
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(apiUrl, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # ✅ NEW: Print token usage information
            if 'usageMetadata' in result:
                usage = result['usageMetadata']
                print("--- Token Usage Report ---")
                print(f"Prompt tokens used: {usage.get('promptTokenCount', 0)}")
                print(f"Response tokens generated: {usage.get('candidatesTokenCount', 0)}")
                print(f"Total tokens used: {usage.get('totalTokenCount', 0)}")
                print("--------------------------")
            
            if 'candidates' in result and result['candidates']:
                return json.loads(result['candidates'][0]['content']['parts'][0]['text'])
            else:
                return {"error": "AI recommendations could not be generated."}
    except httpx.HTTPError as e:
        print(f"❌ HTTP Error calling Gemini API: {e}")
        return {"error": "Failed to get AI recommendations due to a connection error."}
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error from Gemini API: {e}")
        return {"error": "AI generated an invalid JSON response."}
    except Exception as e:
        print(f"❌ Unexpected error calling Gemini API: {e}")
        return {"error": "An unexpected error occurred while generating recommendations."}

# ----------------------- Routes ----------------------- #

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def serve_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def serve_history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})

@app.get("/api/history")
async def get_prediction_history(user: dict = Depends(get_current_user)):
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized.")

    user_uid = user.get("uid")
    
    try:
        doc_ref = db.collection("predictions").document(user_uid)
        doc = doc_ref.get()

        if doc.exists:
            history_data = doc.to_dict().get("history", [])
            
            for item in history_data:
                if 'timestamp' in item and isinstance(item['timestamp'], datetime):
                    item['timestamp'] = item['timestamp'].isoformat()
            
            return JSONResponse(content={"history": history_data})
        else:
            return JSONResponse(content={"history": []})
    except Exception as e:
        print(f"❌ Error fetching history for user {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history.")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    if not model or plant_data_df is None:
        raise HTTPException(status_code=500, detail="Server resources not loaded. Please check server logs.")

    image_bytes = await file.read()
    preprocessed_image = preprocess_image(image_bytes)

    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])

    user_uid = user.get("uid")
    user_email = user.get("email")

    print(f"🔍 Prediction by {user_uid}: {predicted_class_name} ({confidence:.2f})")
    
    plant_data = plant_data_df[plant_data_df['class_name'] == predicted_class_name].iloc[0]

    base_yield = plant_data['base_yield_kg_per_acre']
    reduction_factor = plant_data['yield_reduction_factor']
    predicted_yield = base_yield * (1 - reduction_factor)
    
    plant_name = plant_data['plant_name']
    disease_name = plant_data['disease_name']
    impact = plant_data['impact']
    
    market_price = get_market_price(plant_name)
    total_harvest_value = predicted_yield * market_price

    ai_recommendations = await get_ai_recommendations(plant_name, disease_name, impact)
    if "error" in ai_recommendations:
        ai_recommendations = {"harvest_recommendations": "No AI recommendations available.", "selling_strategies": "No AI recommendations available.", "bargaining_tips": "No AI recommendations available."}

    prediction_details = {
        "prediction": predicted_class_name,
        "plant_name": plant_name,
        "disease_name": disease_name,
        "confidence": confidence,
        "predicted_yield_kg_per_acre": predicted_yield,
        "market_price_per_kg": market_price,
        "total_harvest_value": total_harvest_value,
        "recommendations": ai_recommendations,
        "impact": impact,
        "timestamp": datetime.utcnow()
    }

    if db:
        try:
            user_doc_ref = db.collection("predictions").document(user_uid)
            user_doc_ref.set({
                "user_email": user_email,
                "history": firestore.ArrayUnion([prediction_details])
            }, merge=True)
            print("✅ Prediction appended to user's history array.")
        except Exception as e:
            print(f"⚠️ Failed to save prediction: {e}")
    
    return {
        "prediction": prediction_details['prediction'],
        "confidence": f"{prediction_details['confidence']:.2f}",
        "predicted_yield_kg_per_acre": f"{prediction_details['predicted_yield_kg_per_acre']:.2f}",
        "market_price_per_kg": f"{prediction_details['market_price_per_kg']:.2f}",
        "total_harvest_value": f"{prediction_details['total_harvest_value']:.2f}",
        "recommendations": prediction_details['recommendations'],
        "plant_name": prediction_details['plant_name'],
        "disease_name": prediction_details['disease_name'],
        "impact": prediction_details['impact']
    }

# ----------------------- Run App ----------------------- #

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
