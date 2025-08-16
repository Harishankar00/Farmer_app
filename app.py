from fastapi import FastAPI, File, UploadFile, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import firebase_admin
from firebase_admin import credentials, auth
from google.cloud import firestore
from google.oauth2 import service_account
from datetime import datetime

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

# ----------------------- Model Loading ----------------------- #

@app.on_event("startup")
async def load_model_and_class_names():
    global model, class_names
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'fine_tuned_model', 'best_fine_tuned_model.keras')
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")

        class_names.extend([
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
            'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
            'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ])
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

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

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

    image_bytes = await file.read()
    preprocessed_image = preprocess_image(image_bytes)

    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])

    user_uid = user.get("uid")
    user_email = user.get("email")

    print(f"🔍 Prediction by {user_uid}: {predicted_class_name} ({confidence:.2f})")

    if db:
        try:
            user_doc_ref = db.collection("predictions").document(user_uid)
            user_doc_ref.set({
                "user_email": user_email,
                "history": firestore.ArrayUnion([{
                    "prediction": predicted_class_name,
                    "confidence": confidence,
                    "timestamp": datetime.utcnow()
                }])
            }, merge=True)

            print("✅ Prediction appended to user's history array.")
        except Exception as e:
            print(f"⚠️ Failed to save prediction: {e}")

    return {
        "prediction": predicted_class_name,
        "confidence": f"{confidence:.2f}"
    }

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
            
            # ✅ FIXED: Convert Firestore timestamp to a JSON-friendly string
            for item in history_data:
                if 'timestamp' in item and isinstance(item['timestamp'], datetime):
                    item['timestamp'] = item['timestamp'].isoformat()

            return JSONResponse(content={"history": history_data})
        else:
            return JSONResponse(content={"history": []})
    except Exception as e:
        print(f"❌ Error fetching history for user {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history.")


# ----------------------- Run App ----------------------- #

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)