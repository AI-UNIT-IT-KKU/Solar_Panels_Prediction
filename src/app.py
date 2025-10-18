from fastapi import FastAPI, Form                   
from fastapi.responses import JSONResponse, HTMLResponse  
from fastapi.staticfiles import StaticFiles         
from fastapi.templating import Jinja2Templates    
from fastapi.requests import Request              
import joblib                                     
import numpy as np                                
import uvicorn                                     

app = FastAPI()                                    # Initialize FastAPI application instance

# Mount static folder at /static URL path so templates can load CSS/JS/images
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates") # Point to templates folder for HTML pages

# Load the pre-trained XGBoost model from disk (PCC-based feature set)
model = joblib.load("Data/xgb_best_pcc.joblib")

@app.get("/", response_class=HTMLResponse)         # HTTP GET endpoint for homepage
def home(request: Request):                        # Inject Request for Jinja2 to render properly
    # Render the index.html template and pass the request context
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")                              # HTTP POST endpoint for predictions
async def predict(                                 # Async handler to process form inputs
    feature1: float = Form(...),                   # Parse form field as float (required)
    feature2: float = Form(...),
    feature3: float = Form(...),
    feature4: float = Form(...),
    feature5: float = Form(...),
    feature6: float = Form(...),
    feature7: float = Form(...),
    feature8: float = Form(...),
    feature9: float = Form(...),
    feature10: float = Form(...)
):
    # Build a 2D NumPy array with one row (the sample) and 10 features (columns)
    X = np.array([[feature1, feature2, feature3, feature4, feature5,
                   feature6, feature7, feature8, feature9, feature10]])
    # Run model inference and take the scalar prediction
    prediction = model.predict(X)[0]
    # Return JSON with a rounded numeric result
    return JSONResponse({"result": round(float(prediction), 2)})

if __name__ == "__main__":                         # Entrypoint when running `python app.py`
    # Start the ASGI server with auto-reload for development convenience
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
