from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import joblib
import numpy as np
import uvicorn

app = FastAPI()

# ============================================================
# 🔧 Setup Static and Templates
# ============================================================
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ============================================================
# 🧠 Load Pre-trained Seasonal Models
# ============================================================
models = {
    "winter": joblib.load("Data/xgb_winter.joblib"),
    "spring": joblib.load("Data/xgb_spring.joblib"),
    "summer": joblib.load("Data/xgb_summer.joblib"),
    "fall": joblib.load("Data/xgb_fall.joblib"),
}

# ============================================================
# 📋 Seasonal Feature Lists (exactly as your data)
# ============================================================
season_features = {
    "winter": [
        "geff_reference_w_m2",
        "geff_test_w_m2",
        "module_surface_temperature2_degree_centigrade",
        "module_surface_temperature1_degree_centigrade",
        "temperature_reference_cell_deg_c",
        "temperature_test_deg_c",
        "ambient_temp._degree_centigrade",
        "relative_humidity",
        "soiling_loss_index_isc",
        "soiling_loss_index_geff",
        "wind_speed_m_s",
    ],
    "spring": [
        "geff_reference_w_m2",
        "geff_test_w_m2",
        "module_surface_temperature2_degree_centigrade",
        "temperature_test_deg_c",
        "module_surface_temperature1_degree_centigrade",
        "temperature_reference_cell_deg_c",
        "ambient_temp._degree_centigrade",
        "relative_humidity",
        "wind_speed_m_s",
        "soiling_loss_index_isc",
        "soiling_loss_index_geff",
    ],
    "summer": [
        "temperature_test_deg_c",
        "temperature_reference_cell_deg_c",
        "module_surface_temperature2_degree_centigrade",
        "module_surface_temperature1_degree_centigrade",
        "ambient_temp._degree_centigrade",
        "relative_humidity",
        "wind_speed_m_s",
        "soiling_loss_index_geff",
        "soiling_loss_index_isc",
    ],
    "fall": [
        "module_surface_temperature2_degree_centigrade",
        "module_surface_temperature1_degree_centigrade",
        "temperature_reference_cell_deg_c",
        "temperature_test_deg_c",
        "ambient_temp._degree_centigrade",
        "relative_humidity",
        "soiling_loss_index_geff",
        "soiling_loss_index_isc",
        "wind_speed_m_s",
    ],
}

# ============================================================
# 🏠 Homepage
# ============================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ============================================================
# 🔄 Get Features for Selected Season
# ============================================================
@app.post("/features")
async def get_features(season: str = Form(...)):
    """Return the features based on selected season"""
    if season not in season_features:
        return JSONResponse({"error": "Invalid season"}, status_code=400)
    return JSONResponse({"features": season_features[season]})


# ============================================================
# 🔮 Predict Endpoint
# ============================================================
@app.post("/predict")
async def predict(
    season: str = Form(...),
    features: list[float] = Form(...)
):
    """Run prediction using selected seasonal model"""
    if season not in models:
        return JSONResponse({"error": "Invalid season"}, status_code=400)

    model = models[season]
    X = np.array([features])
    y_pred = model.predict(X)[0]

    result = round(float(y_pred), 4)

    return JSONResponse({
        "result": result,
        "target": "control_ppc_active_power_1m",
        "season": season.capitalize()
    })


# ============================================================
# 🚀 Run FastAPI (Dev Mode)
# ============================================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
