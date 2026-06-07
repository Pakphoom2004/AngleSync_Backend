from fastapi import FastAPI
from app.routes.analysis_routes import router as analysis_router
from app.routes.exercise_route import router as exercise_router
from fastapi.staticfiles import StaticFiles 
from dotenv import load_dotenv
import os   

load_dotenv()

app = FastAPI(
    title="AngleSync Backend",
    version="1.0.0"
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.include_router(analysis_router)
app.include_router(exercise_router)

@app.get("/")
def root():
    return {
        "message": "AngleSync Backend Running"
    }