from fastapi import FastAPI
from app.routes.analysis_route import router as analysis_router
from app.routes.exercise_route import router as exercise_router
from app.routes.admin_route import router as admin_router
from app.routes.history_route import router as history_router
from fastapi.staticfiles import StaticFiles 
from dotenv import load_dotenv
import os   

load_dotenv()

app = FastAPI(
    title="AngleSync Backend",
    version="1.0.0"
)

os.makedirs("outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="outputs"), name="static")
os.makedirs("data/outputs", exist_ok=True)
app.include_router(analysis_router)
app.include_router(exercise_router)
app.include_router(admin_router)
app.include_router(history_router)

@app.get("/")
def root():
    return {
        "message": "AngleSync Backend Running"
    }
