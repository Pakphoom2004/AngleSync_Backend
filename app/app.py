from fastapi import FastAPI
from app.routes.analysis_route import router as analysis_router
from app.routes.exercise_route import router as exercise_router
from app.routes.admin_route import router as admin_router
from app.routes.history_route import router as history_router
from fastapi.staticfiles import StaticFiles 
from dotenv import load_dotenv
import os
from app.routes.auth_route import router as auth_router

load_dotenv()

app = FastAPI(
    title="AngleSync Backend",
    version="1.0.0"
)

app.include_router(analysis_router)
app.include_router(exercise_router)
app.include_router(admin_router)
app.include_router(history_router)
app.include_router(auth_router)

@app.get("/")
def root():
    return {
        "message": "AngleSync Backend Running"
    }
