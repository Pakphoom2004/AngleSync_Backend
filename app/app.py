from fastapi import FastAPI
from routes.analysis_routes import router as analysis_router
from routes.exercise_route import router as exercise_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AngleSync Backend",
    version="1.0.0"
)

app.include_router(analysis_router)
app.include_router(exercise_router)

@app.get("/")
def root():
    return {
        "message": "AngleSync Backend Running"
    }