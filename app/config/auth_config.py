import os
import sys
from dotenv import load_dotenv

load_dotenv()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24 * 30  # 30 วัน

if not GOOGLE_CLIENT_ID or not JWT_SECRET_KEY:
    raise RuntimeError("Missing GOOGLE_CLIENT_ID or JWT_SECRET_KEY in environment variables.")