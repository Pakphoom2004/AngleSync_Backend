import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

url = make_url(DATABASE_URL)
if url.drivername in ("postgres", "postgresql", "postgresql+psycopg"):
    url = url.set(drivername="postgresql+psycopg2")

engine = create_engine(url, pool_pre_ping=True)


@contextmanager
def get_connection():
    # each call opens its own connection/transaction, committed on success
    with engine.connect() as connection:
        with connection.begin():
            yield connection