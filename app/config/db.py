import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


@contextmanager
def get_connection():
    # each call opens its own connection/transaction, committed on success
    with engine.connect() as connection:
        with connection.begin():
            yield connection
