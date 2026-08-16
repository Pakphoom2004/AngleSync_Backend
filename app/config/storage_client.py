import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

GARAGE_S3_ENDPOINT = os.getenv("GARAGE_S3_ENDPOINT")
GARAGE_S3_REGION = os.getenv("GARAGE_S3_REGION", "garage")
GARAGE_ACCESS_KEY_ID = os.getenv("GARAGE_ACCESS_KEY_ID")
GARAGE_SECRET_ACCESS_KEY = os.getenv("GARAGE_SECRET_ACCESS_KEY")
GARAGE_BUCKET_NAME = os.getenv("GARAGE_BUCKET_NAME")
GARAGE_PUBLIC_BASE_URL = os.getenv("GARAGE_PUBLIC_BASE_URL")


def _with_scheme(host_or_url: str) -> str:
    if not host_or_url:
        return host_or_url
    return host_or_url if host_or_url.startswith("http") else f"https://{host_or_url}"


s3_client = boto3.client(
    "s3",
    endpoint_url=_with_scheme(GARAGE_S3_ENDPOINT),
    aws_access_key_id=GARAGE_ACCESS_KEY_ID,
    aws_secret_access_key=GARAGE_SECRET_ACCESS_KEY,
    region_name=GARAGE_S3_REGION,
    # Garage requires path-style addressing instead of AWS virtual-hosted style
    config=Config(s3={"addressing_style": "path"}),
)


def upload_bytes(object_key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    s3_client.put_object(
        Bucket=GARAGE_BUCKET_NAME,
        Key=object_key,
        Body=data,
        ContentType=content_type,
    )


def upload_file(object_key: str, file_path: str, content_type: str = "application/octet-stream") -> None:
    with open(file_path, "rb") as f:
        upload_bytes(object_key, f.read(), content_type)


def remove_object(object_key: str) -> None:
    s3_client.delete_object(Bucket=GARAGE_BUCKET_NAME, Key=object_key)


def get_public_url(object_key: str) -> str:
    # the website domain resolves the bucket by Host header, so no bucket segment in the path
    base = _with_scheme(GARAGE_PUBLIC_BASE_URL).rstrip("/")
    return f"{base}/{object_key}"
