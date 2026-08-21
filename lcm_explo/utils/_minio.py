from __future__ import annotations

from typing import TYPE_CHECKING

import boto3

from lcm_explo.utils._settings import settings

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


def get_minio_client() -> S3Client:
    client = boto3.client(
        "s3",
        endpoint_url=f"http://{settings.MINIO_ENDPOINT}",
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
    )
    try:
        client.head_bucket(Bucket=settings.MINIO_BUCKET)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=settings.MINIO_BUCKET)
    return client
