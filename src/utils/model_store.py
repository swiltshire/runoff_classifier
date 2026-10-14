# src/utils/model_store.py
"""Sync trained model artifacts between outputs/models/{model}/ and
s3://.../models/{model}/ so every state pipeline can reuse one trained model.

Copies are compared by SHA-256 content hash (stored as S3 object metadata on
upload), never by size: retrained checkpoints of the same architecture have
identical sizes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

from utils import states

MODEL_FILES = ("model_final.pth", "classes.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_sha256(s3, key: str) -> tuple[bool, Optional[str]]:
    """(exists, sha256-from-metadata or None)."""
    try:
        head = s3.head_object(Bucket=states.S3_BUCKET, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False, None
        raise
    return True, head.get("Metadata", {}).get("sha256")


def ensure_model_local(model: str = states.DEFAULT_MODEL) -> Path:
    """Make outputs/models/{model}/ match the S3 copy (download when missing or
    when the content hash differs); returns the local model_final.pth path."""
    s3 = boto3.client("s3")
    local_dir = states.model_dir(model)
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        key = f"{states.s3_model_prefix(model)}/{name}"
        local = local_dir / name
        exists, remote_sha = _remote_sha256(s3, key)
        if not exists:
            if local.exists():
                print(f"  {name}: local only (not on S3 at {key})")
                continue
            raise FileNotFoundError(f"model artifact missing locally and on S3: {states.s3_uri(key)}")
        if local.exists() and remote_sha and _sha256(local) == remote_sha:
            print(f"  {name}: already local (sha256 matches S3)")
            continue
        print(f"  {name}: downloading {states.s3_uri(key)}")
        s3.download_file(states.S3_BUCKET, key, str(local))
        if remote_sha and _sha256(local) != remote_sha:
            raise IOError(f"downloaded {local} does not match S3 sha256 metadata")
    return local_dir / "model_final.pth"


def upload_model(model: str = states.DEFAULT_MODEL, *, overwrite: bool = False) -> List[str]:
    """Upload outputs/models/{model}/{model_final.pth, classes.json} to S3 with
    a sha256 metadata tag. Refuses to replace a remote artifact with different
    content unless overwrite=True."""
    s3 = boto3.client("s3")
    local_dir = states.model_dir(model)
    uploaded = []
    for name in MODEL_FILES:
        local = local_dir / name
        if not local.exists():
            raise FileNotFoundError(f"missing local model artifact: {local}")
        key = f"{states.s3_model_prefix(model)}/{name}"
        local_sha = _sha256(local)
        exists, remote_sha = _remote_sha256(s3, key)
        if exists and remote_sha == local_sha:
            print(f"  {name}: already on S3 (same sha256)")
            continue
        if exists and not overwrite:
            raise FileExistsError(
                f"{states.s3_uri(key)} exists with different (or unverifiable) content - "
                f"pass overwrite=True to replace it"
            )
        print(f"  {name}: uploading -> {states.s3_uri(key)}")
        s3.upload_file(str(local), states.S3_BUCKET, key, ExtraArgs={"Metadata": {"sha256": local_sha}})
        uploaded.append(key)
    return uploaded
