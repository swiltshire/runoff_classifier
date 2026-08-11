"""
Delete confirmed-orphan S3 canonical chip objects produced by
scripts/audit_canonical_chips.py.

Only intended to be used on a file produced by that script's
--verify_orphan_dims pass (outputs/chip_audit/orphans_confirmed_wrong_grid_{epsg}.txt),
NOT the raw orphans_{epsg}.txt list - the raw list is purely key-membership
based (orphan w.r.t. the --counties given at audit time) and has NOT been
confirmed to actually be stale/wrong-grid data. Deleting from the raw list
without dimension verification risks destroying valid chips that simply
fall outside whatever county subset was audited.

This script is a dry run by default: it always prints what WOULD be deleted
first. Nothing is deleted unless you pass --confirm.

Usage:
    # dry run - just shows counts/size, deletes nothing
    python scripts/delete_orphan_chips.py --file outputs/chip_audit/orphans_confirmed_wrong_grid_2968.txt

    # actually delete
    python scripts/delete_orphan_chips.py --file outputs/chip_audit/orphans_confirmed_wrong_grid_2968.txt --confirm
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.prepare_reprojected_tiles import S3_BUCKET, s3  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete confirmed-orphan S3 chip objects (dry run unless --confirm is passed)."
    )
    parser.add_argument(
        "--file", type=str, required=True,
        help="path to a confirmed-orphan key list file (one S3 key per line, tab-separated "
             "width/height columns from --verify_orphan_dims are ignored if present)",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="actually delete the objects. Without this flag, only a dry-run summary is printed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.file) as f:
        keys = [line.split("\t")[0].strip() for line in f if line.strip()]

    if not keys:
        print(f"No keys found in {args.file} - nothing to do.")
        return

    print(f"{len(keys)} key(s) loaded from {args.file}")
    print("First 5:")
    for k in keys[:5]:
        print(f"  {k}")

    if not args.confirm:
        print(
            f"\nDRY RUN - nothing deleted. Re-run with --confirm to permanently delete "
            f"these {len(keys)} object(s) from s3://{S3_BUCKET}/"
        )
        return

    deleted = 0
    for i in range(0, len(keys), 1000):  # boto3 delete_objects max 1000 keys/call
        batch = keys[i:i + 1000]
        resp = s3.delete_objects(
            Bucket=S3_BUCKET,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        errors = resp.get("Errors", [])
        if errors:
            print(f"  {len(errors)} error(s) in batch {i // 1000 + 1}:")
            for e in errors[:10]:
                print(f"    {e['Key']}: {e['Code']} {e.get('Message', '')}")
        deleted += len(batch) - len(errors)
        print(f"  deleted {deleted}/{len(keys)}...", flush=True)

    print(f"\nDone. Deleted {deleted}/{len(keys)} object(s) from s3://{S3_BUCKET}/")


if __name__ == "__main__":
    main()
