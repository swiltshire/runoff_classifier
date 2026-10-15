"""
One-time migration from the single-state (Indiana-only) layout to the
state-first layout defined in src/utils/states.py. Run on SageMaker from the
project root, with every pipeline job/notebook kernel STOPPED.

S3 phases (each is resumable; state is kept in outputs/_migration/):

    python scripts/migrate_state_layout.py s3-inventory
        Freeze the exact source objects (key + ETag + size) of the legacy
        layout and compute their new keys. Aborts on any
        unmapped legacy key, destination collision, archived (Glacier)
        object, or pre-existing destination object.
    python scripts/migrate_state_layout.py s3-copy [--workers 64]
        Server-side copy into the new keys, conditional on each source still
        having its inventoried ETag (If-Match), so a source that changed
        after the inventory fails loudly instead of being copied.
        Content is verified per object: sources with a full-object CRC are
        copied with the same algorithm and the checksums must match; sources
        with only a composite (multipart) checksum are verified by size.
        Journaled - rerun to resume.
    python scripts/migrate_state_layout.py s3-verify
        Fresh listing of the destination: every planned object present with
        the right size and a successful journaled copy.
    python scripts/migrate_state_layout.py s3-manifests [--dry-run]
        Rewrite every county manifest LAST (after s3-verify passed): each
        legacy chip key must map exactly once to a verified IN/ key.
    python scripts/migrate_state_layout.py s3-delete-old --confirm-acceptance
        ONLY after the acceptance checks on the new layout pass: normal-delete
        the legacy objects (refuses if any changed since the inventory). Old
        versions then remain until an admin purges them / lifecycle expires.

Local phases:

    python scripts/migrate_state_layout.py local-apply [--dry-run]
        Journaled renames into data/{STATE}/, data/shared/, data/training/,
        outputs/models/, outputs/{STATE}/... Never overwrites or merges.
    python scripts/migrate_state_layout.py local-reverse
        Undo local-apply by replaying its journal backwards.
    python scripts/migrate_state_layout.py local-cleanup [--apply]
        Delete caches invalidated by the move (training VRT, native mosaic
        VRTs, AOI mask caches, .reference_crs). Dry-run unless --apply.

    python scripts/migrate_state_layout.py model-upload [--model train_multicounty]
        Upload outputs/models/{model}/{model_final.pth,classes.json} to S3.

    python scripts/migrate_state_layout.py compare-detections OLD.gpkg NEW.gpkg
        Acceptance check: per-class counts + tolerance-matched detections.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import states  # noqa: E402

BUCKET = states.S3_BUCKET
ROOT = states.S3_ROOT + "/"
LEGACY_STATE = "IN"
NEW_TOPS = tuple(f"{t}/" for t in (*states.STATES, "shared", "models"))
MIGRATION_DIR = PROJECT_ROOT / "outputs" / "_migration"
INVENTORY = MIGRATION_DIR / "s3_inventory.json"
COPY_JOURNAL = MIGRATION_DIR / "s3_copy_journal.jsonl"
VERIFY_REPORT = MIGRATION_DIR / "s3_verify_report.json"
MANIFEST_JOURNAL = MIGRATION_DIR / "s3_manifest_journal.json"
LOCAL_JOURNAL = MIGRATION_DIR / "local_journal.json"

FULL_CHECKSUM_ALGOS = ("CRC64NVME", "CRC32", "CRC32C", "SHA1", "SHA256")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[migrate {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows: a scanner can briefly hold a just-written file open
            if attempt == 9:
                raise
            time.sleep(0.2)


def s3_client(workers: int = 16):
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(
        max_pool_connections=workers + 8,
        retries={"max_attempts": 10, "mode": "adaptive"},
    ))


# =====================================================================
# Legacy -> new key mapping (keys relative to ROOT)
# =====================================================================

class Unmapped(Exception):
    pass


def map_legacy_key(rel: str) -> Tuple[str, Optional[str]]:
    """Return (kind, new_rel) for a legacy key relative to ROOT.
    kind: "copy" (copy as-is to new_rel), "manifest" (rewritten later),
    "marker" (zero-byte folder placeholder; not copied, deleted with the rest)."""
    parts = rel.split("/")
    if rel == "" or rel.endswith("/"):
        return "marker", None
    top = parts[0]
    if top == "canonical_mosaic":
        if len(parts) == 3 and parts[1] == "manifests" and parts[2].endswith(".json"):
            return "manifest", f"{LEGACY_STATE}/canonical_mosaic/manifests/{parts[2]}"
        if len(parts) == 4 and parts[2] == "chips" and parts[1].isdigit():
            return "copy", f"{LEGACY_STATE}/{rel}"
    elif top == "raw_tile_archive" and len(parts) == 3:
        return "copy", f"{LEGACY_STATE}/{rel}"
    elif top == "training" and len(parts) == 4 and parts[1] == parts[2]:
        # training/{county}/{county}/{file} -> IN/training_labels/{county}/{file}
        return "copy", f"{LEGACY_STATE}/training_labels/{parts[1]}/{parts[3]}"
    elif top == "inference" and len(parts) == 3:
        return "copy", f"{LEGACY_STATE}/{rel}"
    elif top == "nhd_masks" and len(parts) == 3 and parts[1] == "IN":
        return "copy", f"{LEGACY_STATE}/nhd_mask/{parts[2]}"
    elif rel == "training_county_imagery_years.csv":
        return "copy", f"{LEGACY_STATE}/{rel}"
    elif rel in ("classes_w_background.json", "classes_no_background.json", "feature_size_threshholds.csv"):
        return "copy", f"shared/{rel}"
    raise Unmapped(rel)


# =====================================================================
# S3: inventory
# =====================================================================

def _list_live(s3, prefix: str, delimiter: Optional[str] = None):
    """Yield (objects, common_prefixes) pages of current objects under prefix."""
    kwargs = {"Bucket": BUCKET, "Prefix": prefix}
    if delimiter:
        kwargs["Delimiter"] = delimiter
    for page in s3.get_paginator("list_objects_v2").paginate(**kwargs):
        yield page.get("Contents", []), [cp["Prefix"] for cp in page.get("CommonPrefixes", [])]


def cmd_s3_inventory(args) -> None:
    if INVENTORY.exists() and not args.force:
        raise SystemExit(f"{INVENTORY} already exists (frozen). Use --force only if no copy has started.")
    if COPY_JOURNAL.exists():
        raise SystemExit(f"{COPY_JOURNAL} exists - a copy already started from the existing inventory.")
    s3 = s3_client()

    # root-level objects + top-level prefixes
    items: List[Dict] = []
    tops: List[str] = []
    for live, cps in _list_live(s3, ROOT, delimiter="/"):
        items.extend(live)
        tops.extend(cps)
    legacy_tops = [t for t in tops if not t[len(ROOT):] in NEW_TOPS]
    log(f"top-level prefixes: legacy={[t[len(ROOT):] for t in legacy_tops]} "
        f"new={[t[len(ROOT):] for t in tops if t not in legacy_tops]}")
    for top in legacy_tops:
        n0 = len(items)
        for live, _ in _list_live(s3, top):
            items.extend(live)
        log(f"  {top[len(ROOT):]}: {len(items) - n0} live objects")

    plan, unmapped, archived = [], [], []
    for v in items:
        rel = v["Key"][len(ROOT):]
        try:
            kind, new_rel = map_legacy_key(rel)
        except Unmapped:
            unmapped.append(rel)
            continue
        if v.get("StorageClass", "STANDARD") in ("GLACIER", "DEEP_ARCHIVE"):
            archived.append(rel)
        plan.append({
            "kind": kind,
            "src_key": v["Key"],
            "size": v["Size"],
            "etag": v["ETag"],
            "storage_class": v.get("StorageClass", "STANDARD"),
            "dst_key": (ROOT + new_rel) if new_rel else None,
        })
    if unmapped:
        raise SystemExit(f"{len(unmapped)} unmapped legacy key(s), e.g. {unmapped[:10]} - extend map_legacy_key()")
    if archived:
        raise SystemExit(f"{len(archived)} archived object(s) need a Glacier restore first, e.g. {archived[:5]}")

    dsts = [p["dst_key"] for p in plan if p["dst_key"]]
    if len(dsts) != len(set(dsts)):
        dup = [k for k in set(dsts) if dsts.count(k) > 1][:5]
        raise SystemExit(f"destination collisions: {dup}")

    # destinations must not exist yet
    existing = set()
    for top in NEW_TOPS:
        for live, _ in _list_live(s3, ROOT + top):
            existing.update(v["Key"] for v in live)
    clash = sorted(set(dsts) & existing)
    if clash:
        raise SystemExit(f"{len(clash)} destination key(s) already exist, e.g. {clash[:5]}")
    if existing:
        log(f"note: {len(existing)} unrelated object(s) already under the new prefixes (left untouched)")

    by_kind: Dict[str, int] = {}
    for p in plan:
        by_kind[p["kind"]] = by_kind.get(p["kind"], 0) + 1
    total = sum(p["size"] for p in plan if p["kind"] == "copy")
    write_json_atomic(INVENTORY, {
        "created_at": now(), "bucket": BUCKET, "root": ROOT,
        "counts": by_kind, "copy_bytes": total, "items": plan,
    })
    log(f"inventory frozen -> {INVENTORY}: {by_kind}, {total / 1e12:.2f} TB to copy")


def load_inventory() -> Dict:
    if not INVENTORY.exists():
        raise SystemExit("no inventory - run s3-inventory first")
    return json.loads(INVENTORY.read_text())


# =====================================================================
# S3: copy
# =====================================================================

def _load_copy_journal() -> Dict[str, Dict]:
    done = {}
    if COPY_JOURNAL.exists():
        for line in COPY_JOURNAL.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["dst_key"]] = rec
    return done


def _full_checksum(head: Dict) -> Tuple[Optional[str], Optional[str]]:
    if head.get("ChecksumType") != "FULL_OBJECT":
        return None, None
    for algo in FULL_CHECKSUM_ALGOS:
        val = head.get(f"Checksum{algo}")
        if val:
            return algo, val
    return None, None


def _copy_one(s3, item: Dict) -> Dict:
    src = {"Bucket": BUCKET, "Key": item["src_key"]}
    head = s3.head_object(Bucket=BUCKET, Key=item["src_key"], IfMatch=item["etag"], ChecksumMode="ENABLED")
    if head["ContentLength"] != item["size"]:
        raise RuntimeError(f"source size changed since inventory: {item['src_key']}")
    algo, src_sum = _full_checksum(head)

    # never overwrite: a destination that already exists here was not
    # journaled by us -> adopt only if it is verifiably the same content
    try:
        dst_head = s3.head_object(Bucket=BUCKET, Key=item["dst_key"], ChecksumMode="ENABLED")
    except s3.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
            raise
        dst_head = None
    if dst_head is not None:
        _, dst_sum = _full_checksum(dst_head)
        same = dst_head["ContentLength"] == item["size"] and (src_sum is None or dst_sum == src_sum)
        if not same:
            raise RuntimeError(f"destination exists with different content: {item['dst_key']}")
        return {"dst_key": item["dst_key"], "size": item["size"], "method": "adopted-existing",
                "algo": algo, "src_checksum": src_sum, "dst_checksum": dst_sum}

    kwargs = {"Bucket": BUCKET, "Key": item["dst_key"], "CopySource": src, "MetadataDirective": "COPY",
              "CopySourceIfMatch": item["etag"]}
    if algo:
        kwargs["ChecksumAlgorithm"] = algo
    resp = s3.copy_object(**kwargs)
    dst_sum = resp.get("CopyObjectResult", {}).get(f"Checksum{algo}") if algo else None
    if algo and dst_sum != src_sum:
        raise RuntimeError(f"checksum mismatch after copy ({algo}): {item['src_key']} -> {item['dst_key']}")
    return {"dst_key": item["dst_key"], "size": item["size"],
            "method": f"checksum-{algo}" if algo else "size-only",
            "algo": algo, "src_checksum": src_sum, "dst_checksum": dst_sum}


def cmd_s3_copy(args) -> None:
    inv = load_inventory()
    todo_all = [p for p in inv["items"] if p["kind"] == "copy"]
    done = _load_copy_journal()
    todo = [p for p in todo_all if p["dst_key"] not in done]
    total_bytes = sum(p["size"] for p in todo)
    log(f"copy: {len(todo_all)} planned, {len(done)} already journaled, {len(todo)} to go "
        f"({total_bytes / 1e12:.2f} TB)")
    if args.dry_run or not todo:
        return
    s3 = s3_client(args.workers)
    lock = threading.Lock()
    failures = []
    t0 = time.time()
    copied = copied_bytes = 0
    MIGRATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(COPY_JOURNAL, "a", encoding="utf-8") as journal, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_copy_one, s3, p): p for p in todo}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                failures.append((p["src_key"], str(e)))
                continue
            with lock:
                journal.write(json.dumps(rec) + "\n")
                journal.flush()
                copied += 1
                copied_bytes += p["size"]
                if copied % 2000 == 0 or copied == len(todo):
                    rate = copied_bytes / max(time.time() - t0, 1) / 1e9
                    log(f"  {copied}/{len(todo)} copied, {copied_bytes / 1e12:.2f} TB, {rate:.2f} GB/s")
    if failures:
        for k, e in failures[:20]:
            log(f"  FAILED {k}: {e}")
        raise SystemExit(f"{len(failures)} copy failure(s) - rerun s3-copy to retry")
    log("copy complete")


# =====================================================================
# S3: verify
# =====================================================================

def cmd_s3_verify(args) -> None:
    inv = load_inventory()
    planned = {p["dst_key"]: p for p in inv["items"] if p["kind"] == "copy"}
    journal = _load_copy_journal()
    s3 = s3_client()
    present: Dict[str, int] = {}
    for top in NEW_TOPS:
        for live, _ in _list_live(s3, ROOT + top):
            for v in live:
                present[v["Key"]] = v["Size"]

    missing = [k for k in planned if k not in present]
    size_bad = [k for k, p in planned.items() if k in present and present[k] != p["size"]]
    unjournaled = [k for k in planned if k not in journal]
    methods: Dict[str, int] = {}
    for k in planned:
        m = journal.get(k, {}).get("method", "none")
        methods[m] = methods.get(m, 0) + 1
    ok = not (missing or size_bad or unjournaled)
    report = {
        "checked_at": now(), "inventory_created_at": inv["created_at"], "pass": ok,
        "planned": len(planned), "present": len(planned) - len(missing),
        "missing": missing[:100], "n_missing": len(missing),
        "size_mismatch": size_bad[:100], "n_size_mismatch": len(size_bad),
        "unjournaled": unjournaled[:100], "n_unjournaled": len(unjournaled),
        "verification_methods": methods,
    }
    write_json_atomic(VERIFY_REPORT, report)
    log(f"verify: pass={ok} planned={len(planned)} missing={len(missing)} "
        f"size_mismatch={len(size_bad)} unjournaled={len(unjournaled)} methods={methods}")
    if not ok:
        raise SystemExit(1)


def _require_verified(inv: Dict) -> None:
    if not VERIFY_REPORT.exists():
        raise SystemExit("run s3-verify first")
    rep = json.loads(VERIFY_REPORT.read_text())
    if not rep.get("pass") or rep.get("inventory_created_at") != inv["created_at"]:
        raise SystemExit("s3-verify has not passed for the current inventory")


# =====================================================================
# S3: manifests
# =====================================================================

def cmd_s3_manifests(args) -> None:
    inv = load_inventory()
    _require_verified(inv)
    copy_map = {p["src_key"]: p["dst_key"] for p in inv["items"] if p["kind"] == "copy"}
    manifests = [p for p in inv["items"] if p["kind"] == "manifest"]
    s3 = s3_client()
    expected_crs = states.require_canonical_crs(LEGACY_STATE)
    new_prefix = states.s3_canonical_prefix(LEGACY_STATE) + "/"
    done = json.loads(MANIFEST_JOURNAL.read_text()) if MANIFEST_JOURNAL.exists() else {}
    planned_out = []
    for p in manifests:
        if p["dst_key"] in done:
            continue
        body = s3.get_object(Bucket=BUCKET, Key=p["src_key"], IfMatch=p["etag"])["Body"].read()
        m = json.loads(body)
        problems = []
        stem = Path(p["src_key"]).stem
        if m.get("county") is None or states.safe_name(m["county"]) != stem:
            problems.append(f"county {m.get('county')!r} != file {stem!r}")
        if m.get("bucket") != BUCKET:
            problems.append(f"bucket {m.get('bucket')!r}")
        if m.get("canonical_crs") != expected_crs:
            problems.append(f"canonical_crs {m.get('canonical_crs')!r}")
        keys = m.get("keys", [])
        if len(keys) != len(set(keys)):
            problems.append("duplicate keys")
        new_keys = []
        for k in keys:
            nk = copy_map.get(k)
            if nk is None:
                problems.append(f"key not in verified copy set: {k}")
                break
            if not nk.startswith(new_prefix):
                problems.append(f"mapped key outside {new_prefix}: {nk}")
                break
            new_keys.append(nk)
        if problems:
            raise SystemExit(f"manifest {p['src_key']} failed validation: {problems}")
        new_m = dict(m)
        new_m["state"] = LEGACY_STATE
        new_m["keys"] = sorted(new_keys)
        new_m["migrated_at"] = now()
        new_m["migrated_from"] = {"key": p["src_key"], "etag": p["etag"]}
        planned_out.append((p, new_m))

    log(f"manifests: {len(manifests)} total, {len(done)} already written, {len(planned_out)} to write")
    if args.dry_run:
        for p, m in planned_out[:3]:
            log(f"  e.g. {p['dst_key']}: {len(m['keys'])} keys, first {m['keys'][:1]}")
        return
    for p, m in planned_out:
        try:
            s3.head_object(Bucket=BUCKET, Key=p["dst_key"])
            raise SystemExit(f"destination manifest already exists (not written by this migration): {p['dst_key']}")
        except s3.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                raise
        s3.put_object(Bucket=BUCKET, Key=p["dst_key"], Body=json.dumps(m, indent=2).encode("utf-8"),
                      ContentType="application/json")
        done[p["dst_key"]] = {"written_at": now(), "n_keys": len(m["keys"])}
        write_json_atomic(MANIFEST_JOURNAL, done)
    log("manifests written")


# =====================================================================
# S3: delete old
# =====================================================================

def cmd_s3_delete_old(args) -> None:
    if not args.confirm_acceptance:
        raise SystemExit("refusing: pass --confirm-acceptance once the new layout has passed acceptance checks")
    inv = load_inventory()
    _require_verified(inv)
    manifests = [p for p in inv["items"] if p["kind"] == "manifest"]
    done = json.loads(MANIFEST_JOURNAL.read_text()) if MANIFEST_JOURNAL.exists() else {}
    if any(p["dst_key"] not in done for p in manifests):
        raise SystemExit("not every manifest has been rewritten - run s3-manifests first")

    s3 = s3_client(32)
    pinned = {p["src_key"]: p["etag"] for p in inv["items"]}
    changed = []

    def check(key):
        try:
            h = s3.head_object(Bucket=BUCKET, Key=key)
        except s3.exceptions.ClientError:
            return None  # already gone
        return key if h.get("ETag") != pinned[key] else None

    with ThreadPoolExecutor(32) as ex:
        changed = [k for k in ex.map(check, pinned) if k]
    if changed:
        raise SystemExit(f"{len(changed)} legacy object(s) changed since the inventory (e.g. {changed[:5]}) - "
                         f"something wrote to the old layout; investigate before deleting")

    keys = list(pinned)
    log(f"deleting {len(keys)} legacy objects (normal delete; old versions remain until purged)")
    errors = []
    for i in range(0, len(keys), 1000):
        batch = [{"Key": k} for k in keys[i:i + 1000]]
        errors += s3.delete_objects(Bucket=BUCKET, Delete={"Objects": batch, "Quiet": True}).get("Errors", [])
    if errors:
        raise SystemExit(f"{len(errors)} delete error(s), e.g. {errors[:3]}")
    log("legacy layout deleted. Ask an admin to purge noncurrent versions under the legacy prefixes.")


# =====================================================================
# Local
# =====================================================================

def _local_ops() -> List[Tuple[str, str]]:
    """(src, dst) renames relative to PROJECT_ROOT, inner-most first."""
    ops = [
        ("data/counties", "data/IN/counties"),
        ("data/_canonical_chip_cache", "data/IN/_canonical_chip_cache"),
        ("data/nhd_masks/IN", "data/IN/nhd_mask"),
        ("data/training_county_imagery_years.csv", "data/IN/training_county_imagery_years.csv"),
        ("data/feature_size_threshholds.csv", "data/shared/feature_size_threshholds.csv"),
        ("data/county_boundaries", "data/shared/county_boundaries"),
        ("data/multi_county_labels.gpkg", "data/training/multi_county_labels.gpkg"),
        ("outputs/logs", "outputs/IN/logs"),
    ]
    out = PROJECT_ROOT / "outputs"
    for p in sorted(out.glob("classes_*.json")):
        ops.append((f"outputs/{p.name}", f"data/training/{p.name}"))
    # model dirs: anything under outputs/ holding checkpoints or per-county inference
    for d in sorted(out.iterdir()) if out.is_dir() else []:
        if not d.is_dir() or d.name in ("models", "_migration", "logs", "_native_mosaic_cache", *states.STATES):
            continue
        infer_dirs = sorted(x for x in d.glob("inferences_*") if x.is_dir())
        if not (list(d.glob("model_*.pth")) or infer_dirs):
            continue
        model = d.name
        for inf in infer_dirs:
            county = inf.name[len("inferences_"):]
            ops.append((f"outputs/{model}/{inf.name}", f"outputs/IN/inference/{model}/{states.safe_name(county)}"))
        if (d / "detection_counts_summary.csv").exists():
            ops.append((f"outputs/{model}/detection_counts_summary.csv",
                        f"outputs/IN/inference/{model}/detection_counts_summary.csv"))
        ops.append((f"outputs/{model}", f"outputs/models/{model}"))
    return ops


def _running_pipeline_procs() -> List[str]:
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    needles = ("prepare_canonical_mosaic", "run_statewide_inference", "scripts/inference.py",
               "scripts/train.py", "torch.distributed.run", "ipykernel")
    me = str(os.getpid())
    return [ln.strip() for ln in out.splitlines()[1:]
            if any(n in ln for n in needles) and not ln.strip().startswith(me + " ")]


def _nearest_existing(p: Path) -> Path:
    while not p.exists():
        p = p.parent
    return p


def cmd_local_apply(args) -> None:
    if LOCAL_JOURNAL.exists():
        journal = json.loads(LOCAL_JOURNAL.read_text())
        if journal.get("status") == "applied":
            raise SystemExit("local migration already applied (see local-reverse)")
        ops = [(o["src"], o["dst"]) for o in journal["ops"]]
        log("resuming partially-applied local migration from journal")
    else:
        ops = _local_ops()
        journal = {"created_at": now(), "status": "planned",
                   "ops": [{"src": s, "dst": d, "done": False} for s, d in ops]}

    procs = _running_pipeline_procs()
    if procs and not args.force:
        raise SystemExit("pipeline/notebook processes are running - stop them first (or --force):\n  "
                         + "\n  ".join(procs))

    problems, plan = [], []
    for rec in journal["ops"]:
        src, dst = PROJECT_ROOT / rec["src"], PROJECT_ROOT / rec["dst"]
        if rec["done"]:
            continue
        if src.exists() and dst.exists():
            problems.append(f"both exist: {rec['src']} and {rec['dst']}")
        elif src.exists():
            if os.stat(src).st_dev != os.stat(_nearest_existing(dst.parent)).st_dev:
                problems.append(f"different filesystems: {rec['src']} -> {rec['dst']}")
            plan.append(rec)
        elif dst.exists():
            rec["done"] = True   # already moved (interrupted before journaling)
        # neither exists: nothing to move
    if problems:
        raise SystemExit("refusing to migrate:\n  " + "\n  ".join(problems))

    for rec in plan:
        log(f"{'would move' if args.dry_run else 'move'}: {rec['src']} -> {rec['dst']}")
    if args.dry_run:
        _report_leftovers(planned=[r["src"] for r in plan])
        return

    write_json_atomic(LOCAL_JOURNAL, journal)
    for rec in plan:
        dst = PROJECT_ROOT / rec["dst"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(PROJECT_ROOT / rec["src"], dst)
        rec["done"] = True
        write_json_atomic(LOCAL_JOURNAL, journal)
    journal["status"] = "applied"
    journal["applied_at"] = now()
    write_json_atomic(LOCAL_JOURNAL, journal)
    log(f"local migration applied ({len(plan)} moves). Next: local-cleanup (dry run), then --apply.")
    _report_leftovers(planned=[])


def _report_leftovers(planned: List[str]) -> None:
    keep = {"data/IN", "data/IL", "data/shared", "data/training", "outputs/models", "outputs/IN",
            "outputs/IL", "outputs/_migration", "data/multi_county_training_mosaic.vrt",
            "outputs/_native_mosaic_cache", "outputs/.reference_crs", "data/NHDmask_Indiana", "data/nhd_masks"}
    left = []
    for top in ("data", "outputs"):
        d = PROJECT_ROOT / top
        if d.is_dir():
            for p in sorted(d.iterdir()):
                rel = f"{top}/{p.name}"
                if rel not in keep and rel not in planned:
                    left.append(rel)
    if left:
        log("left in place (not used by the new pipeline; review/delete manually): " + ", ".join(left))


def cmd_local_reverse(args) -> None:
    if not LOCAL_JOURNAL.exists():
        raise SystemExit("no local journal - nothing to reverse")
    journal = json.loads(LOCAL_JOURNAL.read_text())
    done = [r for r in journal["ops"] if r["done"]]
    for rec in reversed(done):
        src, dst = PROJECT_ROOT / rec["src"], PROJECT_ROOT / rec["dst"]
        if src.exists():
            raise SystemExit(f"cannot reverse: {rec['src']} exists again")
        if not dst.exists():
            raise SystemExit(f"cannot reverse: {rec['dst']} is missing")
    for rec in reversed(done):
        log(f"{'would restore' if args.dry_run else 'restore'}: {rec['dst']} -> {rec['src']}")
        if args.dry_run:
            continue
        src = PROJECT_ROOT / rec["src"]
        src.parent.mkdir(parents=True, exist_ok=True)
        os.rename(PROJECT_ROOT / rec["dst"], src)
        rec["done"] = False
        journal["status"] = "partially-reversed"
        write_json_atomic(LOCAL_JOURNAL, journal)
    if not args.dry_run:
        journal["status"] = "reversed"
        write_json_atomic(LOCAL_JOURNAL, journal)
        LOCAL_JOURNAL.rename(LOCAL_JOURNAL.with_name(
            f"local_journal_reversed_{datetime.now():%Y%m%d_%H%M%S_%f}.json"))
        log("local migration reversed")


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def cmd_local_cleanup(args) -> None:
    if not LOCAL_JOURNAL.exists() or json.loads(LOCAL_JOURNAL.read_text()).get("status") != "applied":
        raise SystemExit("run local-apply first")
    targets: List[Path] = [
        PROJECT_ROOT / "data/multi_county_training_mosaic.vrt",   # relative paths now stale; regenerated by prep
        PROJECT_ROOT / "outputs/_native_mosaic_cache",            # absolute raw-tile paths; regenerated per run
        PROJECT_ROOT / "outputs/.reference_crs",                  # replaced by per-state config
        PROJECT_ROOT / "data/NHDmask_Indiana",                    # pre-rename copy of the NHD mask
    ]
    for state in states.STATES:
        cdir = states.counties_dir(state)
        if cdir.is_dir():
            # AOI mask caches are keyed by the mask's absolute path, which moved
            targets += sorted(cdir.glob("*/*_mosaic_mask_*"))
    total = 0
    for t in targets:
        if not t.exists():
            continue
        sz = _size(t)
        total += sz
        log(f"{'delete' if args.apply else 'would delete'}: {t.relative_to(PROJECT_ROOT)} ({sz / 1e9:.1f} GB)")
        if args.apply:
            shutil.rmtree(t) if t.is_dir() else t.unlink()
    empty = PROJECT_ROOT / "data/nhd_masks"
    if empty.is_dir() and not any(empty.iterdir()):
        log(f"{'remove' if args.apply else 'would remove'} empty dir data/nhd_masks")
        if args.apply:
            empty.rmdir()
    log(f"{'freed' if args.apply else 'would free'} {total / 1e9:.1f} GB")


# =====================================================================
# Model
# =====================================================================

def cmd_model_upload(args) -> None:
    from utils.model_store import upload_model

    upload_model(args.model)


# =====================================================================
# Acceptance helper
# =====================================================================

def cmd_compare_detections(args) -> None:
    """Compare two detections gpkgs. Detections are paired per class by
    greatest overlap (IoU >= --iou); reports per-class counts, how closely
    paired detections agree, and the scores of unpaired ones (unpaired
    detections scored near the 0.75 threshold point to run-to-run GPU
    nondeterminism rather than different inputs)."""
    import geopandas as gpd

    a, b = gpd.read_file(args.old), gpd.read_file(args.new)
    if a.crs != b.crs and a.crs is not None and b.crs is not None:
        b = b.to_crs(a.crs)
    log(f"old: {len(a)} detections  new: {len(b)} detections")
    total_matched = 0
    min_iou, max_dscore = 1.0, 0.0
    for cls in sorted(set(a["classname"]) | set(b["classname"])):
        A = a[a["classname"] == cls].reset_index(drop=True)
        Bc = b[b["classname"] == cls].reset_index(drop=True)
        pairs = []
        if len(A) and len(Bc):
            for i, geom in enumerate(A.geometry):
                for j in Bc.sindex.query(geom, predicate="intersects"):
                    g2 = Bc.geometry.iloc[j]
                    union = geom.union(g2).area
                    iou = geom.intersection(g2).area / union if union else 0.0
                    if iou >= args.iou:
                        pairs.append((iou, i, int(j)))
        used_a, used_b, matched = set(), set(), []
        for iou, i, j in sorted(pairs, reverse=True):
            if i not in used_a and j not in used_b:
                used_a.add(i); used_b.add(j); matched.append((iou, i, j))
        total_matched += len(matched)
        dscores = [abs(float(A["score"].iloc[i]) - float(Bc["score"].iloc[j])) for _, i, j in matched]
        if matched:
            min_iou = min(min_iou, min(m[0] for m in matched))
            max_dscore = max(max_dscore, max(dscores))
        only_a = sorted(float(A["score"].iloc[i]) for i in range(len(A)) if i not in used_a)
        only_b = sorted(float(Bc["score"].iloc[j]) for j in range(len(Bc)) if j not in used_b)
        log(f"  {cls:18s} old={len(A):5d} new={len(Bc):5d} paired={len(matched):5d}"
            + (f"  min IoU={min(m[0] for m in matched):.3f} max |dscore|={max(dscores):.4f}" if matched else ""))
        if only_a:
            log(f"      old-only scores: {[round(x, 3) for x in only_a]}")
        if only_b:
            log(f"      new-only scores: {[round(x, 3) for x in only_b]}")
    log(f"paired {total_matched}/{len(a)} old and {total_matched}/{len(b)} new detections; "
        f"min IoU={min_iou:.3f}, max |score diff|={max_dscore:.4f}")
    if total_matched == len(a) == len(b) and min_iou >= 0.95 and max_dscore <= args.score_tol:
        log("IDENTICAL within tolerance")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare-detections"); p.add_argument("old"); p.add_argument("new")
    p.add_argument("--iou", type=float, default=0.5); p.add_argument("--score_tol", type=float, default=0.01)
    p.set_defaults(fn=cmd_compare_detections)
    p = sub.add_parser("s3-inventory"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_s3_inventory)
    p = sub.add_parser("s3-copy"); p.add_argument("--workers", type=int, default=64); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_s3_copy)
    p = sub.add_parser("s3-verify"); p.set_defaults(fn=cmd_s3_verify)
    p = sub.add_parser("s3-manifests"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_s3_manifests)
    p = sub.add_parser("s3-delete-old"); p.add_argument("--confirm-acceptance", action="store_true"); p.set_defaults(fn=cmd_s3_delete_old)
    p = sub.add_parser("local-apply"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_local_apply)
    p = sub.add_parser("local-reverse"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_local_reverse)
    p = sub.add_parser("local-cleanup"); p.add_argument("--apply", action="store_true"); p.set_defaults(fn=cmd_local_cleanup)
    p = sub.add_parser("model-upload"); p.add_argument("--model", default=states.DEFAULT_MODEL); p.set_defaults(fn=cmd_model_upload)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
