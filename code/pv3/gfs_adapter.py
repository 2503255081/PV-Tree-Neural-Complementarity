"""Read-only, interval-aware GFS adapter for the PVDAQ experiments.

Every call requires an explicit source root. Initialization plus delay is a
research availability assumption; actual historical receipt is unknown."""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file

DEFAULT_ROOT = None
VARIABLES = ("tmp2m", "rh2m", "u10m", "v10m", "surface_pressure", "tcdc_average", "dswrf_average", "precip_accumulation")
UNITS = ("K", "%", "m/s", "m/s", "Pa", "%", "W.m-2", "kg.m-2")
PREFIXES = ("Temperature_height_above_ground", "Relative_humidity_height_above_ground", "u-component_of_wind_height_above_ground", "v-component_of_wind_height_above_ground", "Pressure_surface", "Total_cloud_cover_entire_atmosphere_", "Downward_Short-Wave_Radiation_Flux_surface_", "Total_precipitation_surface_")
QUALITY = {"VALID": 0, "MISSING_RECEIPT": 1, "SOURCE_MISSING": 2, "INVALID_SOURCE": 3, "OUTSIDE_ACQUISITION": 4, "OUTSIDE_NATIVE_BOUNDS": 5, "DOCUMENTED_ENDPOINT_UNAVAILABLE": 6}
FEATURE_NAMES = tuple("gfs_" + v for v in VARIABLES) + tuple("gfs_" + v + "_native_span_hours" for v in VARIABLES) + ("gfs_lead_hours", "gfs_cycle_age_hours", "gfs_endpoint_minus_target_end_hours") + tuple("gfs_" + v + "_source_valid" for v in VARIABLES)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def utc_seconds(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timezone-naive source timestamp is forbidden")
    seconds = parsed.timestamp()
    if seconds != int(seconds):
        raise ValueError("Subsecond source time unexpected")
    return int(seconds)


def utc_string(value: int) -> str:
    return datetime.fromtimestamp(int(value), timezone.utc).isoformat().replace("+00:00", "Z")


def _inputs(issues_utc, horizons, delays_hours):
    issues = np.asarray(issues_utc)
    if issues.ndim != 1 or issues.dtype.kind not in "iu":
        raise TypeError("issues_utc must be a one-dimensional integer-seconds array")
    issues = issues.astype(np.int64, copy=False)
    if np.any(issues % 900):
        raise ValueError("Issues must lie on the 15-minute UTC lattice")
    hs = np.asarray(tuple(horizons))
    if hs.ndim != 1 or hs.dtype.kind not in "iu" or len(hs) == 0 or len(np.unique(hs)) != len(hs) or np.any((hs < 1) | (hs > 16)):
        raise ValueError("Horizons must be unique integers in 1..16")
    ds = tuple(delays_hours)
    if not ds or len(set(ds)) != len(ds) or any(isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d not in (4, 6, 8) for d in ds):
        raise ValueError("Only preregistered integer delays 6/4/8 hours are supported")
    return issues, hs.astype(np.int64), tuple(int(d) for d in ds)


def required_keys(issues_utc, horizons=range(1, 17), delays_hours=(6, 4, 8), config=None):
    """Deterministic required source keys, independent of files/labels/coverage.

    cycle=floor((issue-delay)/6h)*6h, endpoint=ceil(target_end/3h) in
    that cycle. No fallback to an older cycle or to another available lead.
    """
    issues, hs, ds = _inputs(issues_utc, horizons, delays_hours)
    pairs = []
    for delay in ds:
        cycles = ((issues - delay * 3600) // 21600) * 21600
        ends = issues[:, None] + hs[None, :] * 900
        leads = ((ends - cycles[:, None] + 10799) // 10800) * 3
        pairs.append(np.column_stack((np.repeat(cycles, len(hs)), leads.ravel())))
    unique = np.unique(np.concatenate(pairs, axis=0), axis=0) if len(issues) else np.empty((0, 2), np.int64)
    rows = []
    for cycle, lead in unique:
        stamp = datetime.fromtimestamp(int(cycle), timezone.utc).strftime("%Y%m%d%H")
        expected = True
        if config is not None:
            expected = utc_seconds(config["init_start_utc"]) <= cycle <= utc_seconds(config["init_end_utc"]) and int(lead) in config["lead_hours"]
        rows.append({"item_id": f"{stamp}_f{lead:03}", "initialization_utc": utc_string(cycle), "cycle_utc_seconds": int(cycle), "lead_hours": int(lead), "inside_acquisition_recipe": bool(expected)})
    return rows


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _nc_time(axis, value):
    scale, origin = _text(axis.units).split(" since ", 1)
    factor = {"hour": 3600, "hours": 3600, "second": 1, "seconds": 1, "minute": 60, "minutes": 60, "day": 86400, "days": 86400}.get(scale.lower())
    if factor is None:
        raise ValueError("Unrecognized CF time units")
    seconds = utc_seconds(origin) + float(value) * factor
    if not np.isfinite(seconds) or seconds != int(seconds):
        raise ValueError("Invalid CF clock")
    return int(seconds)


def _validate_record(record, key, j, config):
    cycle, lead = key["cycle_utc_seconds"], key["lead_hours"]
    variable = VARIABLES[j]
    if record["variable"] != variable or record["site_id"] != 2107 or utc_seconds(record["initialization_utc"]) != cycle or record["lead_hours"] != lead:
        raise ValueError("Receipt record identity mismatch")
    start, end = utc_seconds(record["interval_start_utc"]), utc_seconds(record["valid_end_utc"])
    if end != cycle + lead * 3600 or start > end or start < cycle:
        raise ValueError("Invalid forecast valid time or bounds")
    span = (end - start) / 3600
    if span != record["native_interval_hours"]:
        raise ValueError("Receipt span differs from native bounds")
    if j < 5 and (span != 0 or record["source_variable"] != PREFIXES[j]):
        raise ValueError("Instantaneous field has interval/incorrect semantics")
    if j >= 5:
        # Accept only the frozen 3/6-hour temporal resolution. A Mixed_intervals
        # source name is allowed, but *only* its decoded CF bounds are decisive.
        if span != (3 if lead % 6 else 6) or not record["source_variable"].startswith(PREFIXES[j]):
            raise ValueError("Unknown statistical source or unsupported native span")
        if record["statistics_type"] != ("Accumulation" if j == 7 else "Average"):
            raise ValueError("Incorrect native statistical process")
    if record["units"] != UNITS[j]:
        raise ValueError("Unexpected source units; no automatic unit guessing")
    height = 2.0 if j in (0, 1) else 10.0 if j in (2, 3) else None
    if record["height_m"] != height:
        raise ValueError("Height selection mismatch")
    site = config["site"]
    if abs(record["latitude"] - site["grid_latitude"]) > 1e-5 or abs((record["longitude"] % 360) - (site["grid_longitude"] % 360)) > 1e-5:
        raise ValueError("Grid identity mismatch")
    if record["actual_historical_available_at"] != "UNKNOWN":
        raise ValueError("Availability source changed; review a new protocol version")
    for delay in (4, 6, 8):
        if utc_seconds(record[f"assumed_available_at_{delay}h"]) != cycle + delay * 3600:
            raise ValueError("Simulated availability differs from cycle+delay")
    value = record["value"]
    if record["quality"] not in ("SOURCE_MISSING", "VALID_NUMERIC"):
        raise ValueError("Unknown native quality")
    if record["quality"] == "VALID_NUMERIC" and (value is None or not np.isfinite(value) or abs(value) > 1e19):
        raise ValueError("Invalid numeric value labelled valid")


def verify_native_netcdf(raw: bytes, records: list[dict], key: dict, config: dict):
    """Independent tiny NetCDF decoder; does not import mutable downloader code."""
    with netcdf_file(io.BytesIO(raw), mode="r", mmap=False) as nc:
        variables = nc.variables
        if _nc_time(variables["reftime"], np.asarray(variables["reftime"].data).item()) != key["cycle_utc_seconds"]:
            raise ValueError("Native initialization mismatch")
        lat = float(np.asarray(variables["latitude"].data).ravel()[0])
        lon = float(np.asarray(variables["longitude"].data).ravel()[0])
        if abs(lat - config["site"]["grid_latitude"]) > 1e-5 or abs(lon % 360 - config["site"]["grid_longitude"] % 360) > 1e-5:
            raise ValueError("Native grid mismatch")
        for j, record in enumerate(records):
            _validate_record(record, key, j, config)
            var = variables[record["source_variable"]]
            indices, bounds_seen = [], None
            for dim in var.dimensions:
                axis = variables[dim]
                vals = np.asarray(axis.data).ravel()
                if dim.startswith("height_above_ground"):
                    found = np.flatnonzero(np.isclose(vals, record["height_m"], atol=1e-6))
                    if len(found) != 1:
                        raise ValueError("Native requested sensor height absent")
                    indices.append(int(found[0]))
                elif dim.startswith("time") or _text(getattr(axis, "standard_name", "")) == "time":
                    name = _text(getattr(axis, "bounds", ""))
                    raw_bounds = np.asarray(variables[name].data).reshape(-1, 2) if name else np.column_stack((vals, vals))
                    if raw_bounds.shape != (len(vals), 2):
                        raise ValueError("Native CF bounds malformed")
                    wanted = (utc_seconds(record["interval_start_utc"]), utc_seconds(record["valid_end_utc"]))
                    candidate_bounds = [tuple(_nc_time(axis, b) for b in pair) for pair in raw_bounds]
                    candidates = [k for k, pair in enumerate(candidate_bounds) if pair == wanted]
                    if len(candidates) != 1:
                        raise ValueError("Native exact CF interval absent or ambiguous")
                    bounds_seen = candidate_bounds[candidates[0]]
                    indices.append(candidates[0])
                else:
                    if len(vals) != 1:
                        raise ValueError("Not a single spatial-point subset")
                    indices.append(0)
            if bounds_seen != (utc_seconds(record["interval_start_utc"]), utc_seconds(record["valid_end_utc"])):
                raise ValueError("Native CF interval differs from receipt")
            if _text(getattr(var, "units", "")) != record["units"]:
                raise ValueError("Native units differ from receipt")
            if j >= 5 and _text(getattr(var, "Grib2_Statistical_Process_Type", "")) != record["statistics_type"]:
                raise ValueError("Native statistical process differs from receipt")
            value = float(var.data[tuple(indices)])
            missing = not np.isfinite(value) or abs(value) > 1e19
            for name in ("_FillValue", "missing_value"):
                sentinel = getattr(var, name, None)
                if sentinel is not None and np.any(np.asarray(sentinel) == value):
                    missing = True
            if (record["quality"] == "SOURCE_MISSING") != missing or (not missing and value != record["value"]):
                raise ValueError("Native numeric value differs from receipt")


def snapshot_sources(issues_utc, horizons=range(1, 17), delays_hours=(6, 4, 8), root=DEFAULT_ROOT, verify_raw=True, verify_decode=True):
    """Return an immutable-by-convention snapshot; this function writes nothing.

    Caller saves this returned object and its hash in its own experiment folder.
    Missing keys remain in the snapshot; files arriving later cannot silently
    enter an already-built feature set. No SQLite/queue access is performed.
    """
    if verify_decode and not verify_raw:
        raise ValueError("Native decoding requires reading/verifying the raw source")
    if root is None:
        raise ValueError("An explicit GFS source root is required")
    root = Path(root).resolve()
    config_bytes = (root / "acquisition_config.json").read_bytes()
    config = json.loads(config_bytes)
    keys = required_keys(issues_utc, horizons, delays_hours, config)
    entries = []
    for key in keys:
        item = key["item_id"]
        receipt_path = root / "receipts" / item[:4] / item[4:6] / (item + ".json")
        raw_path = root / "raw" / item[:4] / item[4:6] / (item + ".nc")
        # Public overlay: each exact key is resolved in one immutable source root.
        if not receipt_path.is_file():
            legacy = root
            receipt_path = legacy / "receipts" / item[:4] / item[4:6] / (item + ".json")
            raw_path = legacy / "raw" / item[:4] / item[4:6] / (item + ".nc")
        entry = dict(key, receipt_path=str(receipt_path), raw_path=str(raw_path), status="MISSING_RECEIPT", records=[])
        if not key["inside_acquisition_recipe"]:
            entry["status"] = "OUTSIDE_ACQUISITION"
        elif receipt_path.is_file():
            try:
                receipt_bytes = receipt_path.read_bytes()
                receipt = json.loads(receipt_bytes)
                entry["receipt_sha256"] = sha256_bytes(receipt_bytes)
                if receipt["item_id"] != item or utc_seconds(receipt["initialization_utc"]) != key["cycle_utc_seconds"] or receipt["lead_hours"] != key["lead_hours"]:
                    raise ValueError("Receipt identity mismatch")
                if tuple(Path(receipt["local_path"].replace("\\", "/")).parts[-4:]) != tuple(raw_path.parts[-4:]):
                    raise ValueError("Receipt raw path is outside expected exact source key")
                if receipt.get("acquisition_config_sha256", sha256_bytes(config_bytes)) != sha256_bytes(config_bytes):
                    raise ValueError("Acquisition recipe hash mismatch")
                source_by_var = {r["variable"]: r for r in receipt["records"]}
                if len(source_by_var) != 8 or len(receipt["records"]) != 8 or set(source_by_var) != set(VARIABLES):
                    raise ValueError("Receipt variable set duplicate/missing/unknown")
                records = [source_by_var[v] for v in VARIABLES]
                for j, record in enumerate(records):
                    _validate_record(record, key, j, config)
                    if record["source_subset_sha256"] != receipt["sha256"]:
                        raise ValueError("Record and receipt raw hash mismatch")
                if verify_raw:
                    raw = raw_path.read_bytes()
                    if len(raw) != receipt["bytes"] or sha256_bytes(raw) != receipt["sha256"]:
                        raise ValueError("Raw source integrity failure")
                    if verify_decode:
                        verify_native_netcdf(raw, records, key, config)
                entry.update(status="VERIFIED" if verify_decode else "RECEIPT_ONLY", records=records, raw_sha256=receipt["sha256"], raw_bytes=receipt["bytes"], retrieved_at_utc=receipt.get("retrieved_at_utc"), source_grib_url=receipt.get("source_grib_url"), raw_sha_verified=bool(verify_raw), native_values_verified=bool(verify_decode))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                entry.update(status="INVALID_SOURCE", error=f"{type(exc).__name__}: {exc}", records=[])
        if entry['status']=='MISSING_RECEIPT':
            failure_path=root/'failures'/(item+'.json')
            if failure_path.is_file():
                failure=json.loads(failure_path.read_text(encoding='utf-8'))
                # Exact source error evidence only; no generic error-to-missing conversion.
                if failure.get('item_id')==item and failure.get('error')=='FileNotFoundError: Official run/lead endpoint returned404' and failure.get('attempts') and all(a.get('http_status')==404 and ('gfs.0p25.'+item[:10]+'.'+item[11:]+'.grib2?') in a.get('url','') for a in failure['attempts']):
                    entry.update(status='DOCUMENTED_ENDPOINT_UNAVAILABLE',failure_path=str(failure_path),failure_sha256=sha256_bytes(failure_path.read_bytes()),records=[])
        entries.append(entry)
    required_hash = sha256_bytes(json.dumps(keys, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {"adapter_schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "config": config, "config_sha256": sha256_bytes(config_bytes), "required_keys_sha256": required_hash, "actual_historical_available_at": "UNKNOWN", "simulated_delay_hours": list(delays_hours), "raw_sha_verification_requested": bool(verify_raw), "native_verification_requested": bool(verify_decode), "entries": entries}


def source_gate(snapshot):
    """Strict gate for the enumerated snapshot, never a full-archive assertion."""
    counts = {}
    for entry in snapshot["entries"]:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    verified = bool(snapshot["entries"]) and all(e["status"] == "VERIFIED" and e.get("raw_sha_verified") and e.get("native_values_verified") for e in snapshot["entries"])
    numeric = verified and all(r["quality"] == "VALID_NUMERIC" for e in snapshot["entries"] for r in e["records"])
    return {"required_key_count": len(snapshot["entries"]), "counts": counts, "all_required_source_integrity_verified": bool(verified), "all_required_native_values_numeric": bool(numeric), "ready_for_complete_queue_feature_check": bool(numeric), "actual_historical_available_at": "UNKNOWN", "production_availability_verified": False, "scope": "Only the exact supplied measurement-derived required keys; no claim of whole-archive completion"}


def build_features(issues_utc, horizons=range(1, 17), delays_hours=(6, 4, 8), snapshot=None, root=DEFAULT_ROOT):
    """Return {scenarios[delay], feature_names, key_table, source_snapshot}.

    Each scenario: features[N,H,27], values[N,H,8], quality[N,H,8],
    key_index[N,H], source_cycle_utc[N], all16_complete[N]. Bounds/valid time
    per variable are losslessly retained in key_table[key_index].records.
    all16_complete is always false for a horizon subset: no subset masquerades
    as the required complete 16-horizon evaluation queue.
    """
    issues, hs, ds = _inputs(issues_utc, horizons, delays_hours)
    if snapshot is None:
        snapshot = snapshot_sources(issues, hs, ds, root=root)
    elif isinstance(snapshot, (str, Path)):
        snapshot = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    if snapshot.get("adapter_schema_version") != 1 or snapshot.get("actual_historical_available_at") != "UNKNOWN":
        raise ValueError("Unrecognized snapshot protocol")
    entries = snapshot["entries"]
    ids = [e["item_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate keys in frozen snapshot")
    id_set = set(ids)
    lookup = {(e["cycle_utc_seconds"], e["lead_hours"]): i for i, e in enumerate(entries)}
    missing = [k["item_id"] for k in required_keys(issues, hs, ds, snapshot["config"]) if k["item_id"] not in id_set]
    if missing:
        raise ValueError(f"Snapshot does not enumerate requested keys: {missing[:5]}")
    # Key-level arrays avoid replicating long provenance strings for all points.
    native_values = np.full((len(entries), 8), np.nan, np.float32)
    native_span = np.full_like(native_values, np.nan)
    native_start = np.zeros((len(entries), 8), np.int64)
    native_quality = np.full((len(entries), 8), QUALITY["MISSING_RECEIPT"], np.uint8)
    for i, entry in enumerate(entries):
        if entry["status"] in ("VERIFIED", "RECEIPT_ONLY"):
            for j, record in enumerate(entry["records"]):
                _validate_record(record, entry, j, snapshot["config"])
                native_span[i, j] = record["native_interval_hours"]
                native_start[i, j] = utc_seconds(record["interval_start_utc"])
                if record["quality"] == "VALID_NUMERIC":
                    native_values[i, j] = record["value"]
                    native_quality[i, j] = QUALITY["VALID"]
                else:
                    native_quality[i, j] = QUALITY["SOURCE_MISSING"]
        else:
            native_quality[i, :] = QUALITY[entry["status"]]
    result: dict[str, Any] = {"issues_utc": issues, "horizons": hs, "feature_names": FEATURE_NAMES, "variables": VARIABLES, "quality_codes": QUALITY, "key_table": entries, "source_snapshot": snapshot, "source_gate": source_gate(snapshot), "scenarios": {}}
    target_start = issues[:, None] + (hs[None, :] - 1) * 900
    target_end = target_start + 900
    for delay in ds:
        cycles = ((issues - delay * 3600) // 21600) * 21600
        leads = ((target_end - cycles[:, None] + 10799) // 10800) * 3
        key_index = np.fromiter((lookup[(int(c), int(l))] for c, row in zip(cycles, leads) for l in row), np.int32, count=len(issues) * len(hs)).reshape(len(issues), len(hs))
        values = native_values[key_index].copy()
        quality = native_quality[key_index].copy()
        span = native_span[key_index]
        starts = native_start[key_index]
        # The selected broad interval must contain the entire 15min target.
        bad_bounds = (quality[:, :, 5:] == 0) & (starts[:, :, 5:] > target_start[:, :, None])
        quality[:, :, 5:][bad_bounds] = QUALITY["OUTSIDE_NATIVE_BOUNDS"]
        values[quality != 0] = np.nan
        feat = np.empty((len(issues), len(hs), len(FEATURE_NAMES)), np.float32)
        feat[:, :, :8] = values
        feat[:, :, 8:16] = span
        feat[:, :, 16] = leads
        feat[:, :, 17] = (issues - cycles)[:, None] / 3600
        feat[:, :, 18] = (cycles[:, None] + leads * 3600 - target_end) / 3600
        feat[:, :, 19:] = (quality == 0).astype(np.float32)
        complete_points = (quality == 0).all(axis=2)
        has_all = set(hs.tolist()) == set(range(1, 17))
        result["scenarios"][delay] = {"features": feat, "values": values, "quality": quality, "key_index": key_index, "source_cycle_utc": cycles, "assumed_available_at_utc": cycles + delay * 3600, "target_start_utc": target_start, "target_end_utc": target_end, "complete_points": complete_points, "all16_complete": complete_points.all(axis=1) & has_all, "actual_historical_available_at": "UNKNOWN", "complete_point_fraction": float(complete_points.mean()) if complete_points.size else None, "all16_complete_fraction": float((complete_points.all(axis=1) & has_all).mean()) if len(issues) else None}
    return result


def sample_provenance(result, issue_indices=(0,), delays_hours=None):
    """JSON-ready long sample table, one row per requested issue/h/variable."""
    rows = []
    for delay in result["scenarios"] if delays_hours is None else delays_hours:
        sc = result["scenarios"][delay]
        for i in issue_indices:
            for col, h in enumerate(result["horizons"]):
                entry = result["key_table"][int(sc["key_index"][i, col])]
                for j, name in enumerate(VARIABLES):
                    native = entry["records"][j] if entry["records"] else {}
                    rows.append({"issue_utc": utc_string(result["issues_utc"][i]), "horizon_minutes": int(h) * 15, "delay_hours": int(delay), "target_start_utc": utc_string(sc["target_start_utc"][i, col]), "target_end_utc": utc_string(sc["target_end_utc"][i, col]), "required_item_id": entry["item_id"], "source_cycle_utc": entry["initialization_utc"], "source_variable": native.get("source_variable"), "variable": name, "source_interval_start_utc": native.get("interval_start_utc"), "source_valid_end_utc": native.get("valid_end_utc"), "native_span_hours": native.get("native_interval_hours"), "statistics_type": native.get("statistics_type"), "units": native.get("units"), "value": float(sc["values"][i, col, j]) if np.isfinite(sc["values"][i, col, j]) else None, "quality_code": int(sc["quality"][i, col, j]), "source_status": entry["status"], "receipt_sha256": entry.get("receipt_sha256"), "raw_sha256": entry.get("raw_sha256"), "simulated_available_at_utc": utc_string(sc["assumed_available_at_utc"][i]), "actual_historical_available_at": "UNKNOWN"})
    return rows
