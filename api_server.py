"""
FastAPI server that runs the full AI pipeline (GAN, DrugEncoder, PharmacodynamicPredictor, TimeSeriesPredictor)
and exposes POST /api/simulate for the web dashboard.
Run: uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""
from pathlib import Path
from typing import Optional
import sys
import math

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Lazy imports after path setup
def _import_models():
    from models.patient_generator_gan import PatientGenerator
    from encoders.drugEncoder import DrugEncoder
    from models.pharmacodynamicPredictor import PharmacodynamicPredictor, LAB_BIOMARKER_FEATURES
    from models.time_series_predictor import TimeSeriesPredictor, TimeSeriesConfig
    return PatientGenerator, DrugEncoder, PharmacodynamicPredictor, LAB_BIOMARKER_FEATURES, TimeSeriesPredictor, TimeSeriesConfig

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
SECONDS_PER_HOUR = 3600
MODEL_TIMEPOINTS_DAYS = [10, 20, 30, 60, 90, 180]

# Normal ranges matching frontend LAB_DEFINITIONS (used for pathological classification)
_LAB_NORMAL = {
    "LBDSCASI": (0, 3), "LBDSCH": (3.6, 5.2), "LBDSCR": (53, 115), "LBDTC": (3.6, 5.2),
    "LBXBCD": (0.1, 1.0), "LBXBPB": (0, 5), "LBXCRP": (0, 3), "LBXSAL": (3.5, 5.0),
    "LBXSAS": (10, 40), "LBXSBU": (7, 20), "LBXSCA": (8.5, 10.5), "LBXSCH": (125, 200),
    "LBXSCL": (98, 106), "LBXSGB": (2.0, 3.5), "LBXSGL": (70, 100), "LBXSGT": (8, 61),
    "LBXSK": (3.5, 5.0), "LBXSNA": (136, 145), "LBXSOS": (275, 295), "LBXSTP": (6.0, 8.3),
    "LBXSUA": (3.5, 7.2), "LBXTC": (125, 200),
}


def _classify_metric(lab_code, final_val):
    """Classify a lab result as improved (in range) or worsened (out of range)."""
    rng = _LAB_NORMAL.get(lab_code)
    if rng and (final_val < rng[0] or final_val > rng[1]):
        return "worsened"
    return "improved"

# Globals: loaded once at startup
_gan = None
_encoder = None
_pd_predictor = None
_ts_predictor = None
_nhanes_path = None


def _generate_timepoints_seconds(total_seconds: int, interval_seconds: int):
    timepoints = [0]
    t = interval_seconds
    while t <= total_seconds:
        timepoints.append(t)
        t += interval_seconds
    if timepoints[-1] != total_seconds:
        timepoints.append(total_seconds)
    return sorted(set(timepoints))


def load_models(
    nhanes_path: str = "preprocessed_nhanes/nhanes_final_complete.csv",
    skip_gan_if_missing: bool = True,
):
    """Load all AI models once. Call at app startup."""
    global _gan, _encoder, _pd_predictor, _ts_predictor, _nhanes_path
    _nhanes_path = Path(nhanes_path)
    if not _nhanes_path.is_absolute():
        _nhanes_path = ROOT / _nhanes_path

    (
        PatientGenerator,
        DrugEncoder,
        PharmacodynamicPredictor,
        LAB_BIOMARKER_FEATURES,
        TimeSeriesPredictor,
        TimeSeriesConfig,
    ) = _import_models()

    # 1) Patient GAN
    if _nhanes_path.exists():
        _gan = PatientGenerator(data_path=str(_nhanes_path))
        best = ROOT / "models/generator/patient_generator_gan_best.pt"
        if not best.exists():
            best = ROOT / "models/patient_generator_gan_best.pt"
        if best.exists():
            import torch
            ckpt = torch.load(best, map_location=DEVICE, weights_only=False)
            _gan.generator.load_state_dict(ckpt["generator"])
    elif not skip_gan_if_missing:
        raise FileNotFoundError(f"NHANES data required for GAN: {_nhanes_path}")

    # 2) Drug encoder
    _encoder = DrugEncoder(encoder_type="hybrid", device=DEVICE)

    # 3) Pharmacodynamic predictor
    _pd_predictor = PharmacodynamicPredictor(
        predictor_type="cross_attention",
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        use_constraints=True,
        device=DEVICE,
    )
    pd_path = ROOT / "models/pharmacodynamic_predictor/predictor_novel_drug_best.pt"
    if pd_path.exists():
        _pd_predictor.load(str(pd_path))

    # 4) Time series predictor
    ts_path = ROOT / "models/time_series_predictor.pt"
    if not ts_path.exists():
        raise FileNotFoundError(f"Time series model required: {ts_path}")
    _ts_predictor = TimeSeriesPredictor(
        drug_vocab={"Unknown": 0},
        metric_names=["LBXTC"],
        patient_feature_names=["age", "sex", "bmi", "adherence", "days_on_drug", "dosage"],
        config=TimeSeriesConfig(timepoints=MODEL_TIMEPOINTS_DAYS),
        device=DEVICE,
    )
    _ts_predictor.load(str(ts_path))

    return True


def _drug_concentration(time_sec: float, dose_mg: float, volume_l: float, half_life_min: float):
    """One-compartment IV bolus."""
    if time_sec < 0:
        return 0.0
    t_min = time_sec / 60.0
    k = 0.693 / half_life_min
    c0 = dose_mg / volume_l
    return max(0.0, c0 * math.exp(-k * t_min))


def run_single_simulation(
    age: int,
    sex: str,
    height: float,
    weight: float,
    smiles: str,
    drug_name: str = "Unknown",
    dosage: float = 10.0,
    total_hours: float = 3.0,
    interval_seconds: int = 30,
    volume_l: float = 50.0,
    half_life_min: float = 60.0,
    seed: int = None,
):
    """
    Run the full AI pipeline for one patient and one drug.
    Returns dict: times, series, baselines, drugConcs, patient, drug, dosage, durationSec.
    """
    if _gan is None and _nhanes_path and _nhanes_path.exists():
        load_models()
    if _gan is None:
        raise RuntimeError("Patient generator (GAN) not available: NHANES data missing. Run with preprocessed data or use CLI.")

    total_seconds = int(total_hours * SECONDS_PER_HOUR)
    timepoints_seconds = _generate_timepoints_seconds(total_seconds, interval_seconds)
    timepoints_seconds_no_zero = [t for t in timepoints_seconds if t > 0]

    # 1) Generate patient (unique seed per patient so cohort members differ)
    S0 = _gan.generate(age=age, sex=sex, height=height, weight=weight, n=1, seed=seed)
    bmi = float(S0["BMXBMI"].values[0])

    # 2) Encode drug
    emb = _encoder.encode(smiles)

    # 3) PD prediction
    delta_df = _pd_predictor.predict_delta(S0, emb)

    predictor_data = {
        "patient_id": [1],
        "drug_name": [drug_name],
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "days_on_drug": [1],
        "adherence": [1.0],
        "dosage": [dosage],
    }
    for metric in _pd_predictor.lab_features:
        if metric in S0.columns:
            baseline = S0[metric].values[0]
            delta = delta_df[metric].values[0] if metric in delta_df.columns else 0.0
        else:
            baseline = 0.0
            delta = 0.0
        predictor_data[f"{metric}_baseline"] = [baseline]
        predictor_data[f"{metric}_delta"] = [delta]

    predictor_df = pd.DataFrame(predictor_data)
    model_metrics = set(_ts_predictor.metric_names)
    required = ["patient_id", "drug_name", "age", "sex", "bmi", "adherence", "days_on_drug", "dosage"]
    filtered = predictor_df[required].copy()
    for metric in model_metrics:
        base = metric.replace("_delta", "")
        filtered[f"{base}_baseline"] = predictor_df.get(f"{base}_baseline", 0.0)
        filtered[f"{base}_delta"] = predictor_df.get(f"{base}_delta", 0.0)
    predictor_df = filtered

    # 4) Time series prediction (day-based) then interpolate to seconds
    ts_df = _ts_predictor.predict(predictor_df, return_uncertainty=False, drug_embedding=emb)

    new_columns_data = {}
    for s in timepoints_seconds_no_zero:
        new_columns_data[f"{s}sec"] = [np.nan] * len(ts_df)

    for metric_name in ts_df["metric_name"].unique():
        mask = ts_df["metric_name"] == metric_name
        rows = ts_df[mask]
        if len(rows) == 0:
            continue
        baseline = rows["baseline"].iloc[0]
        final_delta = rows["final_delta"].iloc[0]
        model_sec_vals = {0: 0.0, total_seconds: final_delta}
        for day in MODEL_TIMEPOINTS_DAYS:
            col = f"day_{day}"
            if col in rows.columns:
                s_at_day = int((day / 180.0) * total_seconds)
                model_sec_vals[s_at_day] = rows[col].iloc[0]
        sorted_sec = sorted(model_sec_vals.keys())
        sorted_val = [model_sec_vals[s] for s in sorted_sec]
        interp = interp1d(sorted_sec, sorted_val, kind="linear", fill_value="extrapolate", bounds_error=False)
        for s in timepoints_seconds_no_zero:
            col = f"{s}sec"
            delta_t = float(interp(s))
            val = baseline + delta_t
            for idx in ts_df[mask].index:
                new_columns_data[col][idx] = val

    if new_columns_data:
        extra = pd.DataFrame(new_columns_data, index=ts_df.index)
        ts_df = pd.concat([ts_df, extra], axis=1)

    # Build response in web format
    times = [0] + timepoints_seconds_no_zero
    series = {}
    baselines = {}
    for metric_name in ts_df["metric_name"].unique():
        rows = ts_df[ts_df["metric_name"] == metric_name]
        if len(rows) == 0:
            continue
        baseline = float(rows["baseline"].iloc[0])
        baselines[metric_name] = baseline
        vals = [baseline]
        for s in timepoints_seconds_no_zero:
            col = f"{s}sec"
            if col in rows.columns:
                vals.append(float(rows[col].iloc[0]))
            else:
                vals.append(baseline)
        series[metric_name] = vals

    drugConcs = [_drug_concentration(t, dosage, volume_l, half_life_min) for t in times]

    patient = {"age": int(age), "sex": sex, "height": float(height), "weight": float(weight), "bmi": bmi}
    drug = {"name": drug_name, "smiles": smiles}

    return _sanitize({
        "times": times,
        "series": series,
        "baselines": baselines,
        "drugConcs": drugConcs,
        "patient": patient,
        "drug": drug,
        "dosage": float(dosage),
        "durationSec": int(total_seconds),
    })


def _sanitize(obj):
    """Convert numpy types to native Python so JSON serialization never fails."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# --- FastAPI app ---
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Install FastAPI and uvicorn: pip install fastapi uvicorn")
import json as _json
import asyncio

app = FastAPI(title="BloodTwin / DAIN AI Simulation API")


class SimulateRequest(BaseModel):
    age: int = 45
    sex: str = "M"
    height: float = 175.0
    weight: float = 80.0
    smiles: str = "CC(=O)OC1=CC=CC=C1C(=O)O"
    drug_name: str = "Aspirin"
    dosage: float = 10.0
    total_hours: float = 3.0
    interval_seconds: int = 30
    patient_count: int = 1
    volume_l: Optional[float] = None
    half_life_min: Optional[float] = None


# Models load on first /api/simulate (lazy) so the server starts in seconds


def _ensure_models_loaded():
    """Load models on first request so startup is fast."""
    global _gan, _encoder, _pd_predictor, _ts_predictor
    if _ts_predictor is not None and _encoder is not None and _pd_predictor is not None:
        return
    try:
        load_models(skip_gan_if_missing=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Models or data missing: {e}")


@app.post("/api/simulate")
def api_simulate(body: SimulateRequest):
    """Run AI pipeline (GAN patient + DrugEncoder + PD predictor + Time series)."""
    _ensure_models_loaded()
    if body.sex not in ("M", "F"):
        raise HTTPException(status_code=400, detail="sex must be M or F")
    if body.patient_count < 1 or body.patient_count > 10000:
        raise HTTPException(status_code=400, detail="patient_count must be 1–10000")

    vol = body.volume_l if body.volume_l is not None else 50.0
    half = body.half_life_min if body.half_life_min is not None else 60.0

    if body.patient_count == 1:
        try:
            out = run_single_simulation(
                age=body.age,
                sex=body.sex,
                height=body.height,
                weight=body.weight,
                smiles=body.smiles.strip(),
                drug_name=body.drug_name or "Unknown",
                dosage=body.dosage,
                total_hours=body.total_hours,
                interval_seconds=body.interval_seconds,
                volume_l=vol,
                half_life_min=half,
                seed=42,
            )
            return out
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Cohort: batched pipeline – encode drug ONCE, generate patients in bulk,
        # batch PD + TS predictions, then build per-patient results.
        import random
        from concurrent.futures import ThreadPoolExecutor

        AGE_GROUPS = [
            {"id": "18-30", "min": 18, "max": 30},
            {"id": "31-45", "min": 31, "max": 45},
            {"id": "46-60", "min": 46, "max": 60},
            {"id": "61-85", "min": 61, "max": 85},
        ]
        N = body.patient_count
        smiles = body.smiles.strip()
        drug_name = body.drug_name or "Unknown"
        total_seconds = int(body.total_hours * SECONDS_PER_HOUR)
        # Fewer timepoints for cohort to cut interpolation work
        cohort_interval = max(body.interval_seconds, 60)
        timepoints_seconds = _generate_timepoints_seconds(total_seconds, cohort_interval)
        timepoints_seconds_no_zero = [t for t in timepoints_seconds if t > 0]

        # -- Demographics --
        patients = []
        half_n = N // 2
        sexes = ["M"] * half_n + ["F"] * (N - half_n)
        random.shuffle(sexes)
        for i in range(N):
            g = AGE_GROUPS[i % len(AGE_GROUPS)]
            age = random.randint(g["min"], g["max"])
            sex = sexes[i]
            is_m = sex == "M"
            height = random.uniform(165, 195) if is_m else random.uniform(155, 180)
            weight = random.uniform(60, 110) if is_m else random.uniform(50, 95)
            bmi = weight / (height / 100) ** 2
            patients.append({"age": age, "sex": sex, "height": height, "weight": weight, "bmi": bmi, "ageGroup": g})

        # -- Encode drug ONCE (biggest single-patient bottleneck) --
        emb = _encoder.encode(smiles)

        # -- Batch GAN generation (process in chunks to limit memory) --
        BATCH = min(N, 128)
        all_S0 = []
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            for idx in range(start, end):
                p = patients[idx]
                s0 = _gan.generate(age=p["age"], sex=p["sex"], height=p["height"],
                                   weight=p["weight"], n=1, seed=1000 + idx)
                all_S0.append(s0)
        S0_all = pd.concat(all_S0, ignore_index=True)

        # -- Batch PD prediction (model supports multi-row DataFrame) --
        delta_all = _pd_predictor.predict_delta(S0_all, emb)

        # -- Batch TS predictor dataframe --
        ts_rows = []
        for idx in range(N):
            p = patients[idx]
            row = {
                "patient_id": idx,
                "drug_name": drug_name,
                "age": p["age"],
                "sex": p["sex"],
                "bmi": p["bmi"],
                "days_on_drug": 1,
                "adherence": 1.0,
                "dosage": body.dosage,
            }
            for metric in _pd_predictor.lab_features:
                baseline = float(S0_all.at[idx, metric]) if metric in S0_all.columns else 0.0
                delta = float(delta_all.at[idx, metric]) if metric in delta_all.columns else 0.0
                row[f"{metric}_baseline"] = baseline
                row[f"{metric}_delta"] = delta
            ts_rows.append(row)

        predictor_df = pd.DataFrame(ts_rows)
        model_metrics = set(_ts_predictor.metric_names)
        required = ["patient_id", "drug_name", "age", "sex", "bmi", "adherence", "days_on_drug", "dosage"]
        filtered = predictor_df[required].copy()
        for metric in model_metrics:
            base = metric.replace("_delta", "")
            filtered[f"{base}_baseline"] = predictor_df.get(f"{base}_baseline", 0.0)
            filtered[f"{base}_delta"] = predictor_df.get(f"{base}_delta", 0.0)
        predictor_df = filtered

        # -- Batch TS prediction (single call for all patients) --
        ts_df = _ts_predictor.predict(predictor_df, return_uncertainty=False, drug_embedding=emb)

        # -- Interpolate per-patient time series in parallel --
        def _build_patient_result(idx):
            p = patients[idx]
            pid_mask = ts_df["patient_id"] == idx
            ts_patient = ts_df[pid_mask]

            times = [0] + timepoints_seconds_no_zero
            series = {}
            baselines_out = {}
            for metric_name in ts_patient["metric_name"].unique():
                rows = ts_patient[ts_patient["metric_name"] == metric_name]
                if len(rows) == 0:
                    continue
                baseline = float(rows["baseline"].iloc[0])
                final_delta = float(rows["final_delta"].iloc[0])
                baselines_out[metric_name] = baseline
                model_sec_vals = {0: 0.0, total_seconds: final_delta}
                for day in MODEL_TIMEPOINTS_DAYS:
                    col = f"day_{day}"
                    if col in rows.columns:
                        s_at_day = int((day / 180.0) * total_seconds)
                        model_sec_vals[s_at_day] = float(rows[col].iloc[0])
                sorted_sec = sorted(model_sec_vals.keys())
                sorted_val = [model_sec_vals[s] for s in sorted_sec]
                interp = interp1d(sorted_sec, sorted_val, kind="linear",
                                  fill_value="extrapolate", bounds_error=False)
                vals = [baseline]
                for s in timepoints_seconds_no_zero:
                    vals.append(float(baseline + interp(s)))
                series[metric_name] = vals

            drugConcs = [_drug_concentration(t, body.dosage, vol, half) for t in times]

            sim = {
                "times": times,
                "series": series,
                "baselines": baselines_out,
                "drugConcs": drugConcs,
                "patient": p,
                "drug": {"name": drug_name, "smiles": smiles},
                "dosage": float(body.dosage),
                "durationSec": total_seconds,
            }

            improved = 0
            worsened = 0
            metrics_list = []
            for lab_code, vals in series.items():
                if len(vals) < 2:
                    continue
                base = vals[0]
                final = vals[-1]
                delta = final - base
                status = _classify_metric(lab_code, final)
                if status == "improved":
                    improved += 1
                elif status == "worsened":
                    worsened += 1
                metrics_list.append({"labCode": lab_code, "status": status,
                                     "baseline": base, "finalVal": final, "delta": delta})
            return {
                "patient": p,
                "sim": _sanitize(sim),
                "classification": {"metrics": metrics_list, "improvedCount": improved, "worsenedCount": worsened},
            }

        # Use threads for the interpolation / numpy work (releases GIL)
        max_workers = min(8, max(1, N // 4))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            per_patient = list(pool.map(_build_patient_result, range(N)))

        return _build_cohort_response(patients, per_patient, AGE_GROUPS, drug_name, smiles, body.dosage, total_seconds)


def _build_cohort_response(patients, per_patient, AGE_GROUPS, drug_name, smiles, dosage, total_seconds):
    """Build the final (or partial) cohort response from whatever patients are done."""
    lab_codes = list(next((pp["sim"]["series"].keys() for pp in per_patient if pp.get("sim")), []))
    lab_stats_map = {}
    for lab_code in lab_codes:
        lab_stats_map[lab_code] = {
            "labCode": lab_code, "name": lab_code, "unit": "", "normal": None,
            "physiological": 0, "pathological": 0, "total": 0,
            "bySex": {"M": {"pathological": 0, "total": 0}, "F": {"pathological": 0, "total": 0}},
            "byAgeGroup": {g["id"]: {"pathological": 0, "total": 0} for g in AGE_GROUPS},
        }
    for pp in per_patient:
        if not pp.get("sim"):
            continue
        p = pp["patient"]
        for m in pp["classification"]["metrics"]:
            ls = lab_stats_map.get(m["labCode"])
            if ls is None:
                continue
            ls["total"] += 1
            if m["status"] == "worsened":
                ls["pathological"] += 1
            else:
                ls["physiological"] += 1
            sx = ls["bySex"].get(p["sex"])
            if sx:
                sx["total"] += 1
                if m["status"] == "worsened":
                    sx["pathological"] += 1
            ag_id = p.get("ageGroup", {}).get("id")
            if ag_id and ag_id in ls["byAgeGroup"]:
                bg = ls["byAgeGroup"][ag_id]
                bg["total"] += 1
                if m["status"] == "worsened":
                    bg["pathological"] += 1
    lab_stats = list(lab_stats_map.values())
    total_pathological = sum(ls["pathological"] for ls in lab_stats)
    total_observations = sum(ls["total"] for ls in lab_stats)
    # Compute age insights: for each age group, find the lab with highest pathological rate
    age_insights = []
    for g in AGE_GROUPS:
        per_lab = []
        for ls in lab_stats:
            bucket = ls["byAgeGroup"].get(g["id"], {"pathological": 0, "total": 0})
            rate = bucket["pathological"] / bucket["total"] if bucket["total"] else 0
            per_lab.append({"labCode": ls["labCode"], "name": ls["labCode"], "rate": rate})
        per_lab.sort(key=lambda x: x["rate"], reverse=True)
        top = per_lab[0] if per_lab and per_lab[0]["rate"] > 0 else None
        g_with_label = {**g, "label": g["id"]}
        age_insights.append({"group": g_with_label, "top": top})

    # Compute sex insights: for each sex, find the lab with highest pathological rate
    sex_insights = []
    for s in ("M", "F"):
        per_lab = []
        for ls in lab_stats:
            bucket = ls["bySex"].get(s, {"pathological": 0, "total": 0})
            rate = bucket["pathological"] / bucket["total"] if bucket["total"] else 0
            per_lab.append({"labCode": ls["labCode"], "name": ls["labCode"], "rate": rate})
        per_lab.sort(key=lambda x: x["rate"], reverse=True)
        top = per_lab[0] if per_lab and per_lab[0]["rate"] > 0 else None
        sex_insights.append({"sex": s, "top": top})

    done_patients = [pp["patient"] for pp in per_patient]
    return _sanitize({
        "patients": done_patients,
        "perPatient": per_patient,
        "labStats": lab_stats,
        "summary": {
            "patientCount": len(per_patient),
            "totalObservations": total_observations,
            "totalPathological": total_pathological,
            "totalPhysiological": total_observations - total_pathological,
        },
        "ageInsights": age_insights,
        "sexInsights": sex_insights,
        "drugInstability": [],
        "baseDrug": {"name": drug_name, "smiles": smiles},
        "dosage": dosage,
        "durationSec": total_seconds,
    })


@app.post("/api/simulate/stream")
async def api_simulate_stream(body: SimulateRequest, request: Request):
    """Streaming cohort simulation with progress events (SSE).
    Yields JSON lines: {"type":"progress","done":n,"total":N}
    Final line:         {"type":"result","data":{...}}
    Client can disconnect at any time to cancel; partial results are sent.
    """
    _ensure_models_loaded()
    if body.sex not in ("M", "F"):
        raise HTTPException(status_code=400, detail="sex must be M or F")
    if body.patient_count < 2 or body.patient_count > 10000:
        raise HTTPException(status_code=400, detail="patient_count must be 2–10000 for streaming")

    import random

    AGE_GROUPS = [
        {"id": "18-30", "min": 18, "max": 30},
        {"id": "31-45", "min": 31, "max": 45},
        {"id": "46-60", "min": 46, "max": 60},
        {"id": "61-85", "min": 61, "max": 85},
    ]
    N = body.patient_count
    smiles = body.smiles.strip()
    drug_name = body.drug_name or "Unknown"
    vol = body.volume_l if body.volume_l is not None else 50.0
    half = body.half_life_min if body.half_life_min is not None else 60.0
    total_seconds = int(body.total_hours * SECONDS_PER_HOUR)
    cohort_interval = max(body.interval_seconds, 60)
    timepoints_seconds = _generate_timepoints_seconds(total_seconds, cohort_interval)
    timepoints_seconds_no_zero = [t for t in timepoints_seconds if t > 0]

    patients = []
    half_n = N // 2
    sexes = ["M"] * half_n + ["F"] * (N - half_n)
    random.shuffle(sexes)
    for i in range(N):
        g = AGE_GROUPS[i % len(AGE_GROUPS)]
        age = random.randint(g["min"], g["max"])
        sex = sexes[i]
        is_m = sex == "M"
        height = random.uniform(165, 195) if is_m else random.uniform(155, 180)
        weight = random.uniform(60, 110) if is_m else random.uniform(50, 95)
        bmi = weight / (height / 100) ** 2
        patients.append({"age": age, "sex": sex, "height": height, "weight": weight, "bmi": bmi, "ageGroup": g})

    emb = _encoder.encode(smiles)

    BATCH = min(N, 64)

    async def event_generator():
        per_patient = []
        for batch_start in range(0, N, BATCH):
            if await request.is_disconnected():
                break
            batch_end = min(batch_start + BATCH, N)
            batch_patients = patients[batch_start:batch_end]
            batch_size = batch_end - batch_start

            S0_list = []
            for idx in range(batch_start, batch_end):
                p = patients[idx]
                s0 = _gan.generate(age=p["age"], sex=p["sex"], height=p["height"],
                                   weight=p["weight"], n=1, seed=1000 + idx)
                S0_list.append(s0)
            S0_batch = pd.concat(S0_list, ignore_index=True)

            delta_batch = _pd_predictor.predict_delta(S0_batch, emb)

            ts_rows = []
            for local_i, idx in enumerate(range(batch_start, batch_end)):
                p = patients[idx]
                row = {
                    "patient_id": idx,
                    "drug_name": drug_name,
                    "age": p["age"], "sex": p["sex"], "bmi": p["bmi"],
                    "days_on_drug": 1, "adherence": 1.0, "dosage": body.dosage,
                }
                for metric in _pd_predictor.lab_features:
                    baseline = float(S0_batch.at[local_i, metric]) if metric in S0_batch.columns else 0.0
                    delta = float(delta_batch.at[local_i, metric]) if metric in delta_batch.columns else 0.0
                    row[f"{metric}_baseline"] = baseline
                    row[f"{metric}_delta"] = delta
                ts_rows.append(row)

            predictor_df = pd.DataFrame(ts_rows)
            model_metrics = set(_ts_predictor.metric_names)
            required_cols = ["patient_id", "drug_name", "age", "sex", "bmi", "adherence", "days_on_drug", "dosage"]
            filtered = predictor_df[required_cols].copy()
            for metric in model_metrics:
                base = metric.replace("_delta", "")
                filtered[f"{base}_baseline"] = predictor_df.get(f"{base}_baseline", 0.0)
                filtered[f"{base}_delta"] = predictor_df.get(f"{base}_delta", 0.0)

            ts_df = _ts_predictor.predict(filtered, return_uncertainty=False, drug_embedding=emb)

            for local_i, idx in enumerate(range(batch_start, batch_end)):
                p = patients[idx]
                pid_mask = ts_df["patient_id"] == idx
                ts_patient = ts_df[pid_mask]
                times = [0] + timepoints_seconds_no_zero
                series = {}
                baselines_out = {}
                for metric_name in ts_patient["metric_name"].unique():
                    rows = ts_patient[ts_patient["metric_name"] == metric_name]
                    if len(rows) == 0:
                        continue
                    baseline = float(rows["baseline"].iloc[0])
                    final_delta = float(rows["final_delta"].iloc[0])
                    baselines_out[metric_name] = baseline
                    model_sec_vals = {0: 0.0, total_seconds: final_delta}
                    for day in MODEL_TIMEPOINTS_DAYS:
                        col = f"day_{day}"
                        if col in rows.columns:
                            s_at_day = int((day / 180.0) * total_seconds)
                            model_sec_vals[s_at_day] = float(rows[col].iloc[0])
                    sorted_sec = sorted(model_sec_vals.keys())
                    sorted_val = [model_sec_vals[s] for s in sorted_sec]
                    interp = interp1d(sorted_sec, sorted_val, kind="linear",
                                      fill_value="extrapolate", bounds_error=False)
                    vals = [baseline]
                    for s in timepoints_seconds_no_zero:
                        vals.append(float(baseline + interp(s)))
                    series[metric_name] = vals

                drugConcs = [_drug_concentration(t, body.dosage, vol, half) for t in times]
                sim = _sanitize({
                    "times": times, "series": series, "baselines": baselines_out,
                    "drugConcs": drugConcs, "patient": p,
                    "drug": {"name": drug_name, "smiles": smiles},
                    "dosage": float(body.dosage), "durationSec": total_seconds,
                })
                improved = 0
                worsened = 0
                metrics_list = []
                for lab_code, vals in series.items():
                    if len(vals) < 2:
                        continue
                    bv = vals[0]
                    fv = vals[-1]
                    d = fv - bv
                    status = _classify_metric(lab_code, fv)
                    if status == "improved":
                        improved += 1
                    elif status == "worsened":
                        worsened += 1
                    metrics_list.append({"labCode": lab_code, "status": status,
                                         "baseline": bv, "finalVal": fv, "delta": d})
                per_patient.append({
                    "patient": p, "sim": sim,
                    "classification": {"metrics": metrics_list, "improvedCount": improved, "worsenedCount": worsened},
                })

            yield f"data: {_json.dumps({'type': 'progress', 'done': batch_end, 'total': N})}\n\n"
            await asyncio.sleep(0)

        result = _build_cohort_response(patients, per_patient, AGE_GROUPS, drug_name, smiles, body.dosage, total_seconds)
        yield f"data: {_json.dumps({'type': 'result', 'data': result})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Serve website: explicit HTML routes + static assets (no html=True to avoid
# the mount swallowing API POSTs with 405)
website_dir = ROOT / "website"
from fastapi.responses import FileResponse

@app.get("/")
async def serve_index():
    return FileResponse(str(website_dir / "index.html"))

@app.get("/simulate")
@app.get("/simulate.html")
async def serve_simulate():
    return FileResponse(str(website_dir / "simulate.html"))

if website_dir.exists():
    app.mount("/", StaticFiles(directory=str(website_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
