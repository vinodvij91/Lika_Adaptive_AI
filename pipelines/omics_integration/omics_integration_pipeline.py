"""
omics_integration_pipeline.py

General, disease- and vaccine-agnostic multi-omics integration engine
for Lika Sciences.

Layers
------
CPU:
  - Integrate genomics, transcriptomics, proteomics, metabolomics evidence.
  - Compute an integrated omics score per (disease, target) or pathway.
  - Package results as JSON bundles for Drug Discovery & Vaccine Discovery UIs.

GPU (BioNeMo, optional):
  - Predict structure-/sequence-based properties from protein/peptide sequences
    (e.g., stability, disorder, binding propensity) to enrich omics evidence.

OpenAI (optional):
  - Turn numeric omics outputs into concise, UI-ready panel titles, tooltips,
    and short narratives per disease/vaccine pipeline.

Environment variables (optional)
--------------------------------
- BIONEMO_API_URL          (for sequence/property prediction)
- BIONEMO_API_KEY
- OPENAI_API_KEY

Job types (standardized CLI)
----------------------------
- build_bundle           : full omics bundle for a disease/vaccine pipeline
- build_table            : omics evidence table only (CPU)
- bionemo_enrich         : BioNeMo sequence property enrichment (GPU optional)
- openai_guidance        : OpenAI UI text generation
- full_pipeline          : all of the above combined

This module is paired conceptually with fc_effector_pipeline.py and is
designed to be plugged into all ~360 disease pipelines and all 10 vaccine
pipelines in Lika Sciences.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================
# 0. Device helper (GPU vs CPU separation)
# ============================================================

def get_device(prefer_gpu: bool = True) -> str:
    """
    Decide whether to use 'cuda' or 'cpu' for GPU-bound tasks.

    All core multi-omics aggregation and scoring remains CPU-bound and
    does NOT depend on this function.
    """
    if prefer_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# 1. CPU-bound multi-omics integration core
# ============================================================

def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.0
    return float(max(0.0, min(1.0, (value - min_val) / (max_val - min_val))))


def _deterministic_seed(disease_id: str, target_id: str) -> int:
    """Produce a stable integer seed from disease + target so the same
    (disease, target) pair always yields the same simulated evidence."""
    return hash(f"{disease_id}::{target_id}") & 0x7FFFFFFF


def build_omics_evidence_for_target(
    disease_id: str,
    target_id: str,
) -> Dict[str, Any]:
    """
    Build a structurally realistic multi-omics evidence record for
    (disease, target).

    In production this function should query external resources
    (GWAS Catalog, GTEx, Human Protein Atlas, PRIDE, metabolomics
    repositories) and aggregate/normalize scores.

    The current implementation uses deterministic simulated values so
    the output is stable and reproducible per (disease, target) pair.
    """
    rng = np.random.RandomState(_deterministic_seed(disease_id, target_id))

    gwas_p = 10 ** (-rng.uniform(2, 12))
    gwas_score = 1.0 - _normalize(-np.log10(gwas_p), 0, 20)
    variant_count = int(rng.randint(0, 15))
    pathogenic_variants = int(rng.randint(0, max(1, variant_count // 3 + 1)))

    genomics = {
        "gwas_score": float(gwas_score),
        "variant_count": variant_count,
        "pathogenic_variants": pathogenic_variants,
    }

    log2_fc = float(rng.uniform(-3, 4))
    tx_p_val = 10 ** (-rng.uniform(1, 8))
    tissue_pool = ["brain", "blood", "liver", "lung", "kidney", "heart",
                   "spleen", "bone_marrow", "lymph_node", "gut"]
    n_tissues = int(rng.randint(1, 4))
    tissues = list(rng.choice(tissue_pool, size=n_tissues, replace=False))

    transcriptomics = {
        "log2_fc": float(log2_fc),
        "p_value": float(tx_p_val),
        "tissues": tissues,
    }

    protein_fc = float(rng.uniform(0.3, 2.5))
    detection_pool = ["strong", "moderate", "weak"]
    detection_evidence = str(rng.choice(detection_pool))
    ptm_pool = ["phosphorylated", "ubiquitinated", "acetylated",
                "glycosylated", "methylated"]
    n_ptms = int(rng.randint(0, 3))
    ptm_flags = list(rng.choice(ptm_pool, size=n_ptms, replace=False)) if n_ptms else []

    proteomics = {
        "protein_fc": float(protein_fc),
        "detection_evidence": detection_evidence,
        "ptm_flags": ptm_flags,
    }

    pathway_score = float(rng.uniform(0.1, 0.95))
    metabolomics = {
        "pathway_score": pathway_score,
    }

    weights = {
        "genomics": 0.35,
        "transcriptomics": 0.30,
        "proteomics": 0.25,
        "metabolomics": 0.10,
    }

    genomics_norm = genomics["gwas_score"]
    tx_norm = _normalize(abs(transcriptomics["log2_fc"]), 0, 4)
    prot_norm = _normalize(abs(proteomics["protein_fc"]), 0, 3)
    metab_norm = metabolomics["pathway_score"]

    integrated_score = (
        weights["genomics"] * genomics_norm
        + weights["transcriptomics"] * tx_norm
        + weights["proteomics"] * prot_norm
        + weights["metabolomics"] * metab_norm
    )

    evidence = {
        "disease_id": disease_id,
        "target_id": target_id,
        "genomics": genomics,
        "transcriptomics": transcriptomics,
        "proteomics": proteomics,
        "metabolomics": metabolomics,
        "integrated_score": round(float(integrated_score), 4),
    }

    return evidence


def build_omics_table(
    disease_id: str,
    target_ids: List[str],
) -> pd.DataFrame:
    """
    CPU-bound helper to build a table of omics evidence for a list of targets.

    Each row corresponds to one (disease, target) combination, and can be
    used for sorting and ranking inside any disease or vaccine pipeline.
    """
    records = []
    for tid in target_ids:
        e = build_omics_evidence_for_target(disease_id, tid)
        records.append({
            "disease_id": disease_id,
            "target_id": tid,
            "integrated_score": e["integrated_score"],
            "gwas_score": e["genomics"]["gwas_score"],
            "variant_count": e["genomics"]["variant_count"],
            "pathogenic_variants": e["genomics"]["pathogenic_variants"],
            "tx_log2_fc": e["transcriptomics"]["log2_fc"],
            "tx_p_value": e["transcriptomics"]["p_value"],
            "tx_tissues": json.dumps(e["transcriptomics"]["tissues"]),
            "protein_fc": e["proteomics"]["protein_fc"],
            "detection_evidence": e["proteomics"]["detection_evidence"],
            "ptm_flags": json.dumps(e["proteomics"]["ptm_flags"]),
            "metab_pathway_score": e["metabolomics"]["pathway_score"],
        })

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values("integrated_score", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# 2. GPU-bound BioNeMo hook (structure/sequence properties)
# ============================================================

def call_bionemo_sequence_properties(
    protein_sequence: str,
    properties: Optional[List[str]] = None,
    prefer_gpu: bool = True,
) -> Dict[str, float]:
    """
    Call BioNeMo API to compute structure-/sequence-based properties
    of a target or antigen (e.g. stability, disorder, aggregation,
    epitope propensity), which can enrich omics evidence.

    Returns a mapping {property_name: normalized_score}, all 0-1.
    Returns empty dict if no BioNeMo configuration is present.
    Fails gracefully on CPU-only machines.
    """
    api_url = os.getenv("BIONEMO_API_URL")
    api_key = os.getenv("BIONEMO_API_KEY")

    if not (api_url and api_key and REQUESTS_AVAILABLE):
        return {}

    device = get_device(prefer_gpu=prefer_gpu)

    payload = {
        "sequence": protein_sequence,
        "properties": properties or ["stability", "disorder", "aggregation"],
        "device_hint": device,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            api_url, headers=headers,
            data=json.dumps(payload), timeout=90,
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("properties", {})
    except Exception:
        return {}


# ============================================================
# 3. OpenAI hook (UI narratives & tooltips)
# ============================================================

def call_openai_omics_guidance(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use OpenAI to convert multi-omics scores into structured UI text for Lika.

    payload should contain at least:
    {
      "disease_or_indication": "...",
      "vaccine_or_therapeutic": "...",
      "targets": [...],
      "optional_sequence_properties": {...}
    }

    Returns JSON with:
      - panel_title
      - panel_subtitle
      - tooltips: { table, genomics, transcriptomics, proteomics, metabolomics }
      - narrative
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not (api_key and OPENAI_AVAILABLE):
        return {}

    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are an expert in AI-driven multi-omics drug and vaccine discovery. "
        "Given integrated omics scores for targets across any disease, you must "
        "produce concise UI text: panel titles, tooltips for different omics layers, "
        "and a short narrative for scientists. Keep explanations clear and non-redundant."
    )

    user_msg = (
        "Here is a JSON payload describing omics-based evidence for several targets "
        "in a disease or vaccine pipeline in the Lika Sciences platform. "
        "Return a JSON object with:\n"
        "1) 'panel_title' (string, <= 60 characters),\n"
        "2) 'panel_subtitle' (string, <= 120 characters),\n"
        "3) 'tooltips' (object with keys 'table', 'genomics', 'transcriptomics', "
        "'proteomics', 'metabolomics'),\n"
        "4) 'narrative' (one short paragraph, <= 120 words) explaining how "
        "multi-omics is used to rank targets or understand host response.\n\n"
        f"Payload:\n{json.dumps(payload, default=str)}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=600,
        )

        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception:
        return {}


# ============================================================
# 4. JSON bundle for TS UI / APIs (CPU core)
# ============================================================

def build_omics_bundle(
    disease_or_indication: str,
    target_ids: List[str],
    vaccine_or_therapeutic: str = "therapeutic_antibody",
    target_sequences: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Main CPU-bound function your backend can call to build a JSON bundle
    for any disease or vaccine pipeline.

    - disease_or_indication: human-readable label or ID.
    - target_ids: list of target IDs (genes/proteins) under consideration.
    - vaccine_or_therapeutic: generic label ("vaccine", "therapeutic_antibody", etc.).
    - target_sequences: optional map {target_id: protein_sequence} for BioNeMo enrichment.

    Returns a dict containing:
      - context
      - omics_table (flattened per target, sorted by integrated_score descending)
      - sequence_properties (if BioNeMo is available)
      - ui_text (if OpenAI is available)
    """
    df = build_omics_table(disease_or_indication, target_ids)

    omics_records = df.to_dict(orient="records")

    sequence_properties: Dict[str, Dict[str, float]] = {}
    if target_sequences:
        for tid, seq in target_sequences.items():
            try:
                props = call_bionemo_sequence_properties(seq)
                if props:
                    sequence_properties[tid] = props
            except Exception:
                continue

    bundle: Dict[str, Any] = {
        "context": {
            "disease_or_indication": disease_or_indication,
            "vaccine_or_therapeutic": vaccine_or_therapeutic,
        },
        "omics_table": omics_records,
        "sequence_properties": sequence_properties,
    }

    guidance_payload = {
        "disease_or_indication": disease_or_indication,
        "vaccine_or_therapeutic": vaccine_or_therapeutic,
        "targets": omics_records,
        "optional_sequence_properties": sequence_properties,
    }

    try:
        ui_text = call_openai_omics_guidance(guidance_payload)
        if ui_text:
            bundle["ui_text"] = ui_text
    except Exception:
        pass

    return bundle


# ============================================================
# 5. Standardized CLI runner (--job-type --params)
# ============================================================

def run_step(job_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single pipeline step and return a standard envelope."""
    try:
        if job_type == "build_table":
            disease = params.get("disease_or_indication", "unknown_disease")
            targets = params.get("target_ids", [])
            df = build_omics_table(disease, targets)
            return {
                "step": job_type,
                "success": True,
                "output": {
                    "rows": len(df),
                    "table": df.to_dict(orient="records"),
                },
                "error": None,
            }

        elif job_type == "bionemo_enrich":
            sequences = params.get("target_sequences", {})
            prefer_gpu = params.get("prefer_gpu", True)
            results: Dict[str, Any] = {}
            for tid, seq in sequences.items():
                try:
                    props = call_bionemo_sequence_properties(
                        seq, prefer_gpu=prefer_gpu,
                    )
                    results[tid] = props
                except Exception as exc:
                    results[tid] = {"error": str(exc)}
            return {
                "step": job_type,
                "success": True,
                "output": results,
                "error": None,
            }

        elif job_type == "openai_guidance":
            ui_text = call_openai_omics_guidance(params)
            return {
                "step": job_type,
                "success": True,
                "output": ui_text,
                "error": None,
            }

        elif job_type in ("build_bundle", "full_pipeline"):
            disease = params.get("disease_or_indication", "unknown_disease")
            targets = params.get("target_ids", [])
            v_or_t = params.get("vaccine_or_therapeutic", "therapeutic_antibody")
            seqs = params.get("target_sequences", None)
            bundle = build_omics_bundle(disease, targets, v_or_t, seqs)
            return {
                "step": job_type,
                "success": True,
                "output": bundle,
                "error": None,
            }

        else:
            return {
                "step": job_type,
                "success": False,
                "output": None,
                "error": f"Unknown job type: {job_type}",
            }

    except Exception as exc:
        return {
            "step": job_type,
            "success": False,
            "output": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def cli_main():
    """Standardized CLI entry point: --job-type --params [--params-file] [--output]."""
    import argparse as ap

    parser = ap.ArgumentParser(
        description="Lika Sciences Omics Integration Pipeline - CLI",
    )
    parser.add_argument("--job-type", required=True, help="Job type to execute")
    parser.add_argument("--params", default="{}", help="JSON params string")
    parser.add_argument("--params-file", default=None, help="Path to JSON params file")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
    else:
        params = json.loads(args.params)

    result = run_step(args.job_type, params)

    output_json = json.dumps(result, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
    print(output_json)


if __name__ == "__main__":
    if len(sys.argv) > 1 and "--job-type" in sys.argv:
        cli_main()
    else:
        example_disease = "Generic inflammatory disease"
        example_targets = ["TNF", "IL6", "FCGR2A", "JAK2", "STAT3"]

        bundle = build_omics_bundle(
            disease_or_indication=example_disease,
            target_ids=example_targets,
            vaccine_or_therapeutic="therapeutic_antibody",
            target_sequences=None,
        )

        print("=== Omics bundle ===")
        print(json.dumps(bundle, indent=2, default=str))
