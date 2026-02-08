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

This module is paired conceptually with fc_effector_pipeline.py and is
designed to be plugged into all ~350 disease pipelines and all vaccine
pipelines in Lika Sciences.
"""

import os
import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Optional heavy deps
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


def build_omics_evidence_for_target(
    disease_id: str,
    target_id: str,
) -> Dict[str, Any]:
    """
    Build a synthetic but structurally realistic multi-omics evidence record
    for (disease, target).

    In production, this function should:
      - query external resources (GWAS, GTEx, HPA, PRIDE, metabolomics repositories),
      - aggregate and normalize scores.

    Here we create mock data with the right structure and clearly label where
    real data should be injected. [web:65][web:67][web:70][web:73][web:76][web:79]
    """
    # --- genomics (e.g., GWAS p-value → 0–1 score) ---
    mock_gwas_p = 1e-6  # placeholder
    gwas_score = 1.0 - _normalize(-np.log10(mock_gwas_p), 0, 20)

    genomics = {
        "gwas_score": float(gwas_score),
        "variant_count": 5,
        "pathogenic_variants": 1,
    }

    # --- transcriptomics (log2 fold change and p-value) ---
    log2_fc = 1.5       # placeholder: upregulated
    p_val = 1e-4
    transcriptomics = {
        "log2_fc": float(log2_fc),
        "p_value": float(p_val),
        "tissues": ["brain", "blood"],
    }

    # --- proteomics (protein-level evidence) ---
    protein_fc = 1.2    # placeholder
    detection_evidence = "strong"
    proteomics = {
        "protein_fc": float(protein_fc),
        "detection_evidence": detection_evidence,
        "ptm_flags": ["phosphorylated"],
    }

    # --- metabolomics (pathway-level impact, optional) ---
    pathway_score = 0.6  # placeholder
    metabolomics = {
        "pathway_score": float(pathway_score),
    }

    # --- integrated score (simple example heuristic) ---
    # You can replace this with a learned model on real multi-omics datasets. [web:65][web:74][web:77]
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
        "integrated_score": float(integrated_score),
    }

    return evidence


def build_omics_table(
    disease_id: str,
    target_ids: List[str],
) -> pd.DataFrame:
    """
    CPU-bound helper to build a table of omics evidence for a list of targets.

    Each row corresponds to one (disease, target) combination, and can be
    used for sorting and ranking inside any of your ~350 disease pipelines
    or vaccine host-response pipelines. [web:65][web:68][web:74][web:77]
    """
    records = []
    for tid in target_ids:
        e = build_omics_evidence_for_target(disease_id, tid)
        records.append({
            "disease_id": disease_id,
            "target_id": tid,
            "integrated_score": e["integrated_score"],
            "gwas_score": e["genomics"]["gwas_score"],
            "tx_log2_fc": e["transcriptomics"]["log2_fc"],
            "protein_fc": e["proteomics"]["protein_fc"],
            "metab_pathway_score": e["metabolomics"]["pathway_score"],
        })

    df = pd.DataFrame.from_records(records)
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
    Placeholder for a BioNeMo API call to compute structure-/sequence-based
    properties of a target or antigen (e.g. stability, disorder, aggregation,
    epitope propensity), which can enrich omics evidence. [web:49][web:50][web:56][web:59]

    Returns a mapping {property_name: normalized_score}, all 0–1.

    In this skeleton, returns empty dict if no BioNeMo configuration is present.
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

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=90)
    resp.raise_for_status()
    result = resp.json()

    # Expect result like {"properties": {"stability": 0.8, "disorder": 0.3, ...}}
    return result.get("properties", {})


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
      "targets": [...],                       # each with integrated_score and components
      "optional_sequence_properties": {...}   # from BioNeMo, if available
    }

    Returns JSON with:
      - panel_title
      - panel_subtitle
      - tooltips: { table, genomics, transcriptomics, proteomics, metabolomics }
      - narrative
    [web:54][web:57][web:60][web:68][web:74]
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
        f"Payload:\n{json.dumps(payload)}"
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    content = resp.output[0].content[0].text  # type: ignore
    return json.loads(content)


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
      - omics_table (flattened per target)
      - sequence_properties (if BioNeMo is available)
      - ui_text (if OpenAI is available)
    """
    # CPU: build table of integrated omics evidence
    df = build_omics_table(disease_or_indication, target_ids)

    omics_records = df.to_dict(orient="records")

    # Optional BioNeMo enrichment (GPU / external)
    sequence_properties: Dict[str, Dict[str, float]] = {}
    if target_sequences:
        for tid, seq in target_sequences.items():
            try:
                props = call_bionemo_sequence_properties(seq)
                if props:
                    sequence_properties[tid] = props
            except Exception:
                # Fail gracefully; keep going for other targets
                continue

    bundle: Dict[str, Any] = {
        "context": {
            "disease_or_indication": disease_or_indication,
            "vaccine_or_therapeutic": vaccine_or_therapeutic,
        },
        "omics_table": omics_records,
        "sequence_properties": sequence_properties,
    }

    # Optional OpenAI UI guidance
    guidance_payload = {
        "disease_or_indication": disease_or_indication,
        "vaccine_or_therapeutic": vaccine_or_therapeutic,
        "targets": omics_records,
        "optional_sequence_properties": sequence_properties,
    }

    try:
        ui_text = call_openai_omics_guidance(guidance_payload)
        bundle["ui_text"] = ui_text
    except Exception:
        # Platform remains functional even without OpenAI
        pass

    return bundle


# ============================================================
# 5. Entry point for manual testing (no UI)
# ============================================================

if __name__ == "__main__":
    # Example usage for any disease/vaccine pipeline
    example_disease = "Generic inflammatory disease"
    example_targets = ["TNF", "IL6", "FCGR2A"]

    bundle = build_omics_bundle(
        disease_or_indication=example_disease,
        target_ids=example_targets,
        vaccine_or_therapeutic="therapeutic_antibody",
        target_sequences=None,  # add sequences when you want BioNeMo enrichment
    )

    print("=== Omics bundle (truncated) ===")
    print(json.dumps(bundle, indent=2)[:1500])
