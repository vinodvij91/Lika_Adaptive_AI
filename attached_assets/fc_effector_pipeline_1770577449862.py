"""
fc_effector_pipeline.py

Disease- and vaccine-agnostic Fc / effector modeling pipeline for Lika Sciences.

Layers:
- CPU: FcγR/FcRn atlas, ADCC/CDC scoring, species-translation similarity, basic visuals.
- GPU (BioNeMo): optional Fc-affinity prediction from Fc sequence.
- OpenAI: optional structured textual guidance and tooltip/help descriptions for UI.

Environment variables expected (optional but recommended):
- BIONEMO_API_URL
- BIONEMO_API_KEY
- OPENAI_API_KEY
"""

import os
import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px

# Optional heavy libs
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


# -------------------------------------------------
# 0. Device helper (for future torch models)
# -------------------------------------------------

def get_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# -------------------------------------------------
# 1. FcγR / FcRn expression atlas (CPU)
# -------------------------------------------------

def build_fc_atlas() -> pd.DataFrame:
    """
    Simplified human FcγR/FcRn atlas (CPU-bound). [web:8][web:36]
    """
    cell_types = [
        "CD56dim NK",
        "Neutrophils",
        "Classical monocytes",
        "Nonclassical monocytes",
        "cDC2",
        "B cells",
        "Macrophages",
        "LSECs",
    ]

    receptors = ["FcγRI", "FcγRIIA", "FcγRIIB", "FcγRIIIA", "FcγRIIIB", "FcRn"]

    expression_matrix = np.array([
        # FcγRI, FcγRIIA, FcγRIIB, FcγRIIIA, FcγRIIIB, FcRn
        [0.1,   0.1,     0.0,     0.9,      0.0,      0.7],  # CD56dim NK
        [0.0,   0.9,     0.1,     0.8,      0.9,      0.9],  # Neutrophils
        [0.9,   0.8,     0.3,     0.2,      0.0,      0.9],  # Classical monocytes
        [0.3,   0.7,     0.2,     0.6,      0.5,      0.8],  # Nonclassical monocytes
        [0.2,   0.6,     0.4,     0.1,      0.0,      0.8],  # cDC2
        [0.0,   0.1,     0.8,     0.0,      0.0,      0.5],  # B cells
        [0.5,   0.8,     0.6,     0.4,      0.0,      0.9],  # Macrophages
        [0.0,   0.2,     0.6,     0.0,      0.0,      0.7],  # LSECs
    ])

    return pd.DataFrame(expression_matrix, columns=receptors, index=cell_types)


# -------------------------------------------------
# 2. Fc variants and ADCC / CDC scoring (CPU)
# -------------------------------------------------

def build_fc_variants() -> pd.DataFrame:
    """
    Define generic Fc variants and relative ADCC / CDC scores. [web:24][web:28][web:34]
    The real platform can replace these scores with BioNeMo-predicted affinities later.
    """
    variants = ["WT IgG1", "Afucosylated IgG1", "GASDALIE-like", "EFTAE-like (CDC+)"]
    adcc_scores = np.array([0.6, 0.95, 0.9, 0.75])
    cdc_scores  = np.array([0.4, 0.4,  0.3, 0.85])

    return pd.DataFrame({
        "variant_id": variants,
        "ADCC_score": adcc_scores,
        "CDC_score": cdc_scores,
    })


# -------------------------------------------------
# 3. Species translation similarity (CPU)
# -------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_species_similarity() -> pd.DataFrame:
    """
    Compare FcγR profiles for human NK vs WT mouse vs FcγR/FcRn-humanized mouse. [web:8][web:16][web:30][web:33]
    """
    human_nk = np.array([0.1, 0.1, 0.0, 0.9, 0.0])       # [FcγRI, FcγRIIA, FcγRIIB, FcγRIIIA, FcγRIIIB]
    wt_mouse_nk = np.array([0.1, 0.1, 0.1, 0.4, 0.0])
    humanized_mouse_nk = np.array([0.1, 0.1, 0.0, 0.9, 0.1])

    sim_wt = cosine_sim(human_nk, wt_mouse_nk)
    sim_hu = cosine_sim(human_nk, humanized_mouse_nk)

    return pd.DataFrame({
        "model_id": ["wt_mouse", "humanized_fcr_fcrn_mouse"],
        "label": ["WT mouse", "FcγR/FcRn-humanized mouse"],
        "similarity_to_human_NK_FcR": [sim_wt, sim_hu],
    })


# -------------------------------------------------
# 4. BioNeMo hook (GPU-capable, optional)
# -------------------------------------------------

def call_bionemo_fc_affinity(
    fc_sequence: str,
    receptors: Optional[List[str]] = None,
    prefer_gpu: bool = True
) -> Dict[str, float]:
    """
    Placeholder for BioNeMo API call to predict Fc affinities to receptors (e.g., FcγRIIIa, FcRn). [web:49][web:50][web:56][web:59]

    Returns a mapping {receptor_name: predicted_affinity_score} with values in [0,1] (you can
    normalize raw Kd values in your real implementation).

    In this skeleton, we return an empty dict if BioNeMo is not configured.
    """
    api_url = os.getenv("BIONEMO_API_URL")
    api_key = os.getenv("BIONEMO_API_KEY")

    if not (api_url and api_key and REQUESTS_AVAILABLE):
        return {}

    device = get_device(prefer_gpu=prefer_gpu)
    # device could be passed as a hint to BioNeMo if needed (depending on their API).

    payload = {
        "sequence": fc_sequence,
        "receptors": receptors or ["FcγRIIIa", "FcγRIIa", "FcγRIIb", "FcRn"],
        "device_hint": device,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    result = resp.json()

    # Expect result like {"affinities": {"FcγRIIIa": 0.9, "FcγRIIa": 0.7, ...}}
    return result.get("affinities", {})


# -------------------------------------------------
# 5. OpenAI hook (UI guidance, optional)
# -------------------------------------------------

def call_openai_guidance(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use OpenAI to turn numeric outputs into structured UI text: panel titles, tooltips, and
    a per-antibody narrative. [web:54][web:57][web:60]

    payload should contain:
    {
      "disease_or_indication": "...",
      "vaccine_or_therapeutic": "...",
      "fc_variants": [...],
      "effector_scores": [...],
      "species_similarity": [...],
    }
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not (api_key and OPENAI_AVAILABLE):
        return {}

    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are an expert antibody and vaccine discovery assistant for a UI. "
        "Given Fc effector scores and species-translation metrics, you must return "
        "short, precise copy for panels and tooltips. Avoid jargon where possible."
    )

    user_msg = (
        "Here is a JSON payload describing Fc variants, predicted ADCC/CDC scores, "
        "and species-translation similarity metrics for one project in Lika Sciences. "
        "Return a JSON object with:\n"
        "1) 'panel_title' (string),\n"
        "2) 'panel_subtitle' (string),\n"
        "3) 'tooltips' (object with keys 'atlas', 'effector', 'species'),\n"
        "4) 'narrative' (short paragraph for a side bar).\n"
        "Keep everything under 150 words total.\n\n"
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

    # responses.create returns a structured object; extract the text content:
    content = resp.output[0].content[0].text  # type: ignore
    return json.loads(content)


# -------------------------------------------------
# 6. Visualization helpers (CPU)
# -------------------------------------------------

def plot_atlas(df: pd.DataFrame) -> None:
    fig = px.imshow(
        df.T,
        color_continuous_scale="RdYlBu_r",
        labels={"x": "Immune cell type", "y": "Fc receptor", "color": "Expression (0–1)"},
        title="Human FcγR / FcRn expression atlas (simplified)"
    )
    fig.update_layout(height=600)
    fig.show()


def plot_effector(df: pd.DataFrame) -> None:
    df_long = df.melt(id_vars="variant_id", var_name="Effector", value_name="Score")
    fig = px.bar(
        df_long,
        x="variant_id",
        y="Score",
        color="Effector",
        barmode="group",
        range_y=[0, 1],
        title="Predicted ADCC / CDC scores for Fc variants (relative)"
    )
    fig.update_layout(height=500)
    fig.show()


def plot_species(df: pd.DataFrame) -> None:
    fig = px.bar(
        df,
        x="label",
        y="similarity_to_human_NK_FcR",
        range_y=[0, 1],
        color="similarity_to_human_NK_FcR",
        color_continuous_scale="RdYlGn",
        title="FcγR profile similarity to human NK cells<br>Humanized mouse >> WT mouse"
    )
    fig.update_layout(height=400)
    fig.show()


# -------------------------------------------------
# 7. JSON bundle for your TS UI
# -------------------------------------------------

def build_fc_bundle(
    disease_or_indication: str,
    vaccine_or_therapeutic: str = "therapeutic_antibody"
) -> Dict[str, Any]:
    """
    Core CPU function your TS front-end can call via an API.
    Returns JSON with atlas, variant scores, species similarity, and (optionally) OpenAI guidance.
    """
    atlas = build_fc_atlas()
    variants = build_fc_variants()
    species = build_species_similarity()

    bundle: Dict[str, Any] = {
        "context": {
            "disease_or_indication": disease_or_indication,
            "vaccine_or_therapeutic": vaccine_or_therapeutic,
        },
        "atlas": {
            "cell_types": list(atlas.index),
            "receptors": list(atlas.columns),
            "matrix": atlas.values.tolist(),
        },
        "variants": variants.to_dict(orient="records"),
        "species_similarity": species.to_dict(orient="records"),
    }

    # Optional: enrich with OpenAI UI text if configured
    try:
        guidance = call_openai_guidance(bundle)
        bundle["ui_text"] = guidance
    except Exception:
        # Fail silently; UI can still render numeric plots.
        pass

    return bundle


# -------------------------------------------------
# 8. Entry point (for manual testing)
# -------------------------------------------------

if __name__ == "__main__":
    # CPU core logic
    atlas_df = build_fc_atlas()
    variants_df = build_fc_variants()
    species_df = build_species_similarity()

    plot_atlas(atlas_df)
    plot_effector(variants_df)
    plot_species(species_df)

    # Example bundle + (optional) OpenAI guidance
    fc_bundle = build_fc_bundle(disease_or_indication="Example oncology indication")
    print("=== Fc bundle for UI (truncated) ===")
    print(json.dumps({k: v for k, v in fc_bundle.items() if k != "atlas"}, indent=2)[:1200])
