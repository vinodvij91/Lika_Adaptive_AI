"""
fc_effector_pipeline.py

Disease- and vaccine-agnostic Fc / effector modeling pipeline for Lika Sciences.

Layers:
  CPU  : FcgammaR/FcRn atlas, ADCC/CDC scoring, species-translation similarity.
  GPU  : BioNeMo Fc-receptor affinity prediction (optional, CUDA-preferred).
  LLM  : OpenAI structured guidance for UI text (optional).

Environment variables (all optional - graceful no-op when absent):
  BIONEMO_API_KEY   - NVIDIA BioNeMo NIM API key
  OPENAI_API_KEY    - OpenAI API key for UI guidance text

Standardized CLI contract:
  python3 fc_effector_pipeline.py --job-type <type> --params '<json>' [--params-file <path>] [--output <path>]

Job types:
  build_atlas              - Build FcgammaR/FcRn expression atlas
  build_variants           - Build Fc variant ADCC/CDC scores
  build_species_similarity - Build species translation similarity
  bionemo_fc_affinity      - Predict Fc-receptor affinities via BioNeMo
  openai_guidance          - Generate UI guidance text via OpenAI
  build_fc_bundle          - Full bundle (atlas + variants + species + optional AI)
  plot_atlas               - Generate atlas heatmap (returns Plotly JSON)
  plot_effector            - Generate effector bar chart (returns Plotly JSON)
  plot_species             - Generate species similarity chart (returns Plotly JSON)
  full_pipeline            - Alias for build_fc_bundle
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import requests as _requests_lib
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

BIONEMO_API_BASE = "https://health.api.nvidia.com/v1"


def get_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_fc_atlas() -> pd.DataFrame:
    """
    Simplified human FcgammaR / FcRn expression atlas (CPU-bound).
    Values represent relative expression levels on [0, 1].
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

    receptors = [
        "FcgammaRI", "FcgammaRIIA", "FcgammaRIIB",
        "FcgammaRIIIA", "FcgammaRIIIB", "FcRn",
    ]

    expression_matrix = np.array([
        [0.1, 0.1, 0.0, 0.9, 0.0, 0.7],
        [0.0, 0.9, 0.1, 0.8, 0.9, 0.9],
        [0.9, 0.8, 0.3, 0.2, 0.0, 0.9],
        [0.3, 0.7, 0.2, 0.6, 0.5, 0.8],
        [0.2, 0.6, 0.4, 0.1, 0.0, 0.8],
        [0.0, 0.1, 0.8, 0.0, 0.0, 0.5],
        [0.5, 0.8, 0.6, 0.4, 0.0, 0.9],
        [0.0, 0.2, 0.6, 0.0, 0.0, 0.7],
    ], dtype=np.float64)

    return pd.DataFrame(expression_matrix, columns=receptors, index=cell_types)


def build_fc_variants() -> pd.DataFrame:
    """
    Define generic Fc variants with relative ADCC / CDC scores (CPU-bound).
    Scores can later be overridden with BioNeMo-predicted affinities.
    """
    variants = ["WT IgG1", "Afucosylated IgG1", "GASDALIE-like", "EFTAE-like (CDC+)"]
    adcc_scores = np.array([0.60, 0.95, 0.90, 0.75], dtype=np.float64)
    cdc_scores = np.array([0.40, 0.40, 0.30, 0.85], dtype=np.float64)

    return pd.DataFrame({
        "variant_id": variants,
        "ADCC_score": adcc_scores,
        "CDC_score": cdc_scores,
    })


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_species_similarity() -> pd.DataFrame:
    """
    Compare FcgammaR profiles for human NK vs WT mouse vs
    FcgammaR/FcRn-humanized mouse (CPU-bound).
    """
    human_nk = np.array([0.1, 0.1, 0.0, 0.9, 0.0])
    wt_mouse_nk = np.array([0.1, 0.1, 0.1, 0.4, 0.0])
    humanized_mouse_nk = np.array([0.1, 0.1, 0.0, 0.9, 0.1])

    return pd.DataFrame({
        "model_id": ["wt_mouse", "humanized_fcr_fcrn_mouse"],
        "label": ["WT mouse", "FcgammaR/FcRn-humanized mouse"],
        "similarity_to_human_NK_FcR": [
            round(_cosine_sim(human_nk, wt_mouse_nk), 6),
            round(_cosine_sim(human_nk, humanized_mouse_nk), 6),
        ],
    })


def call_bionemo_fc_affinity(
    fc_sequence: str,
    receptors: Optional[List[str]] = None,
    prefer_gpu: bool = True,
) -> Dict[str, float]:
    """
    GPU-capable hook: call BioNeMo NIM API to predict normalised
    Fc-receptor and FcRn affinities from an Fc amino-acid sequence.

    Prefers CUDA when torch is available and a GPU is detected,
    but works on CPU-only machines.  Returns {} when credentials
    are missing (graceful no-op).
    """
    api_key = os.getenv("BIONEMO_API_KEY")
    if not api_key:
        return {}
    if not REQUESTS_AVAILABLE:
        return {}

    device = get_device(prefer_gpu=prefer_gpu)
    target_receptors = receptors or [
        "FcgammaRIIIa", "FcgammaRIIa", "FcgammaRIIb", "FcRn",
    ]

    payload = {
        "sequence": fc_sequence,
        "receptors": target_receptors,
        "device_hint": device,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        api_url = os.getenv("BIONEMO_API_URL", f"{BIONEMO_API_BASE}/biology/nvidia/esmfold")
        resp = _requests_lib.post(
            api_url, headers=headers,
            data=json.dumps(payload), timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("affinities", {})
    except Exception as exc:
        return {"_error": str(exc)}


def call_openai_guidance(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI structured-output hook: turn numeric Fc outputs into
    UI-ready JSON (panel_title, panel_subtitle, tooltips, narrative).

    Returns {} when OPENAI_API_KEY is missing (graceful no-op).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not OPENAI_AVAILABLE:
        return {}

    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are an expert antibody and vaccine discovery assistant for a scientific UI. "
        "Given Fc effector scores and species-translation metrics, return short, precise copy "
        "for panels and tooltips. Minimise jargon."
    )

    user_msg = (
        "Below is a JSON payload describing Fc variants, predicted ADCC/CDC scores, "
        "and species-translation similarity metrics for one project in Lika Sciences.\n"
        "Return a JSON object with exactly these keys:\n"
        '  "panel_title"    (string, max 10 words),\n'
        '  "panel_subtitle" (string, max 20 words),\n'
        '  "tooltips"       (object with keys "atlas", "effector", "species" - each a concise sentence),\n'
        '  "narrative"      (short paragraph, max 100 words, explaining how the Fc data should be '
        "interpreted for the given disease or vaccine pipeline).\n\n"
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
            max_tokens=512,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception:
        return {}


def _plotly_atlas_json(df: pd.DataFrame) -> Dict[str, Any]:
    import plotly.express as px
    fig = px.imshow(
        df.T,
        color_continuous_scale="RdYlBu_r",
        labels={"x": "Immune cell type", "y": "Fc receptor", "color": "Expression (0-1)"},
        title="Human FcgammaR / FcRn Expression Atlas",
    )
    fig.update_layout(height=600)
    return json.loads(fig.to_json())


def _plotly_effector_json(df: pd.DataFrame) -> Dict[str, Any]:
    import plotly.express as px
    df_long = df.melt(id_vars="variant_id", var_name="Effector", value_name="Score")
    fig = px.bar(
        df_long, x="variant_id", y="Score", color="Effector",
        barmode="group", range_y=[0, 1],
        title="Predicted ADCC / CDC Scores for Fc Variants",
    )
    fig.update_layout(height=500)
    return json.loads(fig.to_json())


def _plotly_species_json(df: pd.DataFrame) -> Dict[str, Any]:
    import plotly.express as px
    fig = px.bar(
        df, x="label", y="similarity_to_human_NK_FcR",
        range_y=[0, 1],
        color="similarity_to_human_NK_FcR",
        color_continuous_scale="RdYlGn",
        title="FcgammaR Profile Similarity to Human NK Cells",
    )
    fig.update_layout(height=400)
    return json.loads(fig.to_json())


def plot_atlas(df: pd.DataFrame) -> None:
    import plotly.express as px
    fig = px.imshow(
        df.T,
        color_continuous_scale="RdYlBu_r",
        labels={"x": "Immune cell type", "y": "Fc receptor", "color": "Expression (0-1)"},
        title="Human FcgammaR / FcRn Expression Atlas",
    )
    fig.update_layout(height=600)
    fig.show()


def plot_effector(df: pd.DataFrame) -> None:
    import plotly.express as px
    df_long = df.melt(id_vars="variant_id", var_name="Effector", value_name="Score")
    fig = px.bar(
        df_long, x="variant_id", y="Score", color="Effector",
        barmode="group", range_y=[0, 1],
        title="Predicted ADCC / CDC Scores for Fc Variants",
    )
    fig.update_layout(height=500)
    fig.show()


def plot_species(df: pd.DataFrame) -> None:
    import plotly.express as px
    fig = px.bar(
        df, x="label", y="similarity_to_human_NK_FcR",
        range_y=[0, 1],
        color="similarity_to_human_NK_FcR",
        color_continuous_scale="RdYlGn",
        title="FcgammaR Profile Similarity to Human NK Cells",
    )
    fig.update_layout(height=400)
    fig.show()


def build_fc_bundle(
    disease_or_indication: str,
    vaccine_or_therapeutic: str = "therapeutic_antibody",
    fc_sequence: Optional[str] = None,
    include_openai: bool = True,
    include_bionemo: bool = True,
    prefer_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Core CPU function. Returns a single JSON-serialisable dict with:
      context, atlas, variants, species_similarity, and optional ui_text / bionemo_affinities.
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

    if include_bionemo and fc_sequence:
        affinities = call_bionemo_fc_affinity(
            fc_sequence, prefer_gpu=prefer_gpu,
        )
        if affinities:
            bundle["bionemo_affinities"] = affinities

    if include_openai:
        try:
            guidance = call_openai_guidance({
                "disease_or_indication": disease_or_indication,
                "vaccine_or_therapeutic": vaccine_or_therapeutic,
                "fc_variants": bundle["variants"],
                "species_similarity": bundle["species_similarity"],
            })
            if guidance:
                bundle["ui_text"] = guidance
        except Exception:
            pass

    return bundle


def run_step(job_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Standardised dispatcher matching Lika Sciences CLI contract."""
    try:
        if job_type == "build_atlas":
            df = build_fc_atlas()
            return {
                "step": job_type, "success": True,
                "output": {
                    "cell_types": list(df.index),
                    "receptors": list(df.columns),
                    "matrix": df.values.tolist(),
                },
                "error": None,
            }

        elif job_type == "build_variants":
            df = build_fc_variants()
            return {
                "step": job_type, "success": True,
                "output": df.to_dict(orient="records"),
                "error": None,
            }

        elif job_type == "build_species_similarity":
            df = build_species_similarity()
            return {
                "step": job_type, "success": True,
                "output": df.to_dict(orient="records"),
                "error": None,
            }

        elif job_type == "bionemo_fc_affinity":
            seq = params.get("fc_sequence", "")
            receptors = params.get("receptors")
            prefer_gpu = params.get("prefer_gpu", True)
            affinities = call_bionemo_fc_affinity(seq, receptors, prefer_gpu)
            return {
                "step": job_type, "success": True,
                "output": affinities,
                "error": None,
            }

        elif job_type == "openai_guidance":
            guidance = call_openai_guidance(params)
            return {
                "step": job_type, "success": True,
                "output": guidance,
                "error": None,
            }

        elif job_type in ("build_fc_bundle", "full_pipeline"):
            bundle = build_fc_bundle(
                disease_or_indication=params.get("disease_or_indication", "general"),
                vaccine_or_therapeutic=params.get("vaccine_or_therapeutic", "therapeutic_antibody"),
                fc_sequence=params.get("fc_sequence"),
                include_openai=params.get("include_openai", True),
                include_bionemo=params.get("include_bionemo", True),
                prefer_gpu=params.get("prefer_gpu", True),
            )
            return {
                "step": job_type, "success": True,
                "output": bundle,
                "error": None,
            }

        elif job_type == "plot_atlas":
            df = build_fc_atlas()
            return {
                "step": job_type, "success": True,
                "output": _plotly_atlas_json(df),
                "error": None,
            }

        elif job_type == "plot_effector":
            df = build_fc_variants()
            return {
                "step": job_type, "success": True,
                "output": _plotly_effector_json(df),
                "error": None,
            }

        elif job_type == "plot_species":
            df = build_species_similarity()
            return {
                "step": job_type, "success": True,
                "output": _plotly_species_json(df),
                "error": None,
            }

        else:
            return {
                "step": job_type, "success": False,
                "output": None,
                "error": f"Unknown job type: {job_type}",
            }

    except Exception as exc:
        return {
            "step": job_type, "success": False,
            "output": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def cli_main():
    """Standardized CLI entry point: --job-type --params [--params-file] [--output]."""
    import argparse as ap

    parser = ap.ArgumentParser(description="Lika Sciences Fc Effector Pipeline - CLI")
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
        atlas_df = build_fc_atlas()
        variants_df = build_fc_variants()
        species_df = build_species_similarity()

        plot_atlas(atlas_df)
        plot_effector(variants_df)
        plot_species(species_df)

        fc_bundle = build_fc_bundle(disease_or_indication="Example oncology indication")
        print("=== Fc bundle for UI ===")
        print(json.dumps(
            {k: v for k, v in fc_bundle.items() if k != "atlas"},
            indent=2, default=str,
        )[:1200])
