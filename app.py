import os
import re
import io
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter
import streamlit as st
from streamlit_sortables import sort_items

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="CV Analyzer", layout="wide")
st.title("📊 Cyclic Voltammetry (CV) Analyzer")
st.markdown("Upload your **Gamry (.DTA)** or **Biologic (.mpt)** files.")

# ============================================================
# INSTRUMENT SELECTION
# ============================================================
instrument = st.selectbox(
    "Select instrument format:",
    ["Gamry 1010B (.DTA)", "Biologic SP-50e (.mpt)"]
)

# ============================================================
# UTILS
# ============================================================
def mad_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float(np.std(x)) if len(x) else np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else float(np.std(x))

def _to_float(x):
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None

# ============================================================
# GAMRY PARSER
# ============================================================
def parse_gamry_dta_multi_curve(raw: str) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    lines = raw.splitlines()
    meta = {}
    first_curve_idx = None
    
    for i, line in enumerate(lines):
        if re.match(r"^\s*CURVE\d*\s+TABLE\b", line, flags=re.IGNORECASE):
            first_curve_idx = i
            break
        if "\t" in line:
            parts = line.split("\t")
            if parts[0].strip():
                meta[parts[0].strip()] = parts[-1].strip()

    if first_curve_idx is None:
        return meta, []

    curves = []
    i = first_curve_idx

    while i < len(lines):
        if not re.match(r"^\s*CURVE\d*\s+TABLE\b", lines[i], flags=re.IGNORECASE):
            i += 1
            continue

        curve_id = f"Curve {len(curves)+1}"

        j = i + 1
        while j < len(lines):
            if "Pt" in lines[j] and "Im" in lines[j]:
                break
            j += 1

        cols = [c.strip() for c in lines[j].split("\t")]
        data_start = j + 1

        rows = []
        k = data_start
        while k < len(lines):
            if re.match(r"^\s*CURVE\d*\s+TABLE\b", lines[k]):
                break
            parts = lines[k].split("\t")
            if len(parts) >= len(cols):
                rows.append(parts[:len(cols)])
            k += 1

        df = pd.DataFrame(rows, columns=cols)

        for c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

        df = df.dropna(how="all").reset_index(drop=True)

        curves.append((curve_id, df))
        i = k

    return meta, curves

# ============================================================
# BIOLOGIC PARSER
# ============================================================
def parse_biologic_mpt(raw: str) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    lines = raw.splitlines()
    meta = {}
    header_lines = 0

    for line in lines:
        if "Nb header lines" in line:
            header_lines = int(line.split(":")[-1].strip())
            break

    for i in range(header_lines):
        if ":" in lines[i]:
            k, v = lines[i].split(":", 1)
            meta[k.strip()] = v.strip()

    data_lines = lines[header_lines:]
    cols = [c.strip() for c in data_lines[0].split("\t")]

    rows = []
    for line in data_lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) == len(cols):
            rows.append(parts)

    df = pd.DataFrame(rows, columns=cols)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

    df = df.dropna(how="all").reset_index(drop=True)

    # Normalización de columnas
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "ewe" in cl or "potential" in cl:
            col_map[c] = "Vf"
        elif "<i>" in cl or "current" in cl:
            col_map[c] = "Im"

    df = df.rename(columns=col_map)

    # Convertir mA → A si es necesario
    if "Im" in df.columns and df["Im"].abs().max() > 1:
        df["Im"] = df["Im"] / 1000

    curves = [("Curve 1", df)]
    return meta, curves

# ============================================================
# ANALYSIS FUNCTION (sin cambios)
# ============================================================
def recommend_operating_ranges_for_curve(df_curve):
    Ecol = "Vf" if "Vf" in df_curve.columns else ("Vu" if "Vu" in df_curve.columns else None)
    df = df_curve[[Ecol, "Im"]].dropna().copy()
    df.columns = ["E", "I"]

    if len(df) < 20:
        return None

    E = df["E"].values
    I = df["I"].values

    Is = savgol_filter(I, 101, 3)
    resid = I - Is

    sigma = np.std(resid)
    threshold = 1.5 * sigma

    mask = np.abs(resid) < threshold

    safe_E = E[mask]

    return {
        "Noise-Safe Min (V)": float(np.min(safe_E)),
        "Noise-Safe Max (V)": float(np.max(safe_E))
    }

# ============================================================
# EXPORT
# ============================================================
def convert_df_to_excel(curves_list):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for cid, df in curves_list:
            if "Vf" in df.columns and "Im" in df.columns:
                clean_df = df[["Vf", "Im"]].dropna()
                if len(clean_df) > 10:
                    clean_df.to_excel(writer, index=False, sheet_name=cid)
    return output.getvalue()

# ============================================================
# FILE UPLOADER
# ============================================================
uploaded_files = st.file_uploader(
    "Upload CV files",
    type=["DTA", "dta", "mpt", "MPT"],
    accept_multiple_files=True
)

colors = px.colors.qualitative.Plotly

# ============================================================
# MAIN LOGIC
# ============================================================
if uploaded_files:

    file_dict = {f.name: f for f in uploaded_files}
    display_names = [f"⋮⋮ {n}" for n in file_dict]

    if 'file_order' not in st.session_state:
        st.session_state.file_order = display_names

    with st.sidebar:
        st.header("🔄 Rearrange Plots")
        st.session_state.file_order = sort_items(st.session_state.file_order)

    actual_order = [n.replace("⋮⋮ ", "") for n in st.session_state.file_order]

    for file_name in actual_order:

        file = file_dict[file_name]
        raw_text = file.getvalue().decode("utf-8", errors="replace")

        # Selección de parser
        if instrument.startswith("Gamry"):
            meta, curves = parse_gamry_dta_multi_curve(raw_text)
        else:
            meta, curves = parse_biologic_mpt(raw_text)

        st.subheader(f"📄 {file.name}")

        excel = convert_df_to_excel(curves)
        st.download_button("📥 Export Excel", excel, f"{file.name}.xlsx")

        fig = go.Figure()

        for i, (cid, df) in enumerate(curves):

            if "Im" not in df.columns:
                continue

            Ecol = "Vf" if "Vf" in df.columns else ("Vu" if "Vu" in df.columns else None)
            if Ecol is None:
                continue

            dd = df[[Ecol, "Im"]].dropna()

            if len(dd) < 10:
                continue

            fig.add_trace(go.Scatter(
                x=dd[Ecol],
                y=dd["Im"],
                mode="lines",
                name=cid,
                line=dict(color=colors[i % len(colors)])
            ))

        fig.update_layout(
            xaxis_title="E (V)",
            yaxis_title="I (A)"
        )

        st.plotly_chart(fig, use_container_width=True)

