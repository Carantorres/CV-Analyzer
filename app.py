import re
import io
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
# GAMRY PARSER
# ============================================================
def parse_gamry_dta_multi_curve(raw: str):
    lines = raw.splitlines()
    curves = []

    current_data = []
    cols = None

    for line in lines:
        if "CURVE" in line.upper():
            if current_data and cols:
                df = pd.DataFrame(current_data, columns=cols)
                curves.append(("Curve", df))
                current_data = []
                cols = None
            continue

        if "Pt" in line and "Im" in line:
            cols = [c.strip() for c in line.split("\t")]
            continue

        if cols:
            parts = line.split("\t")
            if len(parts) == len(cols):
                current_data.append(parts)

    if current_data and cols:
        df = pd.DataFrame(current_data, columns=cols)
        curves.append(("Curve", df))

    clean_curves = []
    for cid, df in curves:
        for c in df.columns:
            col = df[c].astype(str)
            col = col.str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(col, errors="coerce")
        df = df.dropna(how="all").reset_index(drop=True)
        clean_curves.append((cid, df))

    return {}, clean_curves

# ============================================================
# BIOLOGIC PARSER (CORREGIDO)
# ============================================================
def parse_biologic_mpt(raw: str):
    lines = raw.splitlines()
    meta = {}
    header_lines = 0

    # Detect header
    for line in lines:
        if "Nb header lines" in line:
            try:
                header_lines = int(line.split(":")[-1].strip())
            except:
                header_lines = 0
            break

    # Metadata
    for i in range(min(header_lines, len(lines))):
        if ":" in lines[i]:
            k, v = lines[i].split(":", 1)
            meta[k.strip()] = v.strip()

    # Data starts exactly after the header block
    data_lines = [line for line in lines[header_lines:] if line.strip()]

    if not data_lines:
        return meta, []

    # Biologic headers are sometimes split across the last two lines of the header block
    line_minus_1 = [c.strip() for c in lines[header_lines - 1].split('\t')] if header_lines >= 1 else []
    line_minus_2 = [c.strip() for c in lines[header_lines - 2].split('\t')] if header_lines >= 2 else []
    
    line_minus_1 = [c for c in line_minus_1 if c]
    line_minus_2 = [c for c in line_minus_2 if c]

    first_data_row = [c.strip() for c in data_lines[0].split('\t') if c.strip()]
    num_cols = len(first_data_row)

    # Reconstruct headers based on column length matches
    if len(line_minus_1) == num_cols:
        cols = line_minus_1
    elif len(line_minus_2) + len(line_minus_1) == num_cols:
        cols = line_minus_2 + line_minus_1
    else:
        # Fallback to generic columns if the header structure is unexpected
        cols = [f"Col_{i}" for i in range(num_cols)]
        for col_list in [line_minus_2, line_minus_1]:
            for c in col_list:
                cl = c.lower()
                if "ewe" in cl or "potential" in cl:
                    if len(cols) > 2: cols[2] = c
                if "<i>" in cl or "current" in cl:
                    if len(cols) > 3: cols[3] = c

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) >= num_cols:
            rows.append(parts[:num_cols])
        elif len(parts) > 0:
            rows.append(parts + [np.nan] * (num_cols - len(parts)))

    # 🚨 FIX: Force all column names to be strictly unique
    unique_cols = []
    seen = set()
    for c in cols:
        new_c = c
        counter = 1
        while new_c in seen:
            new_c = f"{c}_{counter}"
            counter += 1
        unique_cols.append(new_c)
        seen.add(new_c)

    df = pd.DataFrame(rows, columns=unique_cols)

    # Apply string replacement and float parsing
    for c in df.columns:
        col = df[c].astype(str)
        col = col.str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(col, errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="all").reset_index(drop=True)

    # ========================================================
    # DETECCIÓN INTELIGENTE DE COLUMNAS
    # ========================================================
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "ewe" in cl or "potential" in cl or "voltage" in cl:
            col_map[c] = "Vf"
        if "<i>" in cl or "current" in cl or "i/ma" in cl:
            col_map[c] = "Im"

    df = df.rename(columns=col_map)

    # Validación
    if "Vf" not in df.columns or "Im" not in df.columns:
        return meta, []

    # mA → A
    if df["Im"].abs().max() > 1:
        df["Im"] = df["Im"] / 1000

    return meta, [("Curve 1", df)]

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
# MAIN APP
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

        if not curves:
            st.error("❌ Could not parse this file. Check format.")
            continue

        # Export
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
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        fig.update_layout(
            xaxis_title="E (V vs Ref.)",
            yaxis_title="I (A)"
        )

        st.plotly_chart(fig, use_container_width=True)
