import os
import re
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
st.set_page_config(page_title="Gamry CV Analyzer", layout="wide")
st.title("📊 Gamry Cyclic Voltammetry (CV) Analyzer")
st.markdown("Upload your **Gamry 1010B** `.DTA` files to visualize the potential sweeps and calculate safe operating ranges. *(Currently optimized for GAMRY 1010B data format).*")

# ============================================================
# Utils & Parser Functions
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

def parse_gamry_dta_multi_curve(raw: str) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    lines = raw.splitlines()
    meta: Dict[str, str] = {}
    first_curve_idx = None
    
    for i, line in enumerate(lines):
        if re.match(r"^\s*CURVE\d*\s+TABLE\b", line, flags=re.IGNORECASE) or line.strip().upper().startswith("CURVE"):
            first_curve_idx = i
            break
        if "\t" in line:
            parts = line.split("\t")
            key = parts[0].strip()
            if key:
                val = parts[2].strip() if len(parts) >= 3 else (parts[1].strip() if len(parts) >= 2 else "")
                if val != "":
                    meta[key] = val
        else:
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$", line)
            if m:
                meta[m.group(1).strip()] = m.group(2).strip()

    if first_curve_idx is None:
        return meta, []

    curves: List[Tuple[str, pd.DataFrame]] = []
    i = first_curve_idx

    while i < len(lines):
        line = lines[i]
        m = re.match(r"^\s*CURVE(\d*)\s+TABLE\b(?:\s+(\d+))?", line, flags=re.IGNORECASE)
        if not m:
            i += 1
            continue

        curve_num = m.group(1) if m.group(1) != "" else "1"
        curve_id = f"Curve {curve_num}" 

        j = i + 1
        col_line_idx = None
        while j < len(lines) and j < i + 60:
            s = lines[j].strip()
            if ("\t" in s and "Pt" in s and "Im" in s and ("Vf" in s or "Vu" in s)) or (
                "Pt" in s and "Im" in s and ("Vf" in s or "Vu" in s)
            ):
                col_line_idx = j
                break
            j += 1

        if col_line_idx is None:
            raise ValueError(f"No pude ubicar encabezado de columnas para {curve_id}.")

        raw_cols = [c.strip() for c in lines[col_line_idx].split("\t") if c.strip()]
        if len(raw_cols) < 3:
            raw_cols = [c.strip() for c in re.split(r"\s{2,}", lines[col_line_idx].strip()) if c.strip()]
        cols = raw_cols

        data_start = col_line_idx + 1
        if data_start < len(lines) and lines[data_start].lstrip().startswith("#"):
            data_start += 1

        rows: List[List[str]] = []
        k = data_start
        while k < len(lines):
            s = lines[k].strip()
            if not s:
                k += 1
                continue
            if re.match(r"^\s*CURVE\d*\s+TABLE\b", s, flags=re.IGNORECASE):
                break

            parts = [p.strip() for p in lines[k].split("\t")]
            if len(parts) == 1:
                parts = [p.strip() for p in re.split(r"\s{2,}", s)]
            if parts and parts[0] == "":
                parts = parts[1:]

            if len(parts) >= len(cols):
                rows.append(parts[:len(cols)])
            k += 1

        df = pd.DataFrame(rows, columns=cols)
        for c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all").reset_index(drop=True)
        curves.append((curve_id, df))
        i = k

    return meta, curves

def recommend_operating_ranges_for_curve(df_curve, baseline_E_window=0.20, smooth_window=151, smooth_poly=3, local_window=101, threshold_mode="percentile", nr_fixed=1.30, nr_percentile=95, min_run_points=60, I_tol=0.0):
    Ecol = "Vf" if "Vf" in df_curve.columns else ("Vu" if "Vu" in df_curve.columns else None)
    df = df_curve[[Ecol, "Im"]].copy()
    df.columns = ["E", "I"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    dfE = df.sort_values("E").reset_index(drop=True)
    E = dfE["E"].values
    I = dfE["I"].values
    N = len(dfE)

    if N < 15:
        return {"N_points": N, "noisy_intervals_E": [], "E_cut_cathodic_V": None, "recommended_noise_safe_V": (float(np.min(E)), float(np.max(E))), "recommended_reduction_only_V": None}

    def odd_cap(n):
        n = n if n % 2 == 1 else n + 1
        n = min(n, N if (N % 2 == 1) else N - 1)
        return max(11, n)

    smooth_window = odd_cap(smooth_window)
    local_window = odd_cap(local_window)
    smooth_poly = min(smooth_poly, smooth_window - 2)

    Is = savgol_filter(I, window_length=smooth_window, polyorder=smooth_poly)
    resid = I - Is

    Emax = float(np.max(E))
    base_mask = (E >= (Emax - baseline_E_window)) & (E <= Emax)
    base_resid = resid[base_mask] if base_mask.sum() >= 10 else resid[np.argsort(E)[-max(10, int(0.10 * N)):]]
    
    sigma_base = mad_sigma(base_resid)
    if not np.isfinite(sigma_base) or sigma_base == 0: sigma_base = float(np.std(resid)) if np.std(resid) > 0 else 1e-12

    half = local_window // 2
    NR = np.empty(N, dtype=float)
    for i in range(N):
        lo, hi = max(0, i - half), min(N, i + half + 1)
        sigma_loc = mad_sigma(resid[lo:hi])
        NR[i] = sigma_loc / sigma_base if np.isfinite(sigma_loc) and sigma_base > 0 else np.nan

    NR_finite = NR[np.isfinite(NR)]
    thr = float(nr_fixed) if threshold_mode == "fixed" else float(np.percentile(NR_finite, nr_percentile))
    bad = np.isfinite(NR) & (NR >= thr)

    min_run_eff = min(min_run_points, max(10, N // 6))
    noisy_intervals = []
    i = 0
    while i < N:
        if bad[i]:
            j = i
            while j < N and bad[j]: j += 1
            if (j - i) >= min_run_eff: noisy_intervals.append((float(E[i]), float(E[j - 1])))
            i = j
        else: i += 1

    idx_desc = np.argsort(E)[::-1]
    bad_desc = bad[idx_desc]
    E_desc = E[idx_desc]

    E_cut = None
    k = 0
    while k < N:
        if bad_desc[k]:
            m = k
            while m < N and bad_desc[m]: m += 1
            if (m - k) >= min_run_eff:
                E_cut = float(E_desc[k])
                break
            k = m
        else: k += 1

    noise_safe = (float(np.min(E)), Emax) if E_cut is None else (E_cut, Emax)
    
    df_safe = df[(df["E"] >= noise_safe[0]) & (df["E"] <= noise_safe[1])].dropna()
    red_range = None
    if not df_safe.empty:
        mask_red = df_safe["I"].values <= I_tol
        if np.any(mask_red):
            red_range = (float(np.min(df_safe["E"].values[mask_red])), float(np.max(df_safe["E"].values[mask_red])))

    return {"N_points": N, "noisy_intervals_E": noisy_intervals, "E_cut_cathodic_V": E_cut, "recommended_noise_safe_V": noise_safe, "recommended_reduction_only_V": red_range}


# ============================================================
# APP LOGIC
# ============================================================

# File Uploader
uploaded_files = st.file_uploader("Upload .DTA files", type=["DTA", "dta"], accept_multiple_files=True)

# Define solid colors for our curves
default_colors = px.colors.qualitative.Plotly

if uploaded_files:
    # Map uploaded files by name
    file_dict = {f.name: f for f in uploaded_files}
    
    # Create display names with the grab icon "⋮⋮"
    display_names = [f"⋮⋮ {name}" for name in file_dict.keys()]
    
    if 'file_order' not in st.session_state:
        st.session_state.file_order = display_names
    else:
        # Sync list if files are added or removed
        current_files = set(display_names)
        for name in display_names:
            if name not in st.session_state.file_order:
                st.session_state.file_order.append(name)
        st.session_state.file_order = [name for name in st.session_state.file_order if name in current_files]

    # --- DRAG AND DROP RENDERED EXPLICITLY INSIDE THE SIDEBAR ---
    with st.sidebar:
        st.header("🔄 Rearrange Plots")
        st.markdown("Drag and drop to reorder:")
        
        # This will now perfectly render inside the left menu!
        sorted_display_names = sort_items(st.session_state.file_order)
        st.session_state.file_order = sorted_display_names
        
        st.markdown("---")
        
        # --- AUTHOR CREDIT ---
        st.markdown(
            """
            <div style='text-align: center; margin-top: 50px;'>
                <p style='color: #888888; font-size: 0.85rem; font-family: sans-serif;'>
                    Developed by<br>
                    <b>PhD(c) Carlos A. Torres-Ramírez</b><br><br>
                    <i>Optimized for GAMRY 1010B data format</i>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- PLOTTING LOGIC ---
    # Remove the "⋮⋮ " to get the real file name back
    actual_file_order = [name.replace("⋮⋮ ", "") for name in sorted_display_names]

    for file_name in actual_file_order:
        # Prevent errors if file state glitches during drag
        if file_name not in file_dict:
            continue 
            
        file = file_dict[file_name]
        
        st.markdown("---")
        st.subheader(f"📄 Data for: {file.name}")
        
        # Read the file text
        raw_text = file.getvalue().decode("utf-8", errors="replace")
        meta, curves = parse_gamry_dta_multi_curve(raw_text)
        
        # --- EXTRACT EXPERIMENT LIMITS ---
        vinit = _to_float(meta.get("VINIT"))
        vlim1 = _to_float(meta.get("VLIMIT1"))
        vlim2 = _to_float(meta.get("VLIMIT2"))
        sr = _to_float(meta.get("SCANRATE"))
        
        # Display the limits nicely
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial Potential", f"{vinit} V" if vinit is not None else "N/A")
        c2.metric("Scan Limit 1", f"{vlim1} V" if vlim1 is not None else "N/A")
        c3.metric("Scan Limit 2", f"{vlim2} V" if vlim2 is not None else "N/A")
        c4.metric("Scan Rate", f"{sr} mV/s" if sr is not None else "N/A")
        # ---------------------------------

        fig = go.Figure()
        results_list = []
        
        for i, (cid, dfi) in enumerate(curves):
            Ecol = "Vf" if "Vf" in dfi.columns else ("Vu" if "Vu" in dfi.columns else None)
            if Ecol is None or "Im" not in dfi.columns:
                continue
                
            dd = dfi[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna()
            
            # Skip artifact curves
            if len(dd) < 10:
                continue
            
            line_color = default_colors[i % len(default_colors)]
            
            # Plot the continuous smooth line (No markers/stars!)
            fig.add_trace(go.Scatter(
                x=dd[Ecol], 
                y=dd["Im"], 
                mode='lines',
                name=cid, 
                line=dict(color=line_color, width=2)
            ))
            
            # Mathematical Range Analysis
            out = recommend_operating_ranges_for_curve(dfi)
            ns = out["recommended_noise_safe_V"]
            ro = out["recommended_reduction_only_V"]
            
            results_list.append({
                "Curve": cid,
                "Points": out["N_points"],
                "Noise-Safe Min (V)": round(ns[0], 4) if ns else None,
                "Noise-Safe Max (V)": round(ns[1], 4) if ns else None,
                "Reduction Min (V)": round(ro[0], 4) if ro else None,
                "Reduction Max (V)": round(ro[1], 4) if ro else None,
            })

        # Layout styling
        fig.update_layout(
            xaxis_title="E (V vs Ref.)",
            yaxis_title="Im (A)",
            legend=dict(
                font=dict(color="white"), 
                yanchor="top", 
                y=0.99, 
                xanchor="left", 
                x=0.01, 
                bgcolor="rgba(0,0,0,0)"
            )
        )
        
        # Display the interactive plot on the web page
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the data table
        if results_list:
            st.write("**Recommended Operating Ranges:**")
            st.dataframe(pd.DataFrame(results_list), use_container_width=True)