import os
import re
import io
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter
from scipy.stats import linregress
import streamlit as st
import streamlit.components.v1 as components
from streamlit_sortables import sort_items

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="CV Analyzer", layout="wide")
st.title("📊 Universal CV & LSV Analyzer")

st.markdown("""
**A comprehensive tool for automated electrochemical data analysis.** 
Seamlessly process CV and LSV files from **Gamry (.DTA)**, **Biologic (.mpt)**, and **PalmSens PSTrace (.csv)** potentiostats. 
This platform automates iR drop compensation, RHE scale conversion, and robust catalytic parameter extraction. Additionally, it features an operating range algorithm that detects noise-free potential windows—avoiding hydrogen evolution and intense oxidation currents—ideal for applications like metal electrodeposition.
""")

with st.popover("📖 View Calculation Methods & Algorithms"):
    st.markdown("""
    **1. Physico-Chemical Corrections**
    *   **iR Drop Compensation:** Corrects for the uncompensated resistance ($R_u$) of the electrolyte using Ohm's Law: 
        $E_{corr} = E_{raw} - (I \\times R_u \\times \\frac{\\text{Comp \\%}}{100})$
    *   **RHE Scale Conversion:** Shifts the potential to a pH-independent thermodynamic scale using the Nernst equation (at 25°C):
        $E_{RHE} = E_{corr} + E^0_{Ref} + (0.0591 \\times \\text{pH})$

    **2. Catalytic Parameter Extraction (LSV)**
    *   **Onset Potential ($E_{onset}$):** Empirically defined as the exact potential where the faradaic current overcomes the capacitive basal noise, set at the point where $|I|$ reaches **5% of the absolute maximum current**.
    *   **Overpotentials ($\\eta_{10}, \\eta_{50}, \\eta_{100}$):** Calculated via univariate linear interpolation over the sorted current density vector.
    *   **Robust Tafel Slope:** Computed using a dynamic **Sliding Window algorithm**. The curve is cropped to the pure kinetic region (2% to 40% of $I_{max}$). A moving window calculates the OLS linear regression of $E$ vs $\\log_{10}|j|$ at every step. The algorithm automatically selects and reports the region with the highest coefficient of determination ($R^2$).

    **3. Averaging & Statistical Analysis**
    *   **Cycle Averaging:** To average cycles with varying data point lengths, individual scans are mapped and interpolated over a normalized coordinate system ($0 \\rightarrow 1$). The arithmetic mean and standard deviation are calculated point-by-point to trace the main solid line and the SD shaded area.

    **4. Operating Range Detection**
    *   **Noise-Free Electrodeposition Window:** Uses a **Savitzky-Golay filter** coupled with **Median Absolute Deviation (MAD)** to estimate local signal-to-noise ratios. It automatically discards segments with intense faradaic noise (e.g., massive hydrogen bubbling) to recommend a stable and reliable potential window.
    
    **5. Scan Rate Kinetics ($b$-value Analysis)**
    *   Based on the power-law relationship $i_p = a \\cdot v^b$, the anodic peak current density ($j_{pa}$) is analyzed as a function of the scan rate ($v$).
    *   The algorithm extracts the absolute maximum anodic current $j_{pa}$ and its potential $E_{pa}$ for each trace.
    *   A linear regression of $\\log_{10}(j_{pa})$ vs $\\log_{10}(v)$ calculates the slope $b$ and its standard error. A value of $b=0.5$ implies a diffusion-controlled process, while $b=1.0$ indicates a surface-confined (capacitive) process.
    """)

st.markdown("---")

# ============================================================
# INSTRUMENT SELECTION
# ============================================================
instrument = st.selectbox(
    "Select instrument format:",
    ["Gamry 1010B (.DTA)", "Biologic SP-50e (.mpt)", "PalmSens PSTrace (.csv)"]
)

# ============================================================
# UTILITIES & MATH
# ============================================================
def to_rgba(color_str: str, alpha: float = 0.2) -> str:
    color_str = color_str.strip().lower()
    if color_str.startswith('#'):
        h = color_str.lstrip('#')
        if len(h) == 6:
            return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"
    elif color_str.startswith('rgb('):
        return color_str.replace('rgb(', 'rgba(').replace(')', f', {alpha})')
    return f"rgba(150, 150, 150, {alpha})"

def get_sr_from_name(name: str, default_sr: float) -> float:
    m = re.search(r'(\d+\.?\d*)\s*mV/s', name, re.IGNORECASE)
    if m: return float(m.group(1))
    return default_sr if default_sr is not None else 0.0

def get_averaged_curve(processed_curves: List[Tuple[str, pd.DataFrame, str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not processed_curves:
        return None, None, None
    max_points = max(len(dd) for _, dd, _ in processed_curves)
    common_idx = np.linspace(0, 1, max_points)
    
    E_interp = []
    I_interp = []
    for _, dd, Ecol in processed_curves:
        idx = np.linspace(0, 1, len(dd))
        E_interp.append(np.interp(common_idx, idx, dd[Ecol].values))
        I_interp.append(np.interp(common_idx, idx, dd["Im"].values))
        
    return np.mean(E_interp, axis=0), np.mean(I_interp, axis=0), np.std(I_interp, axis=0)

def mad_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 10: return float(np.std(x)) if len(x) else np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else float(np.std(x))

def _to_float(x):
    if x is None: return None
    try: return float(str(x).replace(",", "."))
    except: return None

def extract_limits_from_data(df: pd.DataFrame, technique: str) -> Tuple[float, float, float]:
    if len(df) == 0: return None, None, None
    Ecol = "Vf" if "Vf" in df.columns else ("Vu" if "Vu" in df.columns else df.columns[0])
    v_data = df[Ecol].dropna().values
    if len(v_data) == 0: return None, None, None
        
    vinit = round(float(v_data[0]), 3)
    if "LSV" in technique:
        vlim1 = round(float(v_data[-1]), 3)
        vlim2 = None
    else:
        dv = np.diff(v_data)
        dv_non_zero = dv[dv != 0]
        if len(dv_non_zero) == 0: return vinit, vinit, vinit
        signs = np.sign(dv_non_zero)
        sign_changes = np.where(signs[:-1] != signs[1:])[0]
        non_zero_indices = np.where(dv != 0)[0]
        turn_indices = non_zero_indices[sign_changes] + 1
        
        if len(turn_indices) >= 1:
            vlim1 = round(float(v_data[turn_indices[0]]), 3)
            vlim2 = round(float(v_data[turn_indices[1]]), 3) if len(turn_indices) >= 2 else round(float(v_data[-1]), 3)
        else:
            idx_max_dist = np.argmax(np.abs(v_data - v_data[0]))
            vlim1 = round(float(v_data[idx_max_dist]), 3)
            vlim2 = round(float(v_data[-1]), 3)
            
    return vinit, vlim1, vlim2

def recommend_operating_ranges_for_curve(df_curve, baseline_E_window=0.20, smooth_window=151, smooth_poly=3, local_window=101, threshold_mode="percentile", nr_fixed=1.30, nr_percentile=95, min_run_points=60, I_tol=0.0):
    Ecol = "Vf" if "Vf" in df_curve.columns else ("Vu" if "Vu" in df_curve.columns else None)
    df = df_curve[[Ecol, "Im"]].copy()
    df.columns = ["E", "I"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    dfE = df.sort_values("E").reset_index(drop=True)
    E, I = dfE["E"].values, dfE["I"].values
    N = len(dfE)

    if N < 15: return {"N_points": N, "noisy_intervals_E": [], "E_cut_cathodic_V": None, "recommended_noise_safe_V": (float(np.min(E)), float(np.max(E))), "recommended_reduction_only_V": None}

    def odd_cap(n):
        n = n if n % 2 == 1 else n + 1
        return max(11, min(n, N if (N % 2 == 1) else N - 1))

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

    min_run_eff, noisy_intervals, i = min(min_run_points, max(10, N // 6)), [], 0
    while i < N:
        if bad[i]:
            j = i
            while j < N and bad[j]: j += 1
            if (j - i) >= min_run_eff: noisy_intervals.append((float(E[i]), float(E[j - 1])))
            i = j
        else: i += 1

    idx_desc = np.argsort(E)[::-1]
    bad_desc, E_desc = bad[idx_desc], E[idx_desc]
    E_cut, k = None, 0
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
        if np.any(mask_red): red_range = (float(np.min(df_safe["E"].values[mask_red])), float(np.max(df_safe["E"].values[mask_red])))

    return {"N_points": N, "noisy_intervals_E": noisy_intervals, "E_cut_cathodic_V": E_cut, "recommended_noise_safe_V": noise_safe, "recommended_reduction_only_V": red_range}

def extract_lsv_catalytic_parameters(df_curve: pd.DataFrame, area_cm2: float, e_rev: float) -> Tuple[dict, dict]:
    Ecol = "Vf" if "Vf" in df_curve.columns else ("Vu" if "Vu" in df_curve.columns else None)
    if Ecol is None or "Im" not in df_curve.columns: return {}, {}
        
    df_sorted = df_curve.sort_values(by=Ecol).reset_index(drop=True)
    E, I = df_sorted[Ecol].values, df_sorted["Im"].values
    abs_I = np.abs(I)
    
    if len(abs_I) < 20: return {}, {}
        
    j_dens = (abs_I * 1000) / area_cm2
    eta_mV = np.abs(E - e_rev) * 1000
    
    I_max, j_max, eta_max = np.max(abs_I), np.max(j_dens), eta_mV[np.argmax(j_dens)]
    
    onset_mask = abs_I >= 0.05 * I_max
    E_onset = E[onset_mask][0] if np.any(onset_mask) else np.nan
    
    search_mask = (abs_I >= 0.02 * I_max) & (abs_I <= 0.40 * I_max)
    E_search, I_search = E[search_mask], abs_I[search_mask]
    
    best_r2, best_slope, best_intercept, best_log_I_fit = -1, np.nan, np.nan, []
    
    if len(E_search) > 10:
        log_I_search = np.log10(I_search)
        win_size = max(10, len(E_search) // 5) 
        for i in range(len(E_search) - win_size):
            x_win, y_win = log_I_search[i:i+win_size], E_search[i:i+win_size]
            slope, intercept, r_value, _, _ = linregress(x_win, y_win)
            r2 = r_value**2
            if r2 > best_r2 and not np.isnan(r2):
                best_r2, best_slope, best_intercept, best_log_I_fit = r2, slope, intercept, x_win

    tafel_slope = abs(best_slope * 1000) if not np.isnan(best_slope) else np.nan
            
    sort_idx = np.argsort(j_dens)
    j_sorted, eta_sorted = j_dens[sort_idx], eta_mV[sort_idx]
    
    etas = {}
    for target in [10, 20, 50, 100]:
        etas[f"η_{target} (mV)"] = np.interp(target, j_sorted, eta_sorted) if target <= j_max else np.nan
    if j_max < 100: etas[f"η_max@{j_max:.1f} (mV)"] = eta_max

    params = {
        "j_max (mA/cm²)": j_max, "E_onset (V)": E_onset,
        "Tafel Slope (mV/dec)": tafel_slope, "Tafel R²": best_r2 if best_r2 != -1 else np.nan,
        **etas
    }
    
    try:
        win_len = min(31, len(abs_I) - 1 if len(abs_I) % 2 == 0 else len(abs_I))
        win_len = win_len if win_len % 2 == 1 else win_len - 1
        I_smooth = savgol_filter(abs_I, window_length=max(5, win_len), polyorder=2)
    except:
        I_smooth = abs_I
        
    I_smooth = np.where(I_smooth <= 0, 1e-12, I_smooth) 
    
    fit_data = {
        "E_full": E, "log_I_full": np.log10(I_smooth),
        "log_I_fit": best_log_I_fit, "slope": best_slope, "intercept": best_intercept,
        "log_I_max": np.log10(I_max) if I_max > 0 else 0,
        "j_dens": j_dens, "eta_mV": eta_mV
    }
    return params, fit_data

def apply_scientific_style(fig, is_scientific, lx, ly, lxa, lya):
    """Aplica el estilo editorial y posiciona la leyenda de forma personalizada."""
    if is_scientific:
        fig.update_layout(
            title="", 
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=16, color="black"),
            margin=dict(l=80, r=40, t=40, b=60)
        )
        fig.update_xaxes(
            showgrid=False, showline=True, linecolor='black', linewidth=2,
            mirror="all", ticks='inside', tickcolor='black', tickwidth=2, ticklen=8,
            title_font=dict(size=18, family="Arial, sans-serif", color="black"),
            tickfont=dict(size=15, family="Arial, sans-serif", color="black"), zeroline=False
        )
        fig.update_yaxes(
            showgrid=False, showline=True, linecolor='black', linewidth=2,
            mirror="all", ticks='inside', tickcolor='black', tickwidth=2, ticklen=8,
            title_font=dict(size=18, family="Arial, sans-serif", color="black"),
            tickfont=dict(size=15, family="Arial, sans-serif", color="black"), zeroline=False
        )
        
    fig.update_layout(
        legend=dict(
            x=lx, y=ly, xanchor=lxa, yanchor=lya,
            bgcolor='rgba(255, 255, 255, 0.9)', 
            bordercolor='black', 
            borderwidth=1 if is_scientific else 0,
            font=dict(size=14, color="black" if is_scientific else None)
        )
    )
    return fig

# ============================================================
# PARSERS
# ============================================================
def parse_gamry_dta_multi_curve(raw: str) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    lines = raw.splitlines()
    meta: Dict[str, str] = {}
    first_curve_idx = None
    
    for i, line in enumerate(lines):
        if re.match(r"^\s*CURVE\d*\s+TABLE\b", line, flags=re.IGNORECASE) or line.strip().upper().startswith("CURVE"):
            first_curve_idx = i; break
        if "\t" in line:
            parts = line.split("\t")
            if parts[0].strip():
                val = parts[2].strip() if len(parts) >= 3 else (parts[1].strip() if len(parts) >= 2 else "")
                if val: meta[parts[0].strip()] = val
        else:
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$", line)
            if m: meta[m.group(1).strip()] = m.group(2).strip()

    if first_curve_idx is None: return meta, []

    curves: List[Tuple[str, pd.DataFrame]] = []
    i = first_curve_idx
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^\s*CURVE(\d*)\s+TABLE\b(?:\s+(\d+))?", line, flags=re.IGNORECASE)
        if not m:
            i += 1; continue
        curve_id = f"Curve {m.group(1) if m.group(1) else '1'}" 

        j, col_line_idx = i + 1, None
        while j < len(lines) and j < i + 60:
            s = lines[j].strip()
            if ("Pt" in s and "Im" in s and ("Vf" in s or "Vu" in s)):
                col_line_idx = j; break
            j += 1

        if col_line_idx is None: raise ValueError(f"No pude ubicar encabezado de columnas para {curve_id}.")
        cols = [c.strip() for c in lines[col_line_idx].split("\t") if c.strip()]
        if len(cols) < 3: cols = [c.strip() for c in re.split(r"\s{2,}", lines[col_line_idx].strip()) if c.strip()]

        data_start = col_line_idx + 1
        if data_start < len(lines) and lines[data_start].lstrip().startswith("#"): data_start += 1

        rows: List[List[str]] = []
        k = data_start
        while k < len(lines):
            s = lines[k].strip()
            if not s:
                k += 1; continue
            if re.match(r"^\s*CURVE\d*\s+TABLE\b", s, flags=re.IGNORECASE): break
            parts = [p.strip() for p in lines[k].split("\t")]
            if len(parts) == 1: parts = [p.strip() for p in re.split(r"\s{2,}", s)]
            if parts and parts[0] == "": parts = parts[1:]
            if len(parts) >= len(cols): rows.append(parts[:len(cols)])
            k += 1

        df = pd.DataFrame(rows, columns=cols)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all").reset_index(drop=True)
        curves.append((curve_id, df))
        i = k
    return meta, curves

def parse_biologic_mpt(raw: str):
    lines = raw.splitlines()
    meta, header_lines = {}, 0
    for line in lines:
        if "Nb header lines" in line:
            try: header_lines = int(line.split(":")[-1].strip())
            except: header_lines = 0
            break

    for i in range(min(header_lines, len(lines))):
        line = lines[i].strip()
        if not line: continue
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
        else:
            parts = re.split(r'\s{2,}|\t+', line)
            if len(parts) >= 2: meta[parts[0].strip()] = parts[1].strip()

    data_lines = [line for line in lines[header_lines:] if line.strip()]
    if not data_lines: return meta, []

    line_minus_1 = [c for c in [c.strip() for c in lines[header_lines - 1].split('\t')] if c] if header_lines >= 1 else []
    line_minus_2 = [c for c in [c.strip() for c in lines[header_lines - 2].split('\t')] if c] if header_lines >= 2 else []

    num_cols = len([c.strip() for c in data_lines[0].split('\t') if c.strip()])
    if len(line_minus_1) == num_cols: cols = line_minus_1
    elif len(line_minus_2) + len(line_minus_1) == num_cols: cols = line_minus_2 + line_minus_1
    else:
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
        if len(parts) >= num_cols: rows.append(parts[:num_cols])
        elif len(parts) > 0: rows.append(parts + [np.nan] * (num_cols - len(parts)))

    unique_cols, seen = [], set()
    for c in cols:
        new_c, counter = c, 1
        while new_c in seen:
            new_c = f"{c}_{counter}"
            counter += 1
        unique_cols.append(new_c)
        seen.add(new_c)

    df = pd.DataFrame(rows, columns=unique_cols)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all").reset_index(drop=True)

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "ewe" in cl or "potential" in cl or "voltage" in cl: col_map[c] = "Vf"
        elif "<i>" in cl or "current" in cl or "i/ma" in cl: col_map[c] = "Im"
        elif cl == "cycle number" or cl == "cycle": col_map[c] = "Cycle"
    df = df.rename(columns=col_map)
    if "Vf" not in df.columns or "Im" not in df.columns: return meta, []
    if df["Im"].abs().max() > 1: df["Im"] = df["Im"] / 1000

    curves = []
    if "Cycle" in df.columns:
        for cyc in sorted(df["Cycle"].dropna().unique()):
            df_cyc = df[df["Cycle"] == cyc].copy()
            if len(df_cyc) > 0: curves.append((f"Cycle {int(cyc) if float(cyc).is_integer() else cyc}", df_cyc.reset_index(drop=True)))
    else: curves.append(("Curve 1", df))
    return meta, curves

def parse_pstrace_csv(raw: bytes) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    text = None
    for enc in ['utf-8', 'utf-16', 'latin1']:
        try:
            text = raw.decode(enc)
            break
        except: continue
            
    if not text: return {}, []
        
    lines, meta, curves, scan_names, unit_row_idx = text.splitlines(), {}, [], [], -1
    
    for line in lines[:20]:
        if "Linear Sweep" in line or "LSV" in line: meta["TECHNIQUE"] = "Linear Sweep Voltammetry (LSV)"
        elif "Cyclic Voltammetry" in line or "CV" in line: meta["TECHNIQUE"] = "Cyclic Voltammetry (CV)"
    
    for i, line in enumerate(lines[:50]):
        if ("Scan" in line or "Curve" in line or "vs E" in line) and "Date" not in line and "Voltammetry" in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if (len(parts) > 1 or (len(parts)==1 and "Scan" in parts[0])) and not scan_names: scan_names = parts
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2 and any(v in parts[0] for v in ['V', 'mV', 'E']) and any('A' in u for u in parts):
            unit_row_idx = i; break
            
    if unit_row_idx == -1: return meta, curves
        
    unit_parts = [p.strip() for p in lines[unit_row_idx].split(",")]
    rows = [[p.strip() for p in line.split(",")] for line in lines[unit_row_idx+1:] if line.strip()]
    if not rows: return meta, curves
        
    df_raw = pd.DataFrame(rows).replace("", np.nan)
    num_scans = len(unit_parts) // 2
    if not scan_names: scan_names = [f"Scan {i+1}" for i in range(num_scans)]
        
    for i in range(num_scans):
        col_v, col_i = i * 2, i * 2 + 1
        if col_i >= df_raw.shape[1]: break
            
        df_scan = df_raw.iloc[:, [col_v, col_i]].copy()
        df_scan.columns = ["Vf", "Im"]
        df_scan["Vf"] = pd.to_numeric(df_scan["Vf"].astype(str).str.replace(",", "."), errors="coerce")
        df_scan["Im"] = pd.to_numeric(df_scan["Im"].astype(str).str.replace(",", "."), errors="coerce")
        df_scan = df_scan.dropna()
        if len(df_scan) == 0: continue
            
        if "mV" in unit_parts[col_v]: df_scan["Vf"] = df_scan["Vf"] / 1000.0
            
        i_unit = unit_parts[col_i]
        if "mA" in i_unit: df_scan["Im"] = df_scan["Im"] * 1e-3
        elif "µA" in i_unit or "uA" in i_unit: df_scan["Im"] = df_scan["Im"] * 1e-6
        elif "nA" in i_unit: df_scan["Im"] = df_scan["Im"] * 1e-9
            
        name = scan_names[i] if i < len(scan_names) else f"Scan {i+1}"
        if "TECHNIQUE" not in meta:
            meta["TECHNIQUE"] = "Linear Sweep Voltammetry (LSV)" if "Linear Sweep" in name or "LSV" in name else "Cyclic Voltammetry (CV)"
        curves.append((name, df_scan))
        
    return meta, curves

def convert_df_to_excel(curves_list: List[Tuple[str, pd.DataFrame]]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        seen_names = set()
        for cid, df in curves_list:
            Ecol = "Vf" if "Vf" in df.columns else ("Vu" if "Vu" in df.columns else None)
            if Ecol is None or "Im" not in df.columns: continue
            clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[Ecol, "Im"])
            if len(clean_df) >= 10:
                safe_name = re.sub(r'[\\*?:/\[\]]', '_', cid)[:31].strip() or "Sheet"
                original_safe_name, counter = safe_name, 1
                while safe_name in seen_names:
                    suffix = f"_{counter}"
                    safe_name = f"{original_safe_name[:31-len(suffix)]}{suffix}"
                    counter += 1
                seen_names.add(safe_name)
                clean_df.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()

# ============================================================
# APP LOGIC
# ============================================================
uploaded_files = st.file_uploader("Upload CV/LSV files", type=["DTA", "dta", "mpt", "MPT", "csv", "CSV"], accept_multiple_files=True)

publication_palette = [
    '#000000', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628', '#F781BF', '#17BECF', '#BDB76B', '#000080'
] + px.colors.qualitative.Alphabet
combined_palette = publication_palette

dl_config = {'toImageButtonOptions': {'format': 'png', 'filename': 'electrochem_plot', 'height': 720, 'width': 960, 'scale': 4}}

with st.sidebar:
    st.header("⚡ iR Drop Compensation")
    apply_ir = st.toggle("Apply iR Compensation", value=False)
    ru_ohms = st.number_input("Uncompensated Resistance (Ru) [Ohms]", value=10.0, step=1.0) if apply_ir else 0.0
    comp_percent = st.slider("Compensation Percentage (%)", 0, 100, 85, 1) if apply_ir else 0.0

    st.markdown("---")
    st.header("⚖️ Reference Electrode & RHE")
    ref_elec = st.selectbox("Reference Electrode", ["Ag/AgCl (sat. KCl)", "SCE (sat. KCl)", "Hg/HgO (1M KOH)", "Custom"])
    
    if ref_elec == "Custom":
        custom_ref_name = st.text_input("Custom Reference Label", value="Ref.")
    else:
        custom_ref_name = ref_elec.split(" (")[0]
        
    convert_to_rhe = st.toggle("Convert E to RHE scale", value=True)
    
    if convert_to_rhe:
        if ref_elec == "Custom":
            e0_ref = st.number_input("Custom E0_Ref (V)", value=0.000, step=0.01)
        else:
            e0_ref = 0.197 if ref_elec.startswith("Ag") else (0.241 if ref_elec.startswith("SCE") else 0.098)
            st.info(f"Using Standard E₀ = {e0_ref} V")
        ph_val = st.number_input("pH of the solution", value=14.0, step=0.1)
        x_axis_label = "E (V vs RHE)" + (" [iR corrected]" if apply_ir else "")
    else:
        e0_ref, ph_val = 0.0, 0.0
        x_axis_label = f"E (V vs {custom_ref_name})" + (" [iR corrected]" if apply_ir else "")

    st.markdown("---")
    st.header("⚙️ Catalytic Parameters")
    electrode_area = st.number_input("Electrode Area (cm²)", min_value=0.00001, value=1.00000, step=0.001, format="%.5f")
    manual_scan_rate = st.number_input("Manual Scan Rate (mV/s) [Optional]", value=0.0, step=10.0)
    if convert_to_rhe: st.info("💡 **Tip:** E_rev is generally **0.0 V** for HER and **1.23 V** for OER.")
    e_rev = st.number_input("Thermodynamic Potential (E_rev)", value=0.000, step=0.01)

    st.markdown("---")
    st.header("🎨 Plot Formatting")
    st.markdown("Customize plots for publication.")
    scientific_style = st.toggle("Scientific Paper Style (ACS/Elsevier)", value=True)
    show_sd_shadow = st.toggle("Show SD Shadow on Averages", value=True)
    
    st.markdown("📦 **Legend Position Tool**")
    leg_pos = st.selectbox("Quick Positions", ["Top-Right", "Top-Left", "Bottom-Right", "Bottom-Left", "Outside Right", "Custom..."])
    
    if leg_pos == "Top-Right":
        lx, ly, lxa, lya = 0.99, 0.99, "right", "top"
    elif leg_pos == "Top-Left":
        lx, ly, lxa, lya = 0.01, 0.99, "left", "top"
    elif leg_pos == "Bottom-Right":
        lx, ly, lxa, lya = 0.99, 0.01, "right", "bottom"
    elif leg_pos == "Bottom-Left":
        lx, ly, lxa, lya = 0.01, 0.01, "left", "bottom"
    elif leg_pos == "Outside Right":
        lx, ly, lxa, lya = 1.02, 1.0, "left", "top"
    else:
        lx = st.slider("X Coordinate", min_value=-0.2, max_value=1.5, value=0.99, step=0.01)
        ly = st.slider("Y Coordinate", min_value=-0.2, max_value=1.5, value=0.99, step=0.01)
        lxa, lya = "auto", "auto"
    
    st.markdown("---")
    st.header("📄 Export Full Report")
    components.html("""<button onclick="window.parent.print();" style="background-color:#FF4B4B; color:white; border:none; border-radius:4px; padding:0.5rem 1rem; font-size:1rem; font-weight:600; cursor:pointer; width:100%;">🖨️ Save Page as PDF</button>""", height=50)
    
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <p style='color: #888888; font-size: 0.85rem; font-family: sans-serif;'>
                Developed by<br>
                <b>PhD(c) Carlos A. Torres-Ramírez</b><br><br>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

i_axis_label = "Current, I (A)" if scientific_style else "I (A)"
j_axis_label = "Current Density, j (mA cm⁻²)" if scientific_style else "Current Density j (mA/cm²)"

if uploaded_files:
    file_dict = {f.name: f for f in uploaded_files}
    display_names = set([f"⋮⋮ {name}" for name in file_dict.keys()])
    
    if 'file_groups' not in st.session_state: st.session_state.file_groups = [{"header": "📥 Unassigned Files", "items": []}, {"header": "📊 Group 1", "items": []}]
    for group in st.session_state.file_groups: group["items"] = [item for item in group["items"] if item in display_names]
        
    existing_items = set([item for group in st.session_state.file_groups for item in group["items"]])
    new_items = display_names - existing_items
    if new_items: st.session_state.file_groups[0]["items"].extend(list(new_items))

    with st.sidebar:
        st.header("🗂️ Drag & Drop Groups")
        c1, c2 = st.columns(2)
        if c1.button("➕ Add Group"): st.session_state.file_groups.append({"header": f"📊 Group {len(st.session_state.file_groups)}", "items": []}); st.rerun()
        if c2.button("➖ Remove Group") and len(st.session_state.file_groups) > 1:
            st.session_state.file_groups[0]["items"].extend(st.session_state.file_groups[-1]["items"])
            st.session_state.file_groups.pop(); st.rerun()
            
        unassigned_count = len(st.session_state.file_groups[0]["items"])
        if unassigned_count > 0 and len(st.session_state.file_groups) > 1:
            st.markdown(f"**🚀 Bulk Move ({unassigned_count} files)**")
            bc1, bc2 = st.columns([2, 1])
            with bc1:
                target_g = st.selectbox("Target", [g["header"] for g in st.session_state.file_groups[1:]], label_visibility="collapsed")
            with bc2:
                if st.button("Move All"):
                    for g in st.session_state.file_groups:
                        if g["header"] == target_g:
                            g["items"].extend(st.session_state.file_groups[0]["items"])
                            st.session_state.file_groups[0]["items"] = []
                            st.rerun()
                            
        st.session_state.file_groups = sort_items(st.session_state.file_groups, multi_containers=True)

    # --- DICTIONARY TO STORE PROCESSED DATA FOR SUPER GROUPS ---
    prepared_group_data = {}

    # --- TOP AREA: GROUPED COMPARISONS ---
    has_groups_plotted = False
    valid_groups_for_super = []
    
    for g_idx, group in enumerate(st.session_state.file_groups):
        if g_idx == 0 or not group["items"]: continue 
        
        valid_groups_for_super.append(group["header"])
            
        if not has_groups_plotted:
            st.header("📈 Group Comparisons")
            has_groups_plotted = True
            
        st.subheader(f"{group['header']}")
        
        group_data_parsed = []
        is_group_lsv = False
        
        for item in group["items"]:
            fname = item.replace("⋮⋮ ", "")
            if fname not in file_dict: continue
            
            raw_text = file_dict[fname].getvalue().decode("utf-8", errors="replace")
            tech_sg, sr = "Unknown", None
            
            if instrument.startswith("Gamry"):
                meta_sg, curves_comp = parse_gamry_dta_multi_curve(raw_text)
                tech_sg = "LSV" if "LSV" in meta_sg.get("TAG", "").upper() or "LINEAR" in meta_sg.get("TITLE", "").upper() else "CV"
                sr = _to_float(meta_sg.get("SCANRATE"))
            elif instrument.startswith("PalmSens"):
                meta_sg, curves_comp = parse_pstrace_csv(file_dict[fname].getvalue())
                tech_sg = "LSV" if "LSV" in meta_sg.get("TECHNIQUE", "") else "CV"
            else:
                meta_sg, curves_comp = parse_biologic_mpt(raw_text)
                tech_sg = "LSV" if "E2 (V)" not in meta_sg else "CV"
                sr = _to_float(meta_sg.get("dE/dt"))
                
            if manual_scan_rate > 0.0: sr = manual_scan_rate
            if "LSV" in tech_sg: is_group_lsv = True
                
            processed_curves = []
            for cid, df_comp in curves_comp:
                Ecol = "Vf" if "Vf" in df_comp.columns else ("Vu" if "Vu" in df_comp.columns else None)
                if Ecol and "Im" in df_comp.columns:
                    dd_comp = df_comp[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                    if apply_ir: dd_comp[Ecol] = dd_comp[Ecol] - dd_comp["Im"] * ru_ohms * (comp_percent / 100.0)
                    if convert_to_rhe: dd_comp[Ecol] = dd_comp[Ecol] + e0_ref + (0.0591 * ph_val)
                    if len(dd_comp) >= 10: processed_curves.append((cid, dd_comp, Ecol))
                        
            if processed_curves:
                group_data_parsed.append({"fname": fname, "tech": tech_sg, "sr": sr, "curves": processed_curves})

        custom_labels = {}
        if len(group_data_parsed) > 0:
            avg_mode = st.radio(
                "Averaging Mode:", 
                ["None (Plot all)", "Average Scans per File", "Average ALL Files into One Curve"], 
                key=f"mode_{g_idx}", horizontal=True
            )
            
            if avg_mode == "Average Scans per File":
                st.markdown("**Customize Legend Labels:**")
                cols = st.columns(3)
                for i, dat in enumerate(group_data_parsed):
                    def_val = f"{dat['sr']} mV/s" if dat['sr'] else dat['fname']
                    custom_labels[dat['fname']] = cols[i%3].text_input(f"Label for {dat['fname']}", value=def_val, key=f"lbl_g_{g_idx}_{dat['fname']}")
            elif avg_mode == "Average ALL Files into One Curve":
                custom_labels["ALL"] = st.text_input("Legend Label for Averaged Group:", value=group['header'].replace("📊 ", ""), key=f"lbl_all_{g_idx}")

        fig_comp, fig_tafel_comp, fig_jeta_comp = go.Figure(), go.Figure(), go.Figure()
        trace_idx, group_lsv_params, max_log_I_global = 0, [], -10 
        group_plot_data = []
        
        if len(group_data_parsed) > 0:
            if avg_mode == "Average ALL Files into One Curve":
                all_curves = []
                for dat in group_data_parsed:
                    all_curves.extend(dat["curves"])
                
                if all_curves:
                    c_color = combined_palette[0]
                    E_mean, I_mean, I_std = get_averaged_curve(all_curves)
                    df_mean = pd.DataFrame({"Vf": E_mean, "Im": I_mean})
                    trace_name = custom_labels["ALL"]
                    
                    srs = [d['sr'] for d in group_data_parsed if d['sr'] is not None and d['sr'] > 0]
                    fallback_sr = np.mean(srs) if srs else 0.0
                    sr_val = get_sr_from_name(trace_name, fallback_sr)
                    
                    group_plot_data.append({"x": E_mean, "y": I_mean, "std": I_std, "name": trace_name, "df": df_mean, "tech": "LSV" if is_group_lsv else "CV", "sr": sr_val})
                    
                    if show_sd_shadow:
                        fig_comp.add_trace(go.Scatter(x=E_mean, y=I_mean+I_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig_comp.add_trace(go.Scatter(x=E_mean, y=I_mean-I_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                    fig_comp.add_trace(go.Scatter(x=E_mean, y=I_mean, mode='lines', name=trace_name, line=dict(color=c_color, width=2.5)))
                    
                    if is_group_lsv:
                        cat_params, fit_data = extract_lsv_catalytic_parameters(df_mean, electrode_area, e_rev)
                        if cat_params:
                            cat_params = {"File": trace_name, "Curve": "Group Average", **cat_params}
                            group_lsv_params.append(cat_params)
                            max_log_I_global = max(max_log_I_global, fit_data["log_I_max"])
                            fig_tafel_comp.add_trace(go.Scatter(x=fit_data["log_I_full"], y=fit_data["E_full"], mode='lines', name=trace_name, line=dict(color=c_color, width=2.5)))
                            if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                                min_x, max_x = np.min(fit_data["log_I_fit"]), np.max(fit_data["log_I_fit"])
                                span = max_x - min_x
                                fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                                fig_tafel_comp.add_trace(go.Scatter(x=fit_x, y=fit_data["slope"]*fit_x + fit_data["intercept"], mode='lines', name=f"Fit: {cat_params['Tafel Slope (mV/dec)']:.1f} mV/dec", line=dict(color=c_color, width=2, dash='dot')))
                            
                            if show_sd_shadow:
                                j_dens, j_std = (I_mean * 1000) / electrode_area, (I_std * 1000) / electrode_area
                                fig_jeta_comp.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens+j_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                fig_jeta_comp.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens-j_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                            fig_jeta_comp.add_trace(go.Scatter(x=fit_data["eta_mV"], y=fit_data["j_dens"], mode='lines', name=trace_name, line=dict(color=c_color, width=2.5)))

            elif avg_mode == "Average Scans per File":
                for dat in group_data_parsed:
                    c_color = combined_palette[trace_idx % len(combined_palette)]
                    E_mean, I_mean, I_std = get_averaged_curve(dat["curves"])
                    df_mean = pd.DataFrame({"Vf": E_mean, "Im": I_mean})
                    trace_name = custom_labels[dat['fname']]
                    
                    fallback_sr = dat['sr'] if dat['sr'] else 0.0
                    sr_val = get_sr_from_name(trace_name, fallback_sr)
                    
                    group_plot_data.append({"x": E_mean, "y": I_mean, "std": I_std, "name": trace_name, "df": df_mean, "tech": dat["tech"], "sr": sr_val})
                    
                    if show_sd_shadow:
                        fig_comp.add_trace(go.Scatter(x=E_mean, y=I_mean+I_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig_comp.add_trace(go.Scatter(x=E_mean, y=I_mean-I_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                    fig_comp.add_trace(go.Scatter(x=E_mean, y=I_mean, mode='lines', name=trace_name, line=dict(color=c_color, width=2.5)))
                    
                    if "LSV" in dat["tech"]:
                        cat_params, fit_data = extract_lsv_catalytic_parameters(df_mean, electrode_area, e_rev)
                        if cat_params:
                            cat_params = {"File": trace_name, "Curve": "Average", **cat_params}
                            group_lsv_params.append(cat_params)
                            max_log_I_global = max(max_log_I_global, fit_data["log_I_max"])
                            fig_tafel_comp.add_trace(go.Scatter(x=fit_data["log_I_full"], y=fit_data["E_full"], mode='lines', name=trace_name, line=dict(color=c_color, width=2.5)))
                            if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                                min_x, max_x = np.min(fit_data["log_I_fit"]), np.max(fit_data["log_I_fit"])
                                span = max_x - min_x
                                fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                                fig_tafel_comp.add_trace(go.Scatter(x=fit_x, y=fit_data["slope"]*fit_x + fit_data["intercept"], mode='lines', name=f"Fit: {cat_params['Tafel Slope (mV/dec)']:.1f} mV/dec", line=dict(color=c_color, width=2, dash='dot')))
                            
                            if show_sd_shadow:
                                j_dens = (I_mean * 1000) / electrode_area
                                j_std = (I_std * 1000) / electrode_area
                                fig_jeta_comp.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens+j_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                fig_jeta_comp.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens-j_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                            fig_jeta_comp.add_trace(go.Scatter(x=fit_data["eta_mV"], y=fit_data["j_dens"], mode='lines', name=trace_name, line=dict(color=c_color, width=2.5)))
                    trace_idx += 1
            else:
                for dat in group_data_parsed:
                    for cid, dd_comp, Ecol in dat["curves"]:
                        c_color = combined_palette[trace_idx % len(combined_palette)]
                        trace_name = f"{dat['fname']}" if len(dat['curves']) == 1 else f"{dat['fname']} ({cid})"
                        
                        fallback_sr = dat['sr'] if dat['sr'] else 0.0
                        sr_val = get_sr_from_name(trace_name, fallback_sr)
                        
                        group_plot_data.append({"x": dd_comp[Ecol].values, "y": dd_comp["Im"].values, "std": None, "name": trace_name, "df": dd_comp, "tech": dat["tech"], "sr": sr_val})
                        
                        fig_comp.add_trace(go.Scatter(x=dd_comp[Ecol], y=dd_comp["Im"], mode='lines', name=trace_name, line=dict(color=c_color, width=2)))
                        if "LSV" in dat["tech"]:
                            cat_params, fit_data = extract_lsv_catalytic_parameters(dd_comp, electrode_area, e_rev)
                            if cat_params:
                                cat_params = {"File": dat['fname'], "Curve": cid, **cat_params}
                                group_lsv_params.append(cat_params)
                                max_log_I_global = max(max_log_I_global, fit_data["log_I_max"])
                                fig_tafel_comp.add_trace(go.Scatter(x=fit_data["log_I_full"], y=fit_data["E_full"], mode='lines', name=trace_name, line=dict(color=c_color, width=2)))
                                if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                                    min_x, max_x = np.min(fit_data["log_I_fit"]), np.max(fit_data["log_I_fit"])
                                    span = max_x - min_x
                                    fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                                    fig_tafel_comp.add_trace(go.Scatter(x=fit_x, y=fit_data["slope"]*fit_x + fit_data["intercept"], mode='lines', name=f"Fit: {cat_params['Tafel Slope (mV/dec)']:.1f} mV/dec", line=dict(color=c_color, width=2, dash='dot')))
                                fig_jeta_comp.add_trace(go.Scatter(x=fit_data["eta_mV"], y=fit_data["j_dens"], mode='lines', name=trace_name, line=dict(color=c_color, width=2)))
                        trace_idx += 1
                        
            prepared_group_data[group['header']] = {"traces": group_plot_data, "is_lsv": is_group_lsv}
            
            fig_comp.update_layout(title="", xaxis_title=x_axis_label, yaxis_title=i_axis_label, height=500)
            fig_comp = apply_scientific_style(fig_comp, scientific_style, lx, ly, lxa, lya)
            st.plotly_chart(fig_comp, use_container_width=True, config=dl_config)
            
            if is_group_lsv:
                c1, c2 = st.columns(2)
                with c1:
                    fig_jeta_comp.update_layout(title="", xaxis_title="Overpotential η (mV)", yaxis_title=j_axis_label, height=500)
                    fig_jeta_comp = apply_scientific_style(fig_jeta_comp, scientific_style, lx, ly, lxa, lya)
                    st.plotly_chart(fig_jeta_comp, use_container_width=True, config=dl_config)
                with c2:
                    fig_tafel_comp.update_layout(title="", xaxis_title="log₁₀|I| (A)", yaxis_title=x_axis_label, xaxis=dict(range=[max_log_I_global - 4.5, max_log_I_global + 0.2]), height=500)
                    fig_tafel_comp = apply_scientific_style(fig_tafel_comp, scientific_style, lx, ly, lxa, lya)
                    st.plotly_chart(fig_tafel_comp, use_container_width=True, config=dl_config)
            
            if group_lsv_params:
                st.markdown("#### 🧪 Group Catalytic Statistics (LSV)")
                df_cat = pd.DataFrame(group_lsv_params)
                summary = []
                cols_to_summarize = ["j_max (mA/cm²)", "E_onset (V)", "Tafel Slope (mV/dec)", "Tafel R²"] + [c for c in df_cat.columns if c.startswith("η_")]
                for col in cols_to_summarize:
                    if col in df_cat.columns:
                        mean_v, std_v, n_v = df_cat[col].mean(), df_cat[col].std(), df_cat[col].notna().sum()
                        rsd_v = (std_v / abs(mean_v) * 100) if (pd.notna(mean_v) and mean_v != 0) else np.nan
                        summary.append({"Parameter": col, "Mean": round(mean_v, 6) if pd.notna(mean_v) else None, "Std Dev (±)": round(std_v, 6) if pd.notna(std_v) else None, "RSD (%)": round(rsd_v, 2) if pd.notna(rsd_v) else None, "Count (n)": int(n_v)})
                st.dataframe(pd.DataFrame(summary), use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

    # --- ZONA: SUPER GROUPS ---
    st.markdown("---")
    st.header("🧬 Super Groups (Combine Groups)")
    st.markdown("Merge multiple groups into a single master plot. **Ideal for combining averaged 'Species' data and calculating Kinetics.**")
    
    if 'num_super_groups' not in st.session_state:
        st.session_state.num_super_groups = 0

    col_sg1, col_sg2, _ = st.columns([1, 1, 6])
    with col_sg1:
        if st.button("➕ Add Super Group"):
            st.session_state.num_super_groups += 1
    with col_sg2:
        if st.session_state.num_super_groups > 0:
            if st.button("➖ Remove Last"):
                st.session_state.num_super_groups -= 1

    for sg in range(st.session_state.num_super_groups):
        with st.expander(f"Super Group {sg+1} Configurations", expanded=True):
            selected_groups = st.multiselect(
                "Select groups to merge:", 
                options=valid_groups_for_super, 
                key=f"super_group_select_{sg}"
            )
            
            if selected_groups:
                st.markdown("**Customize Legend Labels for this Super Group:**")
                sg_custom_labels = {}
                cols = st.columns(3)
                for i, g_name in enumerate(selected_groups):
                    clean_name = g_name.replace("📊 ", "")
                    sg_custom_labels[g_name] = cols[i%3].text_input(f"Label for {clean_name}", value=clean_name, key=f"sg_lbl_{sg}_{i}")

                fig_super = go.Figure()
                fig_super_tafel = go.Figure()
                fig_super_jeta = go.Figure()
                is_sg_lsv = False
                sg_lsv_params = []
                sg_max_log_I = -10
                
                sg_cv_kinetics = []
                
                for g_idx, g_name in enumerate(selected_groups):
                    if g_name not in prepared_group_data: continue
                    g_data = prepared_group_data[g_name]
                    if g_data["is_lsv"]: is_sg_lsv = True
                    
                    base_color = combined_palette[g_idx % len(combined_palette)]
                    
                    for tr_idx, tr in enumerate(g_data["traces"]):
                        c_color = base_color
                        
                        if len(g_data["traces"]) == 1:
                            final_name = sg_custom_labels[g_name]
                        else:
                            final_name = f"{sg_custom_labels[g_name]} - {tr['name']}"
                            
                        # Capture Scan Rate kinetics if CV
                        if not g_data["is_lsv"] and tr.get("sr") and tr["sr"] > 0:
                            max_idx = np.argmax(tr["y"])
                            i_pa = tr["y"][max_idx]
                            E_pa = tr["x"][max_idx]
                            j_pa = (i_pa * 1000) / electrode_area
                            
                            if j_pa > 0:
                                sg_cv_kinetics.append({
                                    "Curve": final_name,
                                    "v (mV/s)": tr["sr"],
                                    "log_v": np.log10(tr["sr"]),
                                    "E_p (V)": E_pa,
                                    "j_p (mA/cm²)": j_pa,
                                    "log_jp": np.log10(j_pa)
                                })
                        
                        if tr["std"] is not None and show_sd_shadow:
                            fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"]+tr["std"], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                            fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"]-tr["std"], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                        fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode='lines', name=final_name, line=dict(color=c_color, width=2.5)))
                        
                        if "LSV" in tr["tech"]:
                            cat_params, fit_data = extract_lsv_catalytic_parameters(tr["df"], electrode_area, e_rev)
                            if cat_params:
                                sg_lsv_params.append({"Group": g_name, "Curve": tr["name"], **cat_params})
                                sg_max_log_I = max(sg_max_log_I, fit_data["log_I_max"])
                                fig_super_tafel.add_trace(go.Scatter(x=fit_data["log_I_full"], y=fit_data["E_full"], mode='lines', name=final_name, line=dict(color=c_color, width=2.5)))
                                if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                                    min_x, max_x = np.min(fit_data["log_I_fit"]), np.max(fit_data["log_I_fit"])
                                    span = max_x - min_x
                                    fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                                    fig_super_tafel.add_trace(go.Scatter(x=fit_x, y=fit_data["slope"]*fit_x + fit_data["intercept"], mode='lines', name=f"Fit: {cat_params['Tafel Slope (mV/dec)']:.1f} mV/dec", line=dict(color=c_color, width=2, dash='dot')))
                                
                                if tr["std"] is not None and show_sd_shadow:
                                    j_dens = (tr["y"] * 1000) / electrode_area
                                    j_std = (tr["std"] * 1000) / electrode_area
                                    fig_super_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens+j_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                    fig_super_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens-j_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                                fig_super_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=fit_data["j_dens"], mode='lines', name=final_name, line=dict(color=c_color, width=2.5)))

                fig_super.update_layout(title="", xaxis_title=x_axis_label, yaxis_title=i_axis_label, height=500)
                fig_super = apply_scientific_style(fig_super, scientific_style, lx, ly, lxa, lya)
                st.plotly_chart(fig_super, use_container_width=True, config=dl_config)
                
                if is_sg_lsv:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_super_jeta.update_layout(title="", xaxis_title="Overpotential η (mV)", yaxis_title=j_axis_label, height=500)
                        fig_super_jeta = apply_scientific_style(fig_super_jeta, scientific_style, lx, ly, lxa, lya)
                        st.plotly_chart(fig_super_jeta, use_container_width=True, config=dl_config)
                    with c2:
                        fig_super_tafel.update_layout(title="", xaxis_title="log₁₀|I| (A)", yaxis_title=x_axis_label, xaxis=dict(range=[sg_max_log_I - 4.5, sg_max_log_I + 0.2]), height=500)
                        fig_super_tafel = apply_scientific_style(fig_super_tafel, scientific_style, lx, ly, lxa, lya)
                        st.plotly_chart(fig_super_tafel, use_container_width=True, config=dl_config)
                        
                elif len(sg_cv_kinetics) > 1:
                    st.markdown("#### 🔋 CV Kinetics ($b$-value Determination)")
                    df_cv = pd.DataFrame(sg_cv_kinetics).sort_values("v (mV/s)")
                    
                    slope, intercept, r_value, p_value, std_err = linregress(df_cv["log_v"], df_cv["log_jp"])
                    r2 = r_value**2
                    
                    fig_b = go.Figure()
                    fig_b.add_trace(go.Scatter(x=df_cv["log_v"], y=df_cv["log_jp"], mode='markers', marker=dict(size=10, color='black'), name="Data points"))
                    
                    fit_x = np.array([df_cv["log_v"].min() - 0.1, df_cv["log_v"].max() + 0.1])
                    fit_y = slope * fit_x + intercept
                    fig_b.add_trace(go.Scatter(x=fit_x, y=fit_y, mode='lines', line=dict(color='red', dash='dash'), name=f"Fit: b = {slope:.2f}"))
                    
                    fig_b.update_layout(
                        xaxis_title="log₁₀(v) [mV/s]",
                        yaxis_title="log₁₀(j_p) [mA cm⁻²]" if scientific_style else "log₁₀(j_p) [mA/cm²]",
                        height=450
                    )
                    fig_b = apply_scientific_style(fig_b, scientific_style, lx, ly, lxa, lya)
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.plotly_chart(fig_b, use_container_width=True, config=dl_config)
                    with c2:
                        st.metric("b-value (Slope ± SE)", f"{slope:.4f} ± {std_err:.4f}", f"R² = {r2:.4f}", delta_color="off")
                        st.info("💡 **b = 0.5**: Diffusion-controlled process. **b = 1.0**: Surface-controlled (capacitive) process.")
                        st.dataframe(df_cv[["Curve", "v (mV/s)", "E_p (V)", "j_p (mA/cm²)"]].style.format({"v (mV/s)": "{:.1f}", "E_p (V)": "{:.3f}", "j_p (mA/cm²)": "{:.4f}"}), use_container_width=True)

    # --- BOTTOM AREA: INDIVIDUAL ANALYSIS ---
    st.markdown("---")
    st.header("📄 Individual Analysis")

    actual_file_order = [item.replace("⋮⋮ ", "") for group in st.session_state.file_groups for item in group["items"]]

    for file_name in actual_file_order:
        if file_name not in file_dict: continue 
            
        file = file_dict[file_name]
        raw_text = file.getvalue().decode("utf-8", errors="replace")
        technique = "Unknown Technique"

        if instrument.startswith("Gamry"):
            meta, curves = parse_gamry_dta_multi_curve(raw_text)
            technique = "Linear Sweep Voltammetry (LSV)" if "LSV" in meta.get("TAG", "").upper() or "LINEAR" in meta.get("TITLE", "").upper() else "Cyclic Voltammetry (CV)"
            sr = _to_float(meta.get("SCANRATE"))
        elif instrument.startswith("PalmSens"):
            meta, curves = parse_pstrace_csv(file.getvalue())
            technique = meta.get("TECHNIQUE", "Unknown Technique")
            sr = None
        else:
            meta, curves = parse_biologic_mpt(raw_text)
            technique = "Cyclic Voltammetry (CV)" if "E2 (V)" in meta else "Linear Sweep Voltammetry (LSV)"
            sr = _to_float(meta.get("dE/dt"))

        if manual_scan_rate > 0.0: sr = manual_scan_rate
        if not curves: st.error(f"❌ Could not parse {file.name}. Check format."); continue

        vinit, vlim1, vlim2 = extract_limits_from_data(curves[0][1], technique)

        st.markdown(f"### {file.name}")
        col_title, col_btn = st.columns([4, 1])
        with col_title: st.markdown(f"🔬 **Technique Detected:** `{technique}`")
        with col_btn: st.download_button("📥 Export to Excel", convert_df_to_excel(curves), f"{file.name.split('.')[0]}_Data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_{file.name}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial Potential", f"{vinit} V" if vinit is not None else "N/A")
        if "LSV" in technique:
            c2.metric("Final Potential", f"{vlim1} V" if vlim1 is not None else "N/A")
            c3.metric("Scan Limit 2", "N/A")
        else:
            c2.metric("Scan Limit 1", f"{vlim1} V" if vlim1 is not None else "N/A")
            c3.metric("Scan Limit 2", f"{vlim2} V" if vlim2 is not None else "N/A")
        c4.metric("Scan Rate", f"{sr} mV/s" if sr is not None else "N/A")

        processed_curves = []
        for i, (cid, dfi) in enumerate(curves):
            Ecol = "Vf" if "Vf" in dfi.columns else ("Vu" if "Vu" in dfi.columns else None)
            if Ecol is None or "Im" not in dfi.columns: continue
            dd = dfi[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
            if apply_ir: dd[Ecol] = dd[Ecol] - dd["Im"] * ru_ohms * (comp_percent / 100.0)
            if convert_to_rhe: dd[Ecol] = dd[Ecol] + e0_ref + (0.0591 * ph_val)
            if len(dd) >= 10: processed_curves.append((cid, dd, Ecol))

        if not processed_curves: continue

        fig, fig_tafel, fig_jeta = go.Figure(), go.Figure(), go.Figure()
        results_list, lsv_cat_list, max_log_I_ind = [], [], -10

        avg_cycles = st.toggle(f"🌟 Average {len(processed_curves)} Cycles/Scans", key=f"avg_{file.name}") if len(processed_curves) > 1 else False

        if avg_cycles:
            E_mean, I_mean, I_std = get_averaged_curve(processed_curves)
            mean_color = combined_palette[0]
            
            if show_sd_shadow:
                fig.add_trace(go.Scatter(x=E_mean, y=I_mean + I_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=E_mean, y=I_mean - I_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(mean_color, 0.2), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=E_mean, y=I_mean, mode='lines', name='Average', line=dict(color=mean_color, width=2.5)))
            
            df_mean = pd.DataFrame({"Vf": E_mean, "Im": I_mean})
            out = recommend_operating_ranges_for_curve(df_mean)
            ns, ro = out["recommended_noise_safe_V"], out["recommended_reduction_only_V"]
            results_list.append({"Curve": "Average Curve", "Points": out["N_points"], "Noise-Safe Min (V)": round(ns[0], 4) if ns else None, "Noise-Safe Max (V)": round(ns[1], 4) if ns else None, "Reduction Min (V)": round(ro[0], 4) if ro else None, "Reduction Max (V)": round(ro[1], 4) if ro else None})
            
            if "LSV" in technique:
                cat_params, fit_data = extract_lsv_catalytic_parameters(df_mean, electrode_area, e_rev)
                if cat_params:
                    cat_params = {"Curve": "Average Curve", **cat_params}
                    lsv_cat_list.append(cat_params)
                    max_log_I_ind = max(max_log_I_ind, fit_data["log_I_max"])
                    fig_tafel.add_trace(go.Scatter(x=fit_data["log_I_full"], y=fit_data["E_full"], mode='lines', name="Average (Log Curve)", line=dict(color=mean_color, width=2.5)))
                    if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                        min_x, max_x = np.min(fit_data["log_I_fit"]), np.max(fit_data["log_I_fit"])
                        span = max_x - min_x
                        fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                        fig_tafel.add_trace(go.Scatter(x=fit_x, y=fit_data["slope"]*fit_x + fit_data["intercept"], mode='lines', name=f"Fit: {cat_params['Tafel Slope (mV/dec)']:.1f} mV/dec", line=dict(color=mean_color, width=2, dash='dot')))
                    
                    if show_sd_shadow:
                        j_dens, j_std = (I_mean * 1000) / electrode_area, (I_std * 1000) / electrode_area
                        fig_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens+j_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens-j_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(mean_color, 0.2), showlegend=False, hoverinfo='skip'))
                    fig_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=fit_data["j_dens"], mode='lines', name="Average", line=dict(color=mean_color, width=2.5)))
        else:
            for i, (cid, dd, Ecol) in enumerate(processed_curves):
                line_color = combined_palette[i % len(combined_palette)]
                fig.add_trace(go.Scatter(x=dd[Ecol], y=dd["Im"], mode='lines', name=cid, line=dict(color=line_color, width=2)))
                out = recommend_operating_ranges_for_curve(dd)
                ns, ro = out["recommended_noise_safe_V"], out["recommended_reduction_only_V"]
                results_list.append({"Curve": cid, "Points": out["N_points"], "Noise-Safe Min (V)": round(ns[0], 4) if ns else None, "Noise-Safe Max (V)": round(ns[1], 4) if ns else None, "Reduction Min (V)": round(ro[0], 4) if ro else None, "Reduction Max (V)": round(ro[1], 4) if ro else None})
                
                if "LSV" in technique:
                    cat_params, fit_data = extract_lsv_catalytic_parameters(dd, electrode_area, e_rev)
                    if cat_params:
                        cat_params = {"Curve": cid, **cat_params}
                        lsv_cat_list.append(cat_params)
                        max_log_I_ind = max(max_log_I_ind, fit_data["log_I_max"])
                        fig_tafel.add_trace(go.Scatter(x=fit_data["log_I_full"], y=fit_data["E_full"], mode='lines', name=f"{cid}", line=dict(color=line_color, width=2)))
                        if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                            min_x, max_x = np.min(fit_data["log_I_fit"]), np.max(fit_data["log_I_fit"])
                            span = max_x - min_x
                            fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                            fig_tafel.add_trace(go.Scatter(x=fit_x, y=fit_data["slope"]*fit_x + fit_data["intercept"], mode='lines', name=f"Fit: {cat_params['Tafel Slope (mV/dec)']:.1f} mV/dec", line=dict(color=line_color, width=2, dash='dot')))
                        fig_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=fit_data["j_dens"], mode='lines', name=cid, line=dict(color=line_color, width=2)))

        fig.update_layout(title="", xaxis_title=x_axis_label, yaxis_title=i_axis_label, height=500)
        fig = apply_scientific_style(fig, scientific_style, lx, ly, lxa, lya)
        st.plotly_chart(fig, use_container_width=True, config=dl_config)
        
        if "LSV" in technique and lsv_cat_list:
            c1, c2 = st.columns(2)
            with c1:
                fig_jeta.update_layout(title="", xaxis_title="Overpotential η (mV)", yaxis_title=j_axis_label, height=500)
                fig_jeta = apply_scientific_style(fig_jeta, scientific_style, lx, ly, lxa, lya)
                st.plotly_chart(fig_jeta, use_container_width=True, config=dl_config)
            with c2:
                fig_tafel.update_layout(title="", xaxis_title="log₁₀|I| (A)", yaxis_title=x_axis_label, xaxis=dict(range=[max_log_I_ind - 4.5, max_log_I_ind + 0.2]), height=500)
                fig_tafel = apply_scientific_style(fig_tafel, scientific_style, lx, ly, lxa, lya)
                st.plotly_chart(fig_tafel, use_container_width=True, config=dl_config)
        
        if results_list: st.write("**Recommended Operating Ranges:**"); st.dataframe(pd.DataFrame(results_list), use_container_width=True)
        if lsv_cat_list: st.write("**🧪 Catalytic Parameters:**"); st.dataframe(pd.DataFrame(lsv_cat_list), use_container_width=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
