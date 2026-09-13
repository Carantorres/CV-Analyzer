import os
import re
import io
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress
from scipy.optimize import curve_fit
import streamlit as st
import streamlit.components.v1 as components
from streamlit_sortables import sort_items

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="CV & EIS Analyzer", layout="wide")
st.title("📊 Universal CV, LSV & EIS Analyzer")

st.markdown("""
**A comprehensive tool for automated electrochemical data analysis.** 
Seamlessly process CV, LSV, and EIS files. Features robust catalytic parameter extraction, noise-free potential window detection, and equivalent circuit fitting for impedance spectroscopy.
""")

with st.popover("📖 View Calculation Methods & Algorithms"):
    st.markdown("""
    **1. Physico-Chemical Corrections**
    *   **iR Drop Compensation:** Corrects for the uncompensated resistance ($R_u$) of the electrolyte.
    *   **RHE Scale Conversion:** Shifts the potential to a pH-independent thermodynamic scale.

    **2. Catalytic Parameter Extraction (LSV)**
    *   **Onset Potential ($E_{onset}$):** Point where $|I|$ reaches 5% of the absolute maximum current.
    *   **Robust Tafel Slope:** Computed using a dynamic Sliding Window algorithm, maximizing $R^2$.

    **3. Averaging & Statistical Analysis**
    *   **Cycle Averaging:** Individual scans are mapped and interpolated over a normalized coordinate system.

    **4. Scan Rate Kinetics ($b$-value Smart Detection)**
    *   Uses `scipy.signal.find_peaks` to identify true mathematical local maxima.
    *   A linear regression of $\\log_{10}(j_{pa})$ vs $\\log_{10}(v)$ calculates the slope $b$.
    
    **5. Electrochemical Impedance Spectroscopy (EIS)**
    *   **Nyquist & Bode Plots:** Renders $-Z''$ vs $Z'$ (1:1 ratio), Bode $|Z|$, and Bode Phase natively.
    *   **UOR Equivalent Circuit Fitting:** Applies Non-Linear Least Squares (CNLS) to fit the specific UOR model: `Rs-(CPE1||(Rct+(RL||L)))` including an inductive relaxation loop.
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

def get_averaged_curve(processed_curves: List[Tuple[str, pd.DataFrame]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not processed_curves: return None, None, None
    max_points = max(len(dd) for _, dd in processed_curves)
    common_idx = np.linspace(0, 1, max_points)
    
    E_interp, I_interp = [], []
    for _, dd in processed_curves:
        idx = np.linspace(0, 1, len(dd))
        E_interp.append(np.interp(common_idx, idx, dd["x"].values))
        I_interp.append(np.interp(common_idx, idx, dd["y"].values))
        
    return np.mean(E_interp, axis=0), np.mean(I_interp, axis=0), np.std(I_interp, axis=0)

def get_averaged_eis_curve(processed_curves: List[Tuple[str, pd.DataFrame]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not processed_curves: return None, None, None, None, None
    
    all_f = np.concatenate([dd["f"].values for _, dd in processed_curves])
    min_f, max_f = np.min(all_f), np.max(all_f)
    max_points = max(len(dd) for _, dd in processed_curves)
    common_f = np.logspace(np.log10(max_f), np.log10(min_f), max_points)
    
    Zr_interp, Zi_interp = [], []
    for _, dd in processed_curves:
        f_vals = dd["f"].values
        sort_idx = np.argsort(f_vals)
        f_sorted = f_vals[sort_idx]
        zr_sorted = dd["x"].values[sort_idx]
        zi_sorted = dd["y"].values[sort_idx]
        
        Zr_interp.append(np.interp(common_f, f_sorted, zr_sorted))
        Zi_interp.append(np.interp(common_f, f_sorted, zi_sorted))
        
    return common_f[::-1], np.mean(Zr_interp, axis=0)[::-1], np.mean(Zi_interp, axis=0)[::-1], np.std(Zr_interp, axis=0)[::-1], np.std(Zi_interp, axis=0)[::-1]

# --- EIS FITTING MODEL ---
def eis_model_obj(f, Rs, CPE_T, CPE_P, Rct, RL, L):
    w = 2 * np.pi * f
    Z_CPE = 1.0 / (CPE_T * (1j * w)**CPE_P)
    Z_L = 1j * w * L
    Z_ind = 1.0 / (1.0/RL + 1.0/Z_L)
    Z_faradaic = Rct + Z_ind
    Z_total = Rs + 1.0 / (1.0/Z_CPE + 1.0/Z_faradaic)
    return np.hstack([Z_total.real, -Z_total.imag])

def fit_uor_eis(f, zr, zi):
    y_data = np.hstack([zr, zi])
    Z_data = zr - 1j * zi
    Rs_fixed = np.min(zr) 
    
    def obj_func_fixed(f_val, CPE_T, CPE_P, Rct, RL, L):
        return eis_model_obj(f_val, Rs_fixed, CPE_T, CPE_P, Rct, RL, L)
        
    p0 = [1e-4, 0.8, np.abs(np.max(zr)-np.min(zr)), np.abs(np.max(zr)-np.min(zr))*0.5, 1000]
    bounds = ([1e-9, 0.5, 0, -1e6, 1e-5], [1.0, 1.0, 1e6, 1e6, 1e8])
    
    try:
        popt, pcov = curve_fit(obj_func_fixed, f, y_data, p0=p0, bounds=bounds, maxfev=100000)
        Z_fit_arr = eis_model_obj(f, Rs_fixed, *popt)
        N = len(f)
        Z_fit_real = Z_fit_arr[:N]
        Z_fit_imag = Z_fit_arr[N:]
        
        SStot = np.sum((y_data - np.mean(y_data))**2)
        SSres = np.sum((y_data - Z_fit_arr)**2)
        R2 = 1 - (SSres / SStot)
        
        weights = np.abs(Z_data)**2
        Z_fit_complex = Z_fit_real - 1j*Z_fit_imag
        chi2 = np.sum((np.abs(Z_data - Z_fit_complex)**2) / weights) / (len(f) - 5)
        
        errs = np.sqrt(np.diag(pcov))
        rel_errs = (errs / np.abs(popt)) * 100
        
        return [Rs_fixed] + list(popt), [0.0] + list(rel_errs), R2, chi2, Z_fit_real, Z_fit_imag
    except Exception as e:
        return None, None, None, None, None, None

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

def extract_lsv_catalytic_parameters(df_curve: pd.DataFrame, area_cm2: float, e_rev: float) -> Tuple[dict, dict]:
    if "x" not in df_curve.columns or "y" not in df_curve.columns: return {}, {}
    df_sorted = df_curve.sort_values(by="x").reset_index(drop=True)
    E, I = df_sorted["x"].values, df_sorted["y"].values
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
    except: I_smooth = abs_I
    I_smooth = np.where(I_smooth <= 0, 1e-12, I_smooth) 
    
    fit_data = {
        "E_full": E, "log_I_full": np.log10(I_smooth),
        "log_I_fit": best_log_I_fit, "slope": best_slope, "intercept": best_intercept,
        "log_I_max": np.log10(I_max) if I_max > 0 else 0,
        "j_dens": j_dens, "eta_mV": eta_mV
    }
    return params, fit_data

def apply_scientific_style(fig, is_scientific, lx, ly, lxa, lya):
    if is_scientific:
        fig.update_layout(
            title="", plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=16, color="black"),
            margin=dict(l=80, r=40, t=40, b=60)
        )
        fig.update_xaxes(showgrid=False, showline=True, linecolor='black', linewidth=2, mirror="all", ticks='inside', tickcolor='black', tickwidth=2, ticklen=8, title_font=dict(size=18, family="Arial, sans-serif", color="black"), tickfont=dict(size=15, family="Arial, sans-serif", color="black"), zeroline=False)
        fig.update_yaxes(showgrid=False, showline=True, linecolor='black', linewidth=2, mirror="all", ticks='inside', tickcolor='black', tickwidth=2, ticklen=8, title_font=dict(size=18, family="Arial, sans-serif", color="black"), tickfont=dict(size=15, family="Arial, sans-serif", color="black"), zeroline=False)
    fig.update_layout(legend=dict(x=lx, y=ly, xanchor=lxa, yanchor=lya, bgcolor='rgba(255, 255, 255, 0.9)', bordercolor='black', borderwidth=1 if is_scientific else 0, font=dict(size=14, color="black" if is_scientific else None)))
    return fig

# ============================================================
# PARSERS
# ============================================================
def parse_pstrace_csv(raw: bytes) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    text = None
    for enc in ['utf-8', 'utf-16', 'latin1']:
        try:
            text = raw.decode(enc)
            if "Linear Sweep" in text or "Cyclic Voltammetry" in text or "Impedance" in text: break
        except: continue
    if not text: return {}, []
        
    lines = text.splitlines()
    meta = {}
    curves = []
    scan_names = []
    unit_row_idx = -1
    is_eis = False
    
    for line in lines[:20]:
        if "Linear Sweep" in line or "LSV" in line: meta["TECHNIQUE"] = "Linear Sweep Voltammetry (LSV)"
        elif "Cyclic Voltammetry" in line or "CV" in line: meta["TECHNIQUE"] = "Cyclic Voltammetry (CV)"
        elif "Impedance" in line or "EIS" in line: 
            meta["TECHNIQUE"] = "Electrochemical Impedance Spectroscopy (EIS)"
            is_eis = True
            
    if is_eis:
        for i, line in enumerate(lines[:50]):
            if "freq / Hz" in line:
                unit_row_idx = i; break
        if unit_row_idx != -1:
            cols = [p.strip() for p in lines[unit_row_idx].split(",") if p.strip()]
            rows = []
            for line in lines[unit_row_idx+1:]:
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) > len(cols): parts = parts[:len(cols)]
                elif len(parts) < len(cols): parts += [""] * (len(cols) - len(parts))
                rows.append(parts)
            df = pd.DataFrame(rows, columns=cols).replace("", np.nan).dropna(axis=1, how='all')
            for c in df.columns:
                if c: df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
            
            df_clean = pd.DataFrame()
            if "Z' / Ohm" in df.columns: df_clean["Z_real"] = df["Z' / Ohm"]
            if "-Z'' / Ohm" in df.columns: df_clean["neg_Z_imag"] = df["-Z'' / Ohm"]
            elif "Z'' / Ohm" in df.columns: df_clean["neg_Z_imag"] = -df["Z'' / Ohm"]
            if "freq / Hz" in df.columns: df_clean["Frequency"] = df["freq / Hz"]
            
            if "Z_real" in df_clean.columns and "neg_Z_imag" in df_clean.columns and "Frequency" in df_clean.columns:
                df_clean = df_clean.dropna()
                curves.append(("EIS Data", df_clean))
        return meta, curves
    
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

def parse_gamry_dta_multi_curve(raw: str): return {}, [] # Simplified for space, keep your original if needed
def parse_biologic_mpt(raw: str): return {}, []

def convert_df_to_excel(curves_list: List[Tuple[str, pd.DataFrame]]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        seen_names = set()
        for cid, df in curves_list:
            safe_name = re.sub(r'[\\*?:/\[\]]', '_', cid)[:31].strip() or "Sheet"
            original_safe_name, counter = safe_name, 1
            while safe_name in seen_names:
                safe_name = f"{original_safe_name[:31-len(str(counter))-1]}_{counter}"
                counter += 1
            seen_names.add(safe_name)
            df.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()

# ============================================================
# APP LOGIC
# ============================================================
uploaded_files = st.file_uploader("Upload CV/LSV/EIS files", type=["csv", "CSV", "DTA", "dta", "mpt", "MPT"], accept_multiple_files=True)

publication_palette = ['#000000', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628', '#F781BF'] + px.colors.qualitative.Alphabet
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
    custom_ref_name = st.text_input("Custom Reference Label", value="Ref.") if ref_elec == "Custom" else ref_elec.split(" (")[0]
        
    convert_to_rhe = st.toggle("Convert E to RHE scale", value=True)
    if convert_to_rhe:
        e0_ref = st.number_input("Custom E0_Ref (V)", value=0.000, step=0.01) if ref_elec == "Custom" else (0.197 if ref_elec.startswith("Ag") else (0.241 if ref_elec.startswith("SCE") else 0.098))
        if ref_elec != "Custom": st.info(f"Using Standard E₀ = {e0_ref} V")
        ph_val = st.number_input("pH of the solution", value=14.0, step=0.1)
        x_axis_label = "E (V vs RHE)" + (" [iR corrected]" if apply_ir else "")
    else:
        e0_ref, ph_val = 0.0, 0.0
        x_axis_label = f"E (V vs {custom_ref_name})" + (" [iR corrected]" if apply_ir else "")

    st.markdown("---")
    st.header("⚙️ Catalytic Parameters")
    electrode_area = st.number_input("Electrode Area (cm²)", min_value=0.00001, value=1.00000, step=0.001, format="%.5f")
    manual_scan_rate = st.number_input("Manual Scan Rate (mV/s) [Optional]", value=0.0, step=10.0)
    e_rev = st.number_input("Thermodynamic Potential (E_rev)", value=0.000, step=0.01)

    st.markdown("---")
    st.header("🔋 Peak Search (Kinetics)")
    limit_peak_search = st.toggle("Limit Peak Search Window", value=False)
    if limit_peak_search:
        c_min, c_max = st.columns(2)
        with c_min: peak_min_v = st.number_input("Min E (V)", value=0.20, step=0.05)
        with c_max: peak_max_v = st.number_input("Max E (V)", value=0.60, step=0.05)
    else:
        peak_min_v, peak_max_v = None, None

    st.markdown("---")
    st.header("🎨 Plot Formatting")
    scientific_style = st.toggle("Scientific Paper Style (ACS/Elsevier)", value=True)
    show_sd_shadow = st.toggle("Show SD Shadow on Averages", value=True)
    leg_pos = st.selectbox("Quick Positions", ["Top-Right", "Top-Left", "Bottom-Right", "Bottom-Left", "Outside Right", "Custom..."])
    if leg_pos == "Top-Right": lx, ly, lxa, lya = 0.99, 0.99, "right", "top"
    elif leg_pos == "Top-Left": lx, ly, lxa, lya = 0.01, 0.99, "left", "top"
    elif leg_pos == "Bottom-Right": lx, ly, lxa, lya = 0.99, 0.01, "right", "bottom"
    elif leg_pos == "Bottom-Left": lx, ly, lxa, lya = 0.01, 0.01, "left", "bottom"
    elif leg_pos == "Outside Right": lx, ly, lxa, lya = 1.02, 1.0, "left", "top"
    else:
        lx = st.slider("X Coordinate", min_value=-0.2, max_value=1.5, value=0.99, step=0.01)
        ly = st.slider("Y Coordinate", min_value=-0.2, max_value=1.5, value=0.99, step=0.01)
        lxa, lya = "auto", "auto"
    
    st.markdown("---")
    st.header("📄 Export Full Report")
    components.html("""<button onclick="window.parent.print();" style="background-color:#FF4B4B; color:white; border:none; border-radius:4px; padding:0.5rem 1rem; font-size:1rem; font-weight:600; cursor:pointer; width:100%;">🖨️ Save Page as PDF</button>""", height=50)
    st.markdown("<div style='text-align: center; margin-top: 50px;'><p style='color: #888888; font-size: 0.85rem; font-family: sans-serif;'>Developed by<br><b>PhD(c) Carlos A. Torres-Ramírez</b><br><br></p></div>", unsafe_allow_html=True)

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
            bc1, bc2 = st.columns([2, 1])
            with bc1: target_g = st.selectbox("Target", [g["header"] for g in st.session_state.file_groups[1:]], label_visibility="collapsed")
            with bc2:
                if st.button("Move All"):
                    for g in st.session_state.file_groups:
                        if g["header"] == target_g:
                            g["items"].extend(st.session_state.file_groups[0]["items"]); st.session_state.file_groups[0]["items"] = []; st.rerun()
        st.session_state.file_groups = sort_items(st.session_state.file_groups, multi_containers=True)

    prepared_group_data = {}
    valid_groups_for_super = []
    
    for g_idx, group in enumerate(st.session_state.file_groups):
        if g_idx == 0 or not group["items"]: continue 
        valid_groups_for_super.append(group["header"])
            
        group_data_parsed = []
        is_group_lsv, is_group_eis = False, False
        
        for item in group["items"]:
            fname = item.replace("⋮⋮ ", "")
            if fname not in file_dict: continue
            raw_bytes = file_dict[fname].getvalue()
            
            # Universal parsing dispatch
            meta_sg, curves_comp = parse_pstrace_csv(raw_bytes)
            if not curves_comp:
                try:
                    raw_str = raw_bytes.decode('utf-8')
                    if instrument.startswith("Gamry"): meta_sg, curves_comp = parse_gamry_dta_multi_curve(raw_str)
                    else: meta_sg, curves_comp = parse_biologic_mpt(raw_str)
                except: pass
            
            tech_sg = meta_sg.get("TECHNIQUE", "")
            sr = manual_scan_rate if manual_scan_rate > 0.0 else _to_float(meta_sg.get("SCANRATE", meta_sg.get("dE/dt")))
            
            if "LSV" in tech_sg: is_group_lsv = True
            if "EIS" in tech_sg: is_group_eis = True
                
            processed_curves = []
            for cid, df_comp in curves_comp:
                if "EIS" in tech_sg:
                    if "Z_real" in df_comp.columns and "neg_Z_imag" in df_comp.columns:
                        dd_comp = df_comp[["Z_real", "neg_Z_imag", "Frequency"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                        dd_comp.columns = ["x", "y", "f"]
                        if len(dd_comp) >= 5: processed_curves.append((cid, dd_comp))
                else:
                    Ecol = "Vf" if "Vf" in df_comp.columns else ("Vu" if "Vu" in df_comp.columns else None)
                    if Ecol and "Im" in df_comp.columns:
                        dd_comp = df_comp[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                        if apply_ir: dd_comp[Ecol] = dd_comp[Ecol] - dd_comp["Im"] * ru_ohms * (comp_percent / 100.0)
                        if convert_to_rhe: dd_comp[Ecol] = dd_comp[Ecol] + e0_ref + (0.0591 * ph_val)
                        dd_comp.columns = ["x", "y"]
                        if len(dd_comp) >= 10: processed_curves.append((cid, dd_comp))
            if processed_curves:
                group_data_parsed.append({"fname": fname, "tech": tech_sg, "sr": sr, "curves": processed_curves})

        group_plot_data = []
        if len(group_data_parsed) > 0:
            for dat in group_data_parsed:
                for cid, dd_comp in dat["curves"]:
                    trace_name = f"{dat['fname']}" if len(dat['curves']) == 1 else f"{dat['fname']} ({cid})"
                    sr_val = get_sr_from_name(trace_name, dat['sr'] if dat['sr'] else 0.0)
                    group_plot_data.append({"x": dd_comp["x"].values, "y": dd_comp["y"].values, "f": dd_comp.get("f", pd.Series(dtype=float)).values, "std": None, "name": trace_name, "df": dd_comp, "tech": dat["tech"], "sr": sr_val})
                        
        prepared_group_data[group['header']] = {"traces": group_plot_data, "is_lsv": is_group_lsv, "is_eis": is_group_eis}

    # --- ZONA: SUPER GROUPS ---
    st.markdown("---")
    st.header("🧬 Super Groups (Combine & Fit)")
    st.markdown("Merge groups into a single plot. **Supports Cycle Averaging, Kinetic Analysis ($b$-value), and EIS Equivalent Circuit Fitting.**")
    
    if 'num_super_groups' not in st.session_state: st.session_state.num_super_groups = 0

    col_sg1, col_sg2, _ = st.columns([1, 1, 6])
    with col_sg1:
        if st.button("➕ Add Super Group"): st.session_state.num_super_groups += 1
    with col_sg2:
        if st.session_state.num_super_groups > 0:
            if st.button("➖ Remove Last"): st.session_state.num_super_groups -= 1

    for sg in range(st.session_state.num_super_groups):
        with st.expander(f"Super Group {sg+1} Configurations", expanded=True):
            selected_groups = st.multiselect("Select groups to merge:", options=valid_groups_for_super, key=f"super_group_select_{sg}")
            
            if selected_groups:
                is_sg_eis = any(prepared_group_data[g]["is_eis"] for g in selected_groups if g in prepared_group_data)
                is_sg_lsv = any(prepared_group_data[g]["is_lsv"] for g in selected_groups if g in prepared_group_data)
                
                # --- OPTIONS ---
                avg_mode = st.radio("Super Group Mode:", ["Plot Individual Files", "Average ALL Files inside each Group"], key=f"sg_avg_{sg}", horizontal=True)
                
                if is_sg_eis: fit_eis_model_toggle = st.toggle("🔋 Perform EIS Equivalent Circuit Fitting: Rs-(CPE||(Rct+(RL||L)))", value=False, key=f"fit_eis_{sg}")
                else: fit_eis_model_toggle = False

                st.markdown("**Customize Legend Labels for this Super Group:**")
                sg_custom_labels = {}
                cols = st.columns(3)
                for i, g_name in enumerate(selected_groups):
                    clean_name = g_name.replace("📊 ", "")
                    sg_custom_labels[g_name] = cols[i%3].text_input(f"Label for {clean_name}", value=clean_name, key=f"sg_lbl_{sg}_{i}")

                fig_super = go.Figure()
                fig_super_tafel, fig_super_jeta = go.Figure(), go.Figure()
                fig_bode_mod, fig_bode_phase = go.Figure(), go.Figure()
                
                sg_lsv_params, sg_cv_kinetics, sg_eis_params = [], [], []
                sg_max_log_I = -10
                
                for g_idx, g_name in enumerate(selected_groups):
                    if g_name not in prepared_group_data: continue
                    g_data = prepared_group_data[g_name]
                    base_color = combined_palette[g_idx % len(combined_palette)]
                    
                    traces_to_plot = []
                    if avg_mode == "Average ALL Files inside each Group" and len(g_data["traces"]) > 0:
                        if is_sg_eis:
                            common_f, zr_mean, zi_mean, zr_std, zi_std = get_averaged_eis_curve([(tr["name"], tr["df"]) for tr in g_data["traces"]])
                            df_mean = pd.DataFrame({"x": zr_mean, "y": zi_mean, "f": common_f})
                            # Standard deviation for complex plane is tricky, using None to keep plot clean
                            traces_to_plot = [{"x": zr_mean, "y": zi_mean, "f": common_f, "std": None, "name": sg_custom_labels[g_name], "df": df_mean, "tech": "EIS"}]
                        else:
                            E_mean, I_mean, I_std = get_averaged_curve([(tr["name"], tr["df"]) for tr in g_data["traces"]])
                            df_mean = pd.DataFrame({"x": E_mean, "y": I_mean})
                            fallback_sr = np.mean([t["sr"] for t in g_data["traces"] if t["sr"]>0]) if any(t["sr"]>0 for t in g_data["traces"]) else 0.0
                            sr_val = get_sr_from_name(sg_custom_labels[g_name], fallback_sr)
                            traces_to_plot = [{"x": E_mean, "y": I_mean, "f": None, "std": I_std, "name": sg_custom_labels[g_name], "df": df_mean, "tech": "LSV" if is_sg_lsv else "CV", "sr": sr_val}]
                    else:
                        for tr in g_data["traces"]:
                            final_name = sg_custom_labels[g_name] if len(g_data["traces"]) == 1 else f"{sg_custom_labels[g_name]} - {tr['name']}"
                            tr_copy = tr.copy()
                            tr_copy["name"] = final_name
                            traces_to_plot.append(tr_copy)

                    for tr in traces_to_plot:
                        c_color = base_color
                        
                        if is_sg_eis:
                            # --- EIS PLOTTING & FITTING ---
                            zr, zi, f_hz = tr["x"], tr["y"], tr["f"]
                            fig_super.add_trace(go.Scatter(x=zr, y=zi, mode='markers', name=tr["name"], marker=dict(color=c_color, size=6)))
                            
                            z_mod = np.sqrt(zr**2 + zi**2)
                            phase = np.degrees(np.arctan2(zi, zr)) # Phase angle (positive)
                            
                            fig_bode_mod.add_trace(go.Scatter(x=f_hz, y=z_mod, mode='markers', name=tr["name"], marker=dict(color=c_color, size=6)))
                            fig_bode_phase.add_trace(go.Scatter(x=f_hz, y=phase, mode='markers', name=tr["name"], marker=dict(color=c_color, size=6)))
                            
                            if fit_eis_model_toggle:
                                popt, err, r2, chi2, zr_fit, zi_fit = fit_uor_eis(f_hz, zr, zi)
                                if popt is not None:
                                    sg_eis_params.append({"Curve": tr["name"], "Rs (Ω)": popt[0], "CPE-T (F s^(n-1))": popt[1], "CPE-P (n)": popt[2], "Rct (Ω)": popt[3], "RL (Ω)": popt[4], "L (H)": popt[5], "R²": r2, "χ²": chi2})
                                    fig_super.add_trace(go.Scatter(x=zr_fit, y=zi_fit, mode='lines', line=dict(color=c_color, width=2, dash='dash'), showlegend=False))
                                    z_mod_fit = np.sqrt(zr_fit**2 + zi_fit**2)
                                    phase_fit = np.degrees(np.arctan2(zi_fit, zr_fit))
                                    fig_bode_mod.add_trace(go.Scatter(x=f_hz, y=z_mod_fit, mode='lines', line=dict(color=c_color, width=2, dash='dash'), showlegend=False))
                                    fig_bode_phase.add_trace(go.Scatter(x=f_hz, y=phase_fit, mode='lines', line=dict(color=c_color, width=2, dash='dash'), showlegend=False))
                        else:
                            # --- CV / LSV PLOTTING & KINETICS ---
                            if not is_sg_lsv and tr.get("sr") and tr["sr"] > 0:
                                x_anodic = tr["x"][:np.argmax(tr["x"])+1] if np.argmax(tr["x"]) > 0 else tr["x"]
                                y_anodic = tr["y"][:np.argmax(tr["x"])+1] if np.argmax(tr["x"]) > 0 else tr["y"]
                                
                                if limit_peak_search and peak_min_v is not None and peak_max_v is not None:
                                    mask = (x_anodic >= peak_min_v) & (x_anodic <= peak_max_v)
                                    if np.any(mask):
                                        peaks, _ = find_peaks(y_anodic[mask])
                                        if len(peaks) > 0:
                                            b_idx = peaks[np.argmax(y_anodic[mask][peaks])]
                                            i_pa, E_pa = y_anodic[mask][b_idx], x_anodic[mask][b_idx]
                                        else:
                                            max_idx = np.argmax(y_anodic[mask])
                                            i_pa, E_pa = y_anodic[mask][max_idx], x_anodic[mask][max_idx]
                                    else:
                                        max_idx = np.argmax(y_anodic)
                                        i_pa, E_pa = y_anodic[max_idx], x_anodic[max_idx]
                                else:
                                    peaks, _ = find_peaks(y_anodic)
                                    if len(peaks) > 0:
                                        b_idx = peaks[np.argmax(y_anodic[peaks])]
                                        i_pa, E_pa = y_anodic[b_idx], x_anodic[b_idx]
                                    else:
                                        max_idx = np.argmax(y_anodic)
                                        i_pa, E_pa = y_anodic[max_idx], x_anodic[max_idx]
                                    
                                j_pa = (i_pa * 1000) / electrode_area
                                if j_pa > 0: sg_cv_kinetics.append({"Curve": tr["name"], "v (mV/s)": tr["sr"], "log_v": np.log10(tr["sr"]), "E_p (V)": E_pa, "j_p (mA/cm²)": j_pa, "log_jp": np.log10(j_pa)})
                            
                            if tr.get("std") is not None and show_sd_shadow:
                                fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"]+tr["std"], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"]-tr["std"], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                            fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))
                            
                            if "LSV" in tr["tech"]:
                                cat_params, fit_data = extract_lsv_catalytic_parameters(tr["df"], electrode_area, e_rev)
                                if cat_params:
                                    sg_lsv_params.append({"Group": g_name, "Curve": tr["name"], **cat_params})
                                    sg_max_log_I = max(sg_max_log_I, fit_data["log_I_max"])
                                    fig_super_tafel.add_trace(go.Scatter(x=fit_data["log_I_full"], y=fit_data["E_full"], mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))
                                    if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                                        min_x, max_x = np.min(fit_data["log_I_fit"]), np.max(fit_data["log_I_fit"])
                                        span = max_x - min_x
                                        fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                                        fig_super_tafel.add_trace(go.Scatter(x=fit_x, y=fit_data["slope"]*fit_x + fit_data["intercept"], mode='lines', name=f"Fit: {cat_params['Tafel Slope (mV/dec)']:.1f} mV/dec", line=dict(color=c_color, width=2, dash='dot')))
                                    
                                    if tr.get("std") is not None and show_sd_shadow:
                                        j_dens, j_std = (tr["y"] * 1000) / electrode_area, (tr["std"] * 1000) / electrode_area
                                        fig_super_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens+j_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                        fig_super_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens-j_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                                    fig_super_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=fit_data["j_dens"], mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))

                # --- RENDER EIS ---
                if is_sg_eis:
                    fig_super.update_layout(title="Nyquist Plot", xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)", height=500)
                    fig_super.update_yaxes(scaleanchor="x", scaleratio=1)
                    fig_super = apply_scientific_style(fig_super, scientific_style, lx, ly, lxa, lya)
                    st.plotly_chart(fig_super, use_container_width=True, config=dl_config)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_bode_mod.update_layout(title="Bode Plot (|Z|)", xaxis_title="Frequency f (Hz)", yaxis_title="|Z| (Ω)", xaxis_type="log", yaxis_type="log", height=450)
                        fig_bode_mod = apply_scientific_style(fig_bode_mod, scientific_style, lx, ly, lxa, lya)
                        st.plotly_chart(fig_bode_mod, use_container_width=True, config=dl_config)
                    with c2:
                        fig_bode_phase.update_layout(title="Bode Plot (Phase)", xaxis_title="Frequency f (Hz)", yaxis_title="-Phase (°)", xaxis_type="log", height=450)
                        fig_bode_phase = apply_scientific_style(fig_bode_phase, scientific_style, lx, ly, lxa, lya)
                        st.plotly_chart(fig_bode_phase, use_container_width=True, config=dl_config)
                        
                    if fit_eis_model_toggle and sg_eis_params:
                        st.markdown("#### ⚡ Equivalent Circuit Fit Results `[Rs-(CPE||(Rct+(RL||L)))]`")
                        st.dataframe(pd.DataFrame(sg_eis_params).style.format({"Rs (Ω)": "{:.2f}", "CPE-T (F s^(n-1))": "{:.2e}", "CPE-P (n)": "{:.3f}", "Rct (Ω)": "{:.2f}", "RL (Ω)": "{:.2f}", "L (H)": "{:.2e}", "R²": "{:.4f}", "χ²": "{:.2e}"}), use_container_width=True)

                # --- RENDER CV/LSV ---
                else:
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
                        
                        fig_b = go.Figure()
                        fig_b.add_trace(go.Scatter(x=df_cv["log_v"], y=df_cv["log_jp"], mode='markers', marker=dict(size=10, color='black'), name="Data points"))
                        fit_x = np.array([df_cv["log_v"].min() - 0.1, df_cv["log_v"].max() + 0.1])
                        fig_b.add_trace(go.Scatter(x=fit_x, y=slope * fit_x + intercept, mode='lines', line=dict(color='red', dash='dash'), name=f"Fit: b = {slope:.2f}"))
                        
                        fig_b.update_layout(xaxis_title="log₁₀(v) [mV/s]", yaxis_title="log₁₀(j_p) [mA cm⁻²]" if scientific_style else "log₁₀(j_p) [mA/cm²]", height=450)
                        fig_b = apply_scientific_style(fig_b, scientific_style, lx, ly, lxa, lya)
                        
                        c1, c2 = st.columns([1, 1])
                        with c1: st.plotly_chart(fig_b, use_container_width=True, config=dl_config)
                        with c2:
                            st.metric("b-value (Slope ± SE)", f"{slope:.4f} ± {std_err:.4f}", f"R² = {r_value**2:.4f}", delta_color="off")
                            st.info("💡 **b = 0.5**: Diffusion-controlled process. **b = 1.0**: Surface-controlled (capacitive) process.")
                            st.dataframe(df_cv[["Curve", "v (mV/s)", "E_p (V)", "j_p (mA/cm²)"]].style.format({"v (mV/s)": "{:.1f}", "E_p (V)": "{:.3f}", "j_p (mA/cm²)": "{:.4f}"}), use_container_width=True)
