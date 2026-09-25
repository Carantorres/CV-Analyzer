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
st.set_page_config(page_title="SPARK Analyzer", layout="wide")
st.title("⚡ SPARK")
st.markdown("### System for Potentiostat Analysis & Research Knowledge")

st.markdown("""
**A comprehensive tool for automated chemical and electrochemical data analysis.** 
Seamlessly process CV, LSV, CP (Galvanostatic), EIS files, and **Chemical Speciation Diagrams**. Features robust catalytic parameter extraction, noise-free potential window detection, and equivalent circuit fitting for impedance spectroscopy.
""")

with st.popover("📖 View Calculation Methods & Algorithms"):
    st.markdown("""
    **1. Physico-Chemical Corrections**
    *   **iR Drop Compensation:** Corrects for the uncompensated resistance ($R_u$) of the electrolyte.
    *   **RHE Scale Conversion:** Shifts the potential to a pH-independent thermodynamic scale.
    *   **Current Density Normalization:** Automatically scales raw current to geometric Current Density (mA/cm²).

    **2. ECSA & Capacitance Extraction (CV)**
    *   **$C_{dl}$ & Roughness Factor (RF):** Reads $\Delta j / 2$ at a user-defined non-faradaic potential across multiple scan rates. Regression is forced through the origin ($y = C_{dl} \cdot v$) for physical accuracy. $RF = C_{dl} / C_{dl,ref}$.
    *   **ECSA-Normalized Kinetics:** Automatically corrects peak current densities ($j_{p, ECSA}$) based on calculated RF.

    **3. Catalytic Parameter Extraction (LSV)**
    *   **Onset Potential ($E_{onset}$):** Point where $\vert{}I\vert{}$ reaches 5% of the absolute maximum current.
    *   **Robust Tafel Slope:** Computed using a dynamic Sliding Window algorithm, maximizing $R^2$.

    **4. Scan Rate Kinetics ($b$-value)**
    *   A linear regression of $\\log_{10}(j_{pa})$ vs $\\log_{10}(v)$ calculates the slope $b$.
    
    **5. Electrochemical Impedance Spectroscopy (EIS)**
    *   **Equivalent Circuit Fitting:** Applies Non-Linear Least Squares (CNLS) with **Modulus Weighting** ($\\sigma = \vert{}Z\vert{}$). Generates high-density synthetic curves.
    *   **Effective Capacitance ($C_{eff}$):** Converts CPE to true capacitance using $C_{eff} = Q^{1/n} \cdot R^{(1-n)/n}$.
    *   **Polarization Sensitivity:** Calculates $R_{ct}$ ratios to evaluate potential-dependent activity.
    
    **6. Chemical Speciation (Medusa)**
    *   Features a custom reverse-engineering algorithm that reads raw `.plt` vector files and maps screen coordinates back to chemical thermodynamic data (Log Conc / Fraction vs pH) to generate high-quality plots.
    """)

st.markdown("---")

# ============================================================
# SIDEBAR SETUP (ALL SETTINGS DECLARED FIRST)
# ============================================================
with st.sidebar:
    st.header("🎛️ Instrument Format")
    instrument = st.selectbox(
        "Select default parser / equipment:",
        [
            "Gamry 1010B (.DTA)", 
            "Biologic SP-50e (.mpt)", 
            "PalmSens PSTrace (.csv)", 
            "Hydra & Medusa Speciation (.plt / .txt)"
        ]
    )
    st.info("💡 Tip: PalmSens, CP and Medusa (.plt) files are auto-detected!")

    st.markdown("---")
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
    st.header("📏 ECSA & C_dl Parameters")
    c_dl_ref = st.number_input("Reference C_dl (mF/cm²)", min_value=0.001, value=0.040, step=0.001, format="%.3f")
    e_non_faradaic = st.number_input("Non-Faradaic E (V) for C_dl", value=0.25, step=0.05)
    manual_rf = st.number_input("Calculated RF for EIS (Optional)", value=1.0, step=0.1, help="If known from CV, enter here to compute Rct·A/RF in EIS.")

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
    st.header("✂️ EIS Frequency Cropping")
    crop_eis = st.toggle("Limit Frequency Range", value=False, help="Discard noisy data at very low or high frequencies before fitting (e.g. gas bubble noise at low Hz).")
    if crop_eis:
        c_fmin, c_fmax = st.columns(2)
        with c_fmin: eis_min_f = st.number_input("Min Freq (Hz)", value=0.05, format="%.3f")
        with c_fmax: eis_max_f = st.number_input("Max Freq (Hz)", value=100000.0, step=1000.0)
    else:
        eis_min_f, eis_max_f = 1e-9, 1e9

    st.markdown("---")
    st.header("✂️ Medusa Axis Cropping")
    crop_medusa = st.toggle("Limit X/Y Axes (Medusa Only)", value=False, help="Crop the pH and Log Concentration/Fraction axes.")
    if crop_medusa:
        c_mx, c_my = st.columns(2)
        with c_mx: 
            medusa_xmin = st.number_input("Min X (pH)", value=4.0, step=0.5)
            medusa_xmax = st.number_input("Max X (pH)", value=5.0, step=0.5)
        with c_my: 
            medusa_ymin = st.number_input("Min Y", value=-4.0, step=0.5)
            medusa_ymax = st.number_input("Max Y", value=-1.0, step=0.5)
    else:
        medusa_xmin, medusa_xmax, medusa_ymin, medusa_ymax = None, None, None, None

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

# Define dynamic labels globally after variables are loaded
i_axis_label = "Current, I (A)" if scientific_style else "I (A)"
j_axis_label = "I/mA·cm⁻²" if scientific_style else "Current Density j (mA/cm²)"

publication_palette = ['#000000', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628', '#F781BF'] + px.colors.qualitative.Alphabet
combined_palette = publication_palette
dl_config = {'toImageButtonOptions': {'format': 'png', 'filename': 'electrochem_plot', 'height': 720, 'width': 960, 'scale': 4}}


# ============================================================
# UTILITIES & MATH FUNCTIONS
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

def _to_float(x):
    if x is None: return None
    try: return float(str(x).replace(",", "."))
    except: return None

def mad_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 10: return float(np.std(x)) if len(x) else np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else float(np.std(x))

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
EIS_MODELS_LIST = [
    "Randles: Rs-(CPE||Rct) [Ma et al. 2022]",
    "Randles + Warburg: Rs-(CPE||(Rct+W)) [Metrohm/Generic]",
    "Two Time Constants: Rs-(CPE1||R1)-(CPE2||R2) [Guo et al. 2016]",
    "Parallel Adsorption: Rs-(CPE||(Rct||(RL+L))) [Classic UOR]",
    "Series Adsorption: Rs-(CPE||(Rct+(RL||L))) [Harrington-Conway]",
    "Adsorption Capacitance: Rs-(CPE1||(Rct+(CPE2||Rads))) [ACS Appl. Mater. 2026]",
    "Bilayer + Series Adsorption: Rs-(CPE1||R1)-(CPE2||(Rct+(RL||L))) [NiOOH/Ni UOR]",
    "Three Time Constants: Rs-(CPE1||R1)-(CPE2||R2)-(CPE3||R3) [Bilayer NiOOH]"
]

def fit_uor_eis(f, zr, zi, model_type, area_cm2, rf_val):
    y_data = np.hstack([zr, zi])
    Z_data = zr - 1j * zi
    abs_Z = np.abs(Z_data)
    sigma = np.hstack([abs_Z, abs_Z]) 
    
    Rs_fixed = np.min(zr)
    R_tot = np.abs(np.max(zr) - Rs_fixed)
    Rct_guess = R_tot
    
    if "Randles: Rs-(CPE||Rct)" in model_type:
        def obj(f_val, CPE_T, CPE_P, Rct):
            w = 2 * np.pi * f_val
            Z_CPE = 1.0 / (CPE_T * (1j * w)**CPE_P)
            Z_total = Rs_fixed + 1.0 / (1.0/Z_CPE + 1.0/Rct)
            return np.hstack([Z_total.real, -Z_total.imag])
        p0 = [1e-4, 0.8, Rct_guess]
        bounds = ([1e-9, 0.5, 0], [1.0, 1.0, 1e7])
        param_names = ["CPE-T", "CPE-P", "Rct (Ω)"]
        
    elif "Randles + Warburg" in model_type:
        def obj(f_val, CPE_T, CPE_P, Rct, W_R):
            w = 2 * np.pi * f_val
            Z_CPE = 1.0 / (CPE_T * (1j * w)**CPE_P)
            Z_W = W_R / (1j * w)**0.5
            Z_total = Rs_fixed + 1.0 / (1.0/Z_CPE + 1.0/(Rct + Z_W))
            return np.hstack([Z_total.real, -Z_total.imag])
        p0 = [1e-4, 0.8, Rct_guess, Rct_guess]
        bounds = ([1e-9, 0.5, 0, 0], [1.0, 1.0, 1e7, 1e8])
        param_names = ["CPE-T", "CPE-P", "Rct (Ω)", "W (Ω·s^-0.5)"]
        
    elif "Two Time Constants" in model_type:
        def obj(f_val, CPE1_T, CPE1_P, R1, CPE2_T, CPE2_P, R2):
            w = 2 * np.pi * f_val
            Z_CPE1 = 1.0 / (CPE1_T * (1j * w)**CPE1_P)
            Z_CPE2 = 1.0 / (CPE2_T * (1j * w)**CPE2_P)
            Z_1 = 1.0 / (1.0/Z_CPE1 + 1.0/R1)
            Z_2 = 1.0 / (1.0/Z_CPE2 + 1.0/R2)
            Z_total = Rs_fixed + Z_1 + Z_2
            return np.hstack([Z_total.real, -Z_total.imag])
        p0 = [1e-5, 0.8, R_tot*0.1, 1e-3, 0.8, R_tot*0.9]
        bounds = ([1e-9, 0.5, 0, 1e-9, 0.5, 0], [1.0, 1.0, 1e7, 1.0, 1.0, 1e7])
        param_names = ["CPE1-T", "CPE1-P", "R1 (Ω)", "CPE2-T", "CPE2-P", "R2 (Ω)"]
        
    elif "Parallel Adsorption" in model_type:
        def obj(f_val, CPE_T, CPE_P, Rct, RL, L):
            w = 2 * np.pi * f_val
            Z_CPE = 1.0 / (CPE_T * (1j * w)**CPE_P)
            Z_ind = RL + 1j * w * L
            Z_faradaic = 1.0 / (1.0/Rct + 1.0/Z_ind)
            Z_total = Rs_fixed + 1.0 / (1.0/Z_CPE + 1.0/Z_faradaic)
            return np.hstack([Z_total.real, -Z_total.imag])
        p0 = [1e-4, 0.8, R_tot*1.5, R_tot*0.5, 1000]
        bounds = ([1e-9, 0.5, 0, 0, 1e-5], [1.0, 1.0, 1e7, 1e7, 1e8])
        param_names = ["CPE-T", "CPE-P", "Rct (Ω)", "RL (Ω)", "L (H)"]
        
    elif "Series Adsorption" in model_type:
        def obj(f_val, CPE_T, CPE_P, Rct, RL, L):
            w = 2 * np.pi * f_val
            Z_CPE = 1.0 / (CPE_T * (1j * w)**CPE_P)
            Z_L = 1j * w * L
            Z_ind = 1.0 / (1.0/RL + 1.0/Z_L)
            Z_faradaic = Rct + Z_ind
            Z_total = Rs_fixed + 1.0 / (1.0/Z_CPE + 1.0/Z_faradaic)
            return np.hstack([Z_total.real, -Z_total.imag])
        p0 = [1e-4, 0.8, R_tot, R_tot*0.5, 5000]
        bounds = ([1e-9, 0.5, 0, -1e7, 1e-5], [1.0, 1.0, 1e7, 1e7, 1e8])
        param_names = ["CPE-T", "CPE-P", "Rct (Ω)", "RL (Ω)", "L (H)"]
        
    elif "Adsorption Capacitance" in model_type:
        def obj(f_val, CPE1_T, CPE1_P, Rct, CPE2_T, CPE2_P, Rads):
            w = 2 * np.pi * f_val
            Z_CPE1 = 1.0 / (CPE1_T * (1j * w)**CPE1_P)
            Z_CPE2 = 1.0 / (CPE2_T * (1j * w)**CPE2_P)
            Z_ads = 1.0 / (1.0/Z_CPE2 + 1.0/Rads)
            Z_faradaic = Rct + Z_ads
            Z_total = Rs_fixed + 1.0 / (1.0/Z_CPE1 + 1.0/Z_faradaic)
            return np.hstack([Z_total.real, -Z_total.imag])
        p0 = [1e-5, 0.8, R_tot*0.5, 1e-3, 0.9, R_tot*0.5]
        bounds = ([1e-9, 0.5, 0, 1e-9, 0.5, 0], [1.0, 1.0, 1e7, 1.0, 1.0, 1e7])
        param_names = ["CPE1-T", "CPE1-P", "Rct (Ω)", "CPE2-T", "CPE2-P", "Rads (Ω)"]
        
    elif "Bilayer + Series Adsorption" in model_type:
        def obj(f_val, CPE1_T, CPE1_P, R1, CPE2_T, CPE2_P, Rct, RL, L):
            w = 2 * np.pi * f_val
            Z_CPE1 = 1.0 / (CPE1_T * (1j * w)**CPE1_P)
            Z_1 = 1.0 / (1.0/Z_CPE1 + 1.0/R1)
            Z_CPE2 = 1.0 / (CPE2_T * (1j * w)**CPE2_P)
            Z_L = 1j * w * L
            Z_ind = 1.0 / (1.0/RL + 1.0/Z_L)
            Z_faradaic = Rct + Z_ind
            Z_2 = 1.0 / (1.0/Z_CPE2 + 1.0/Z_faradaic)
            Z_total = Rs_fixed + Z_1 + Z_2
            return np.hstack([Z_total.real, -Z_total.imag])
        p0 = [1e-5, 0.8, R_tot*0.1, 1e-4, 0.8, R_tot*0.8, R_tot*0.3, 10000]
        bounds = ([1e-9, 0.5, 0, 1e-9, 0.5, 0, -1e7, 1e-5], [1.0, 1.0, 1e7, 1.0, 1.0, 1e7, 1e7, 1e8])
        param_names = ["CPE1-T", "CPE1-P", "R1 (Ω)", "CPE2-T", "CPE2-P", "Rct (Ω)", "RL (Ω)", "L (H)"]

    elif "Three Time Constants" in model_type:
        def obj(f_val, CPE1_T, CPE1_P, R1, CPE2_T, CPE2_P, R2, CPE3_T, CPE3_P, R3):
            w = 2 * np.pi * f_val
            Z_CPE1 = 1.0 / (CPE1_T * (1j * w)**CPE1_P)
            Z_1 = 1.0 / (1.0/Z_CPE1 + 1.0/R1)
            Z_CPE2 = 1.0 / (CPE2_T * (1j * w)**CPE2_P)
            Z_2 = 1.0 / (1.0/Z_CPE2 + 1.0/R2)
            Z_CPE3 = 1.0 / (CPE3_T * (1j * w)**CPE3_P)
            Z_3 = 1.0 / (1.0/Z_CPE3 + 1.0/R3)
            Z_total = Rs_fixed + Z_1 + Z_2 + Z_3
            return np.hstack([Z_total.real, -Z_total.imag])
        p0 = [1e-5, 0.8, R_tot*0.05, 1e-4, 0.8, R_tot*0.35, 1e-3, 0.8, R_tot*0.6]
        bounds = ([1e-9, 0.5, 0, 1e-9, 0.5, 0, 1e-9, 0.5, 0], [1.0, 1.0, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0, 1e7])
        param_names = ["CPE1-T", "CPE1-P", "R1 (Ω)", "CPE2-T", "CPE2-P", "R2 (Ω)", "CPE3-T", "CPE3-P", "R3 (Ω)"]
        
    try:
        popt, pcov = curve_fit(obj, f, y_data, p0=p0, bounds=bounds, sigma=sigma, maxfev=100000)
        
        Z_fit_raw = obj(f, *popt)
        N = len(f)
        Z_fit_complex = Z_fit_raw[:N] - 1j * Z_fit_raw[N:]
        
        SStot = np.sum((y_data - np.mean(y_data))**2)
        SSres = np.sum((y_data - Z_fit_raw)**2)
        R2 = 1 - (SSres / SStot)
        chi2 = np.sum((np.abs(Z_data - Z_fit_complex)**2) / (abs_Z**2)) / (len(f) - len(popt))
        
        f_sim = np.logspace(np.log10(np.max(f)), np.log10(np.min(f)), 200)
        Z_sim_raw = obj(f_sim, *popt)
        Z_sim_real = Z_sim_raw[:200]
        Z_sim_imag = Z_sim_raw[200:]
        
        results = {"Rs (Ω)": Rs_fixed}
        for i, name in enumerate(param_names):
            results[name] = popt[i]
        results["R²"] = R2
        results["χ²"] = chi2
        
        # Calculate Effective Capacitance
        def calc_ceff(Q, n, R):
            if R <= 0 or Q <= 0 or n <= 0: return np.nan
            return (Q**(1/n)) * (R**((1-n)/n))

        if "CPE-T" in results and "Rct (Ω)" in results:
            results["C_eff (F)"] = calc_ceff(results["CPE-T"], results["CPE-P"], results["Rct (Ω)"])
        if "CPE1-T" in results and "R1 (Ω)" in results:
            results["C_eff1 (F)"] = calc_ceff(results["CPE1-T"], results["CPE1-P"], results["R1 (Ω)"])
        if "CPE2-T" in results:
            r_tgt = results.get("R2 (Ω)", results.get("Rct (Ω)", None))
            if r_tgt is not None:
                results["C_eff2 (F)"] = calc_ceff(results["CPE2-T"], results["CPE2-P"], r_tgt)
        if "CPE3-T" in results and "R3 (Ω)" in results:
            results["C_eff3 (F)"] = calc_ceff(results["CPE3-T"], results["CPE3-P"], results["R3 (Ω)"])
            
        rct_val = results.get("Rct (Ω)", results.get("R2 (Ω)", None))
        if rct_val is not None:
            results["Rct·A (Ω·cm²)"] = rct_val * area_cm2
            if rf_val > 0:
                results["Rct·A/RF (Ω·cm²_ECSA)"] = (rct_val * area_cm2) / rf_val
            
        return results, f_sim, Z_sim_real, Z_sim_imag
    except Exception as e:
        return None, None, None, None

def extract_limits_from_data(df: pd.DataFrame, technique: str) -> Tuple[float, float, float]:
    if len(df) == 0: return None, None, None
    Ecol = "Vf" if "Vf" in df.columns else ("Vu" if "Vu" in df.columns else (df.columns[0] if len(df.columns)>0 else None))
    if Ecol is None or Ecol not in df.columns: return None, None, None
    
    col_data = df[Ecol]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
        
    v_data = pd.to_numeric(col_data, errors='coerce').dropna().values
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
    if "x" not in df_curve.columns or "y" not in df_curve.columns:
        return {"N_points": 0, "noisy_intervals_E": [], "E_cut_cathodic_V": None, "recommended_noise_safe_V": None, "recommended_reduction_only_V": None}
    
    df = df_curve[["x", "y"]].copy()
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
            font=dict(family="Arial, sans-serif", size=20, color="black"),
            margin=dict(l=80, r=40, t=40, b=60)
        )
        fig.update_xaxes(
            showgrid=False, showline=True, linecolor='black', linewidth=2, 
            mirror="all", ticks='inside', tickcolor='black', tickwidth=2, ticklen=8, 
            title_font=dict(size=24, family="Arial, sans-serif", color="black"), 
            tickfont=dict(size=18, family="Arial, sans-serif", color="black"), 
            zeroline=False
        )
        fig.update_yaxes(
            showgrid=False, showline=True, linecolor='black', linewidth=2, 
            mirror="all", ticks='inside', tickcolor='black', tickwidth=2, ticklen=8, 
            title_font=dict(size=24, family="Arial, sans-serif", color="black"), 
            tickfont=dict(size=18, family="Arial, sans-serif", color="black"), 
            zeroline=False
        )
    fig.update_layout(
        legend=dict(
            x=lx, y=ly, xanchor=lxa, yanchor=lya, 
            bgcolor='rgba(255, 255, 255, 0.9)', bordercolor='black', 
            borderwidth=1 if is_scientific else 0, 
            font=dict(size=18, color="black" if is_scientific else None)
        )
    )
    return fig

# ============================================================
# PARSERS
# ============================================================
def parse_medusa_plt(raw: bytes) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    text = raw.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    
    meta = {"TECHNIQUE": "Chemical Speciation (Medusa)"}
    curves = []
    
    x_low, x_high, y_low, y_high = 0.0, 14.0, -10.0, 0.0
    
    for line in lines[:50]:
        if "X/Y low and high:" in line:
            parts = line.split(":")[-1].split()
            if len(parts) >= 4:
                x_low, x_high, y_low, y_high = map(float, parts[:4])
                
    parsing_curves = False
    current_curve = None
    current_data = []
    
    for line in lines:
        line = line.strip()
        if "-- CURVES --" in line:
            parsing_curves = True
            continue
        if parsing_curves and "-- LABELS ON CURVES --" in line:
            break
            
        if parsing_curves:
            if line.startswith("5"):
                if current_curve and current_data:
                    df = pd.DataFrame(current_data, columns=["x_px", "y_px"])
                    df["x"] = x_low + (df["x_px"] - 263) / (1763 - 263) * (x_high - x_low)
                    df["y"] = y_low + (df["y_px"] - 140) / (1140 - 140) * (y_high - y_low)
                    curves.append((current_curve, df[["x", "y"]].copy()))
                
                parts = line.split(maxsplit=2)
                if len(parts) >= 3:
                    current_curve = parts[2].strip()
                    current_data = []
            else:
                m = re.match(r"^([01])\s*(\d+)\s+(\d+)$", line)
                if m:
                    current_data.append([float(m.group(2)), float(m.group(3))])
                    
    if current_curve and current_data:
        df = pd.DataFrame(current_data, columns=["x_px", "y_px"])
        df["x"] = x_low + (df["x_px"] - 263) / (1763 - 263) * (x_high - x_low)
        df["y"] = y_low + (df["y_px"] - 140) / (1140 - 140) * (y_high - y_low)
        curves.append((current_curve, df[["x", "y"]].copy()))
        
    return meta, curves

def parse_gamry_dta_multi_curve(raw: str) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    lines = raw.splitlines()
    meta: Dict[str, str] = {}
    first_curve_idx = None
    for i, line in enumerate(lines):
        if "CURVE" in line.upper() and "TABLE" in line.upper():
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
        if "CURVE" in line.upper() and "TABLE" in line.upper():
            m = re.search(r"(Z?CURVE)(\d*)", line, flags=re.IGNORECASE)
            curve_id = f"Curve {m.group(2) if m and m.group(2) else '1'}" 

            j, col_line_idx = i + 1, None
            while j < len(lines) and j < i + 60:
                s = lines[j].strip()
                parts = s.split("\t")
                if len(parts) >= 3 and any(c.upper() in ["PT", "V", "I", "FREQ", "ZREAL", "T", "TIME", "VF", "VU", "IM"] for c in parts):
                    col_line_idx = j; break
                j += 1

            if col_line_idx is None: 
                i += 1; continue

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
                if "CURVE" in s.upper() and "TABLE" in s.upper(): break
                parts = [p.strip() for p in lines[k].split("\t")]
                if len(parts) == 1: parts = [p.strip() for p in re.split(r"\s{2,}", s)]
                if parts and parts[0] == "": parts = parts[1:]
                if len(parts) >= len(cols): rows.append(parts[:len(cols)])
                k += 1

            df = pd.DataFrame(rows, columns=cols)
            for c in df.columns: df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")
            df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all").reset_index(drop=True)
            
            col_map = {}
            for c in df.columns:
                cu = c.upper()
                if cu in ["VF", "VU", "VM", "V"]: col_map[c] = "Vf"
                elif cu in ["IM", "I"]: col_map[c] = "Im"
                elif cu in ["FREQ", "FREQUENCY"]: col_map[c] = "Frequency"
                elif cu in ["ZREAL", "Z'"]: col_map[c] = "Z_real"
                elif cu in ["ZIMAG", "Z''"]: col_map[c] = "neg_Z_imag"
                elif cu in ["T", "TIME"]: col_map[c] = "Time"
            df = df.rename(columns=col_map)
            df = df.loc[:, ~df.columns.duplicated()]

            if "neg_Z_imag" in df.columns and any("Zimag" in c for c in cols):
                df["neg_Z_imag"] = -df["neg_Z_imag"]
                
            curves.append((curve_id, df))
            i = k
        else:
            i += 1
            
    method_clues = (meta.get("METHOD", "") + " " + meta.get("TAG", "") + " " + meta.get("TITLE", "") + " " + meta.get("EXPLAIN", "")).upper()
    if "IMP" in method_clues or "EIS" in method_clues or (curves and "Frequency" in curves[0][1].columns):
        meta["TECHNIQUE"] = "Electrochemical Impedance Spectroscopy (EIS)"
    elif "CP" in method_clues or "CHRONOP" in method_clues or (curves and "Time" in curves[0][1].columns and "Im" not in curves[0][1].columns):
        meta["TECHNIQUE"] = "Chronopotentiometry (CP)"
    elif "CV" in method_clues or "CYCLIC" in method_clues:
        meta["TECHNIQUE"] = "Cyclic Voltammetry (CV)"
    elif "LSV" in method_clues or "LINEAR" in method_clues:
        meta["TECHNIQUE"] = "Linear Sweep Voltammetry (LSV)"
        
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
                if "ewe" in cl or "potential" in cl or "v" == cl: cols[cols.index(c)] = c
                if "<i>" in cl or "current" in cl or "i" == cl: cols[cols.index(c)] = c

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
    for c in df.columns: df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all").reset_index(drop=True)

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "ewe" in cl or "potential" in cl or "voltage" in cl or "v" == cl: col_map[c] = "Vf"
        elif "<i>" in cl or "current" in cl or "i/ma" in cl or "i" == cl: col_map[c] = "Im"
        elif cl == "cycle number" or cl == "cycle": col_map[c] = "Cycle"
        elif "freq" in cl: col_map[c] = "Frequency"
        elif "re(z)" in cl or "z'" in cl: col_map[c] = "Z_real"
        elif "-im(z)" in cl or "-z''" in cl: col_map[c] = "neg_Z_imag"
        elif "im(z)" in cl or "z''" in cl: col_map[c] = "z_imag"
        elif "time" in cl or "t/s" in cl: col_map[c] = "Time"
    df = df.rename(columns=col_map)
    df = df.loc[:, ~df.columns.duplicated()]
    
    if "z_imag" in df.columns and "neg_Z_imag" not in df.columns:
        df["neg_Z_imag"] = -df["z_imag"]
        
    if "Im" in df.columns and df["Im"].abs().max() > 1: df["Im"] = df["Im"] / 1000

    curves = []
    if "Cycle" in df.columns:
        for cyc in sorted(df["Cycle"].dropna().unique()):
            df_cyc = df[df["Cycle"] == cyc].copy()
            if len(df_cyc) > 0: curves.append((f"Cycle {int(cyc) if float(cyc).is_integer() else cyc}", df_cyc.reset_index(drop=True)))
    else: curves.append(("Curve 1", df))
    
    if "Frequency" in df.columns and "Z_real" in df.columns: meta["TECHNIQUE"] = "Electrochemical Impedance Spectroscopy (EIS)"
    elif "Time" in df.columns and "Vf" in df.columns and "Im" not in df.columns: meta["TECHNIQUE"] = "Chronopotentiometry (CP)"
    elif "Vf" in df.columns and "Im" in df.columns: meta["TECHNIQUE"] = "Cyclic Voltammetry (CV)"
    
    return meta, curves

def parse_pstrace_csv(raw: bytes) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    text = None
    for enc in ['utf-8', 'utf-16', 'latin1']:
        try:
            text = raw.decode(enc)
            if "Fraction" in text or "fraction" in text or "pH" in text: break
            if "Linear Sweep" in text or "Cyclic Voltammetry" in text or "Impedance" in text or "Chronopotentiometry" in text or "CP" in text: break
        except: continue
    if not text: return {}, []
        
    lines = text.splitlines()
    meta = {}
    curves = []
    scan_names = []
    unit_row_idx = -1
    
    is_eis, is_cp, is_medusa = False, False, False
    
    if any("Fraction" in l or "fraction" in l for l in lines[:10]):
        meta["TECHNIQUE"] = "Chemical Speciation (Medusa)"
        is_medusa = True
        
        header_idx = -1
        for i, line in enumerate(lines[:20]):
            if "pH" in line or "Fraction" in line:
                header_idx = i
                break
        
        if header_idx != -1:
            delimiter = "\t" if "\t" in lines[header_idx] else ","
            df = pd.read_csv(io.StringIO(text), sep=delimiter, header=header_idx)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
            
            x_col = df.columns[0] 
            for c in df.columns[1:]:
                df_clean = df[[x_col, c]].dropna().copy()
                df_clean.columns = ["x", "y"]
                curves.append((c.strip(), df_clean))
        return meta, curves

    for line in lines[:20]:
        if "Linear Sweep" in line or "LSV" in line: meta["TECHNIQUE"] = "Linear Sweep Voltammetry (LSV)"
        elif "Cyclic Voltammetry" in line or "CV" in line: meta["TECHNIQUE"] = "Cyclic Voltammetry (CV)"
        elif "Impedance" in line or "EIS" in line: 
            meta["TECHNIQUE"] = "Electrochemical Impedance Spectroscopy (EIS)"
            is_eis = True
        elif "Chronopotentiometry" in line or "CP E vs t" in line:
            meta["TECHNIQUE"] = "Chronopotentiometry (CP)"
            is_cp = True
            
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
        
    if is_cp:
        for i, line in enumerate(lines[:50]):
            cols = [p.strip() for p in line.split(",")]
            if "s" in cols and "V" in cols:
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
            if "s" in df.columns: df_clean["Time"] = df["s"]
            if "V" in df.columns: df_clean["Vf"] = df["V"]
            
            if "Time" in df_clean.columns and "Vf" in df_clean.columns:
                df_clean = df_clean.dropna()
                curves.append(("CP Data", df_clean))
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

def convert_df_to_excel(curves_list: List[Tuple[str, pd.DataFrame]]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        seen_names = set()
        sheets_written = 0
        for cid, df in curves_list:
            if "Z_real" in df.columns and "neg_Z_imag" in df.columns:
                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Z_real", "neg_Z_imag"])
            elif "Time" in df.columns and "Vf" in df.columns:
                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Time", "Vf"])
            elif "x" in df.columns and "y" in df.columns:
                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
            else:
                Ecol = "Vf" if "Vf" in df.columns else ("Vu" if "Vu" in df.columns else None)
                if Ecol is None or "Im" not in df.columns: continue
                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[Ecol, "Im"])
            
            if len(clean_df) >= 2:
                safe_name = re.sub(r'[\\*?:/\[\]]', '_', cid)[:31].strip() or "Sheet"
                original_safe_name, counter = safe_name, 1
                while safe_name in seen_names:
                    suffix = f"_{counter}"
                    safe_name = f"{original_safe_name[:31-len(suffix)]}{suffix}"
                    counter += 1
                seen_names.add(safe_name)
                clean_df.to_excel(writer, index=False, sheet_name=safe_name)
                sheets_written += 1
                
        if sheets_written == 0:
            pd.DataFrame({"Message": ["No parseable data found"]}).to_excel(writer, index=False, sheet_name="Empty")
            
    return output.getvalue()


# ============================================================
# APP MAIN LOGIC & PLOTTING
# ============================================================
uploaded_files = st.file_uploader("Upload CV/LSV/CP/EIS or Medusa (.plt) files", type=["csv", "CSV", "DTA", "dta", "mpt", "MPT", "txt", "TXT", "plt", "PLT"], accept_multiple_files=True)

if uploaded_files:
    file_dict = {f.name: f for f in uploaded_files}
    display_names = set([f"⋮⋮ {name}" for name in file_dict.keys()])
    
    if 'file_groups' not in st.session_state: st.session_state.file_groups = [{"header": "📥 Unassigned Files", "items": []}, {"header": "📊 Group 1", "items": []}]
    for group in st.session_state.file_groups: group["items"] = [item for item in group["items"] if item in display_names]
    existing_items = set([item for group in st.session_state.file_groups for item in group["items"]])
    new_items = display_names - existing_items
    if new_items: st.session_state.file_groups[0]["items"].extend(list(new_items))

    with st.sidebar:
        st.markdown("---")
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
        is_group_lsv, is_group_eis, is_group_cp, is_group_medusa = False, False, False, False
        
        for item in group["items"]:
            fname = item.replace("⋮⋮ ", "")
            if fname not in file_dict: continue
            raw_bytes = file_dict[fname].getvalue()
            
            if fname.lower().endswith(".plt"):
                meta_sg, curves_comp = parse_medusa_plt(raw_bytes)
            else:
                meta_sg, curves_comp = parse_pstrace_csv(raw_bytes)
                if not curves_comp:
                    try:
                        raw_str = raw_bytes.decode('utf-8', errors='ignore')
                        if instrument.startswith("Gamry"): meta_sg, curves_comp = parse_gamry_dta_multi_curve(raw_str)
                        else: meta_sg, curves_comp = parse_biologic_mpt(raw_str)
                    except Exception as e:
                        pass
            
            tech_sg = meta_sg.get("TECHNIQUE", "")
            sr = manual_scan_rate if manual_scan_rate > 0.0 else _to_float(meta_sg.get("SCANRATE", meta_sg.get("dE/dt")))
            
            if "LSV" in tech_sg: is_group_lsv = True
            if "EIS" in tech_sg: is_group_eis = True
            if "CP" in tech_sg or "Chronopotentiometry" in tech_sg: is_group_cp = True
            if "Medusa" in tech_sg: is_group_medusa = True
                
            processed_curves = []
            for cid, df_comp in curves_comp:
                if "EIS" in tech_sg:
                    if "Z_real" in df_comp.columns and "neg_Z_imag" in df_comp.columns:
                        _df = df_comp.loc[:, ~df_comp.columns.duplicated()]
                        dd_comp = _df[["Z_real", "neg_Z_imag", "Frequency"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                        dd_comp.columns = ["x", "y", "f"]
                        if crop_eis:
                            dd_comp = dd_comp[(dd_comp["f"] >= eis_min_f) & (dd_comp["f"] <= eis_max_f)]
                        if len(dd_comp) >= 5: processed_curves.append((cid, dd_comp))
                elif "CP" in tech_sg or "Chronopotentiometry" in tech_sg:
                    if "Time" in df_comp.columns and "Vf" in df_comp.columns:
                        _df = df_comp.loc[:, ~df_comp.columns.duplicated()]
                        dd_comp = _df[["Time", "Vf"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                        if convert_to_rhe: dd_comp["Vf"] = dd_comp["Vf"] + e0_ref + (0.0591 * ph_val)
                        dd_comp.columns = ["x", "y"]
                        if len(dd_comp) >= 2: processed_curves.append((cid, dd_comp))
                elif "Medusa" in tech_sg:
                    processed_curves.append((cid, df_comp))
                else:
                    Ecol = "Vf" if "Vf" in df_comp.columns else ("Vu" if "Vu" in df_comp.columns else None)
                    if Ecol and "Im" in df_comp.columns:
                        _df = df_comp.loc[:, ~df_comp.columns.duplicated()]
                        dd_comp = _df[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
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
                    trace_name = f"{cid}" if "Medusa" in dat["tech"] else (f"{dat['fname']}" if len(dat['curves']) == 1 else f"{dat['fname']} ({cid})")
                    sr_val = get_sr_from_name(trace_name, dat['sr'] if dat['sr'] else 0.0)
                    group_plot_data.append({"x": dd_comp["x"].values, "y": dd_comp["y"].values, "f": dd_comp.get("f", pd.Series(dtype=float)).values, "std": None, "name": trace_name, "df": dd_comp, "tech": dat["tech"], "sr": sr_val})
                        
        prepared_group_data[group['header']] = {"traces": group_plot_data, "is_lsv": is_group_lsv, "is_eis": is_group_eis, "is_cp": is_group_cp, "is_medusa": is_group_medusa}

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
                is_sg_eis = any(prepared_group_data[g].get("is_eis", False) for g in selected_groups if g in prepared_group_data)
                is_sg_lsv = any(prepared_group_data[g].get("is_lsv", False) for g in selected_groups if g in prepared_group_data)
                is_sg_cp = any(prepared_group_data[g].get("is_cp", False) for g in selected_groups if g in prepared_group_data)
                is_sg_medusa = any(prepared_group_data[g].get("is_medusa", False) for g in selected_groups if g in prepared_group_data)
                
                avg_mode = st.radio("Super Group Mode:", ["Plot Individual Files", "Average ALL Files inside each Group"], key=f"sg_avg_{sg}", horizontal=True)
                
                if is_sg_eis: 
                    col_fit, col_model = st.columns([1, 2])
                    with col_fit:
                        fit_eis_model_toggle = st.toggle("🔋 Perform EIS Fit", value=False, key=f"fit_eis_{sg}")
                    with col_model:
                        sg_eis_models = {}
                        if fit_eis_model_toggle:
                            st.markdown("**Select Equivalent Circuit Model for each Group:**")
                            cols_models = st.columns(min(3, len(selected_groups)))
                            for i, g_name in enumerate(selected_groups):
                                clean_name = g_name.replace("📊 ", "")
                                sg_eis_models[g_name] = cols_models[i%3].selectbox(
                                    f"Model for {clean_name}:", 
                                    EIS_MODELS_LIST, 
                                    key=f"eis_model_sel_{sg}_{i}"
                                )
                else: 
                    fit_eis_model_toggle = False
                    sg_eis_models = {}

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
                y_axis_label_medusa = "Fraction"
                
                # Para rastrear los máximos y forzar cuadrados en EIS
                sg_min_z, sg_max_z = 0.0, 1.0 
                
                for g_idx, g_name in enumerate(selected_groups):
                    if g_name not in prepared_group_data: continue
                    g_data = prepared_group_data[g_name]
                    base_color = combined_palette[g_idx % len(combined_palette)]
                    
                    traces_to_plot = []
                    if avg_mode == "Average ALL Files inside each Group" and len(g_data["traces"]) > 0:
                        if is_sg_eis:
                            common_f, zr_mean, zi_mean, zr_std, zi_std = get_averaged_eis_curve([(tr["name"], tr["df"]) for tr in g_data["traces"]])
                            df_mean = pd.DataFrame({"x": zr_mean, "y": zi_mean, "f": common_f})
                            traces_to_plot = [{"x": zr_mean, "y": zi_mean, "f": common_f, "std": None, "name": sg_custom_labels[g_name], "df": df_mean, "tech": "EIS", "group": g_name}]
                        elif is_sg_cp:
                            E_mean, I_mean, I_std = get_averaged_curve([(tr["name"], tr["df"]) for tr in g_data["traces"]])
                            df_mean = pd.DataFrame({"x": E_mean, "y": I_mean})
                            traces_to_plot = [{"x": E_mean, "y": I_mean, "f": None, "std": I_std, "name": sg_custom_labels[g_name], "df": df_mean, "tech": "CP", "sr": 0.0, "group": g_name}]
                        elif is_sg_medusa:
                            st.warning("Averaging is not supported for Medusa diagrams. Plotting individually.")
                            for tr in g_data["traces"]:
                                tr_copy = tr.copy()
                                tr_copy["group"] = g_name
                                traces_to_plot.append(tr_copy)
                        else:
                            E_mean, I_mean, I_std = get_averaged_curve([(tr["name"], tr["df"]) for tr in g_data["traces"]])
                            df_mean = pd.DataFrame({"x": E_mean, "y": I_mean})
                            fallback_sr = np.mean([t["sr"] for t in g_data["traces"] if t["sr"]>0]) if any(t["sr"]>0 for t in g_data["traces"]) else 0.0
                            sr_val = get_sr_from_name(sg_custom_labels[g_name], fallback_sr)
                            traces_to_plot = [{"x": E_mean, "y": I_mean, "f": None, "std": I_std, "name": sg_custom_labels[g_name], "df": df_mean, "tech": "LSV" if is_sg_lsv else "CV", "sr": sr_val, "group": g_name}]
                    else:
                        for idx_tr, tr in enumerate(g_data["traces"]):
                            if is_sg_medusa:
                                final_name = f"{tr['name']}"
                                c_color = combined_palette[idx_tr % len(combined_palette)]
                            else:
                                final_name = sg_custom_labels[g_name] if len(g_data["traces"]) == 1 else f"{sg_custom_labels[g_name]} - {tr['name']}"
                                c_color = base_color
                                
                            tr_copy = tr.copy()
                            tr_copy["name"] = final_name
                            tr_copy["group"] = g_name
                            tr_copy["color"] = c_color
                            traces_to_plot.append(tr_copy)

                    for tr in traces_to_plot:
                        c_color = tr.get("color", base_color)
                        
                        if is_sg_eis:
                            zr, zi, f_hz = tr["x"], tr["y"], tr["f"]
                            
                            sg_min_z = min(sg_min_z, np.min(zr), np.min(zi))
                            sg_max_z = max(sg_max_z, np.max(zr), np.max(zi))
                            
                            fig_super.add_trace(go.Scatter(x=zr, y=zi, mode='markers', name=tr["name"], marker=dict(color=c_color, size=6)))
                            
                            z_mod = np.sqrt(zr**2 + zi**2)
                            phase = np.degrees(np.arctan2(zi, zr)) 
                            
                            fig_bode_mod.add_trace(go.Scatter(x=f_hz, y=z_mod, mode='markers', name=tr["name"], marker=dict(color=c_color, size=6)))
                            fig_bode_phase.add_trace(go.Scatter(x=f_hz, y=phase, mode='markers', name=tr["name"], marker=dict(color=c_color, size=6)))
                            
                            if fit_eis_model_toggle:
                                selected_model = sg_eis_models[tr["group"]]
                                results_dict, f_sim, zr_sim, zi_sim = fit_uor_eis(f_hz, zr, zi, selected_model, electrode_area, manual_rf)
                                if results_dict is not None:
                                    results_dict["Curve"] = tr["name"]
                                    results_dict["Model"] = selected_model.split(" [")[0] 
                                    sg_eis_params.append(results_dict)
                                    
                                    sg_min_z = min(sg_min_z, np.min(zr_sim), np.min(zi_sim))
                                    sg_max_z = max(sg_max_z, np.max(zr_sim), np.max(zi_sim))
                                    
                                    fig_super.add_trace(go.Scatter(x=zr_sim, y=zi_sim, mode='lines', name=f"{tr['name']} Model", line=dict(color=c_color, width=2), showlegend=True, hoverinfo='skip'))
                                    z_mod_sim = np.sqrt(zr_sim**2 + zi_sim**2)
                                    phase_sim = np.degrees(np.arctan2(zi_sim, zr_sim))
                                    fig_bode_mod.add_trace(go.Scatter(x=f_sim, y=z_mod_sim, mode='lines', name=f"{tr['name']} Model", line=dict(color=c_color, width=2), showlegend=True, hoverinfo='skip'))
                                    fig_bode_phase.add_trace(go.Scatter(x=f_sim, y=phase_sim, mode='lines', name=f"{tr['name']} Model", line=dict(color=c_color, width=2), showlegend=True, hoverinfo='skip'))
                        elif is_sg_cp:
                            if tr.get("std") is not None and show_sd_shadow:
                                fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"]+tr["std"], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"]-tr["std"], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                            fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))
                        elif is_sg_medusa:
                            fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))
                            if tr["y"].min() < -1.0: y_axis_label_medusa = "Log Concentration"
                        else:
                            x_arr = tr["x"]
                            y_arr = tr["y"]
                            idx_max_E = np.argmax(x_arr)
                            
                            try:
                                x_a, y_a = x_arr[:idx_max_E+1], y_arr[:idx_max_E+1]
                                x_c, y_c = x_arr[idx_max_E:], y_arr[idx_max_E:]
                                sort_a = np.argsort(x_a)
                                sort_c = np.argsort(x_c)
                                
                                if e_non_faradaic >= np.min(x_arr) and e_non_faradaic <= np.max(x_arr):
                                    i_a_val = np.interp(e_non_faradaic, x_a[sort_a], y_a[sort_a])
                                    i_c_val = np.interp(e_non_faradaic, x_c[sort_c], y_c[sort_c])
                                    j_a_val = i_a_val * 1000 / electrode_area
                                    j_c_val = i_c_val * 1000 / electrode_area
                                    delta_j_half = (j_a_val - j_c_val) / 2.0
                                else:
                                    delta_j_half = np.nan
                            except:
                                delta_j_half = np.nan

                            x_anodic = x_arr[:idx_max_E+1] if idx_max_E > 0 else x_arr
                            y_anodic = y_arr[:idx_max_E+1] if idx_max_E > 0 else y_arr
                            
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
                            if j_pa > 0 and tr.get("sr") and tr["sr"] > 0 and not is_sg_lsv: 
                                sg_cv_kinetics.append({
                                    "Curve": tr["name"], "v (mV/s)": tr["sr"], "log_v": np.log10(tr["sr"]), 
                                    "E_p (V)": E_pa, "j_p (mA/cm²)": j_pa, "log_jp": np.log10(j_pa),
                                    "delta_j_half": delta_j_half
                                })
                            
                            j_y = y_arr * 1000 / electrode_area
                            if tr.get("std") is not None and show_sd_shadow:
                                j_std = tr["std"] * 1000 / electrode_area
                                fig_super.add_trace(go.Scatter(x=x_arr, y=j_y+j_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                fig_super.add_trace(go.Scatter(x=x_arr, y=j_y-j_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                            fig_super.add_trace(go.Scatter(x=x_arr, y=j_y, mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))
                            
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

                if is_sg_eis:
                    axis_min = sg_min_z * 1.05 if sg_min_z < 0 else 0.0
                    axis_max = sg_max_z * 1.05
                    
                    fig_super.update_layout(title="Nyquist Plot", xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)", height=650)
                    fig_super.update_xaxes(range=[axis_min, axis_max], constrain="domain")
                    fig_super.update_yaxes(range=[axis_min, axis_max], scaleanchor="x", scaleratio=1, constrain="domain")
                    
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
                        st.markdown(f"#### ⚡ Equivalent Circuit Fit Results")
                        df_eis = pd.DataFrame(sg_eis_params)
                        
                        if "Rct (Ω)" in df_eis.columns:
                            df_eis["Rct Ratio (Max/Current)"] = df_eis["Rct (Ω)"].max() / df_eis["Rct (Ω)"]
                        elif "R2 (Ω)" in df_eis.columns:
                            df_eis["Rct Ratio (Max/Current)"] = df_eis["R2 (Ω)"].max() / df_eis["R2 (Ω)"]
                            
                        all_possible_cols = ["Curve", "Model", "Rs (Ω)", "CPE-T", "CPE-P", "Rct (Ω)", "W (Ω·s^-0.5)", "RL (Ω)", "L (H)", "Ls (H)", "CPE1-T", "CPE1-P", "R1 (Ω)", "CPE2-T", "CPE2-P", "R2 (Ω)", "CPE3-T", "CPE3-P", "R3 (Ω)", "Rads (Ω)", "Rct·A (Ω·cm²)", "Rct·A/RF (Ω·cm²_ECSA)", "C_eff (F)", "C_eff1 (F)", "C_eff2 (F)", "C_eff3 (F)", "Rct Ratio (Max/Current)", "R²", "χ²"]
                        cols = [c for c in all_possible_cols if c in df_eis.columns]
                        df_eis = df_eis[cols]
                        
                        format_dict = {
                            "Rs (Ω)": "{:.2f}", "CPE-T": "{:.2e}", "CPE-P": "{:.3f}", 
                            "Rct (Ω)": "{:.2f}", "W (Ω·s^-0.5)": "{:.2e}", "RL (Ω)": "{:.2f}", "L (H)": "{:.2e}", "Ls (H)": "{:.2e}",
                            "CPE1-T": "{:.2e}", "CPE1-P": "{:.3f}", "R1 (Ω)": "{:.2f}", 
                            "CPE2-T": "{:.2e}", "CPE2-P": "{:.3f}", "R2 (Ω)": "{:.2f}", 
                            "CPE3-T": "{:.2e}", "CPE3-P": "{:.3f}", "R3 (Ω)": "{:.2f}", 
                            "Rads (Ω)": "{:.2f}", "Rct·A (Ω·cm²)": "{:.2f}", "Rct·A/RF (Ω·cm²_ECSA)": "{:.2f}", 
                            "C_eff (F)": "{:.2e}", "C_eff1 (F)": "{:.2e}", "C_eff2 (F)": "{:.2e}", "C_eff3 (F)": "{:.2e}", 
                            "Rct Ratio (Max/Current)": "{:.2f}", "R²": "{:.4f}", "χ²": "{:.2e}"
                        }
                        st.dataframe(df_eis.style.format({k:v for k,v in format_dict.items() if k in df_eis.columns}), use_container_width=True)
                elif is_sg_cp:
                    fig_super.update_layout(title="Galvanostatic Charge-Discharge" if not scientific_style else "", xaxis_title="Time t (s)", yaxis_title=x_axis_label, height=500)
                    fig_super = apply_scientific_style(fig_super, scientific_style, lx, ly, lxa, lya)
                    st.plotly_chart(fig_super, use_container_width=True, config=dl_config)
                elif is_sg_medusa:
                    fig_super.update_layout(title="Chemical Speciation" if not scientific_style else "", xaxis_title="pH", yaxis_title=y_axis_label_medusa, height=500)
                    if crop_medusa:
                        fig_super.update_xaxes(range=[medusa_xmin, medusa_xmax])
                        fig_super.update_yaxes(range=[medusa_ymin, medusa_ymax])
                    fig_super = apply_scientific_style(fig_super, scientific_style, lx, ly, lxa, lya)
                    st.plotly_chart(fig_super, use_container_width=True, config=dl_config)
                else:
                    fig_super.update_layout(title="", xaxis_title=x_axis_label, yaxis_title=j_axis_label, height=500)
                    fig_super = apply_scientific_style(fig_super, scientific_style, lx, ly, lxa, lya)
                    st.plotly_chart(fig_super, use_container_width=True, config=dl_config)
                    
                    if is_sg_lsv:
                        c1, c2 = st.columns(2)
                        with c1:
                            fig_super_jeta.update_layout(title="Catalytic Performance" if not scientific_style else "", xaxis_title="Overpotential η (mV)", yaxis_title=j_axis_label, height=500)
                            fig_super_jeta = apply_scientific_style(fig_super_jeta, scientific_style, lx, ly, lxa, lya)
                            st.plotly_chart(fig_super_jeta, use_container_width=True, config=dl_config)
                        with c2:
                            fig_super_tafel.update_layout(title="Tafel Plot" if not scientific_style else "", xaxis_title="log₁₀|I| (A)", yaxis_title=x_axis_label, xaxis=dict(range=[sg_max_log_I - 4.5, sg_max_log_I + 0.2]), height=500)
                            fig_super_tafel = apply_scientific_style(fig_super_tafel, scientific_style, lx, ly, lxa, lya)
                            st.plotly_chart(fig_super_tafel, use_container_width=True, config=dl_config)
                            
                        if sg_lsv_params:
                            st.markdown("#### 🧪 Catalytic Parameters (LSV)")
                            st.dataframe(pd.DataFrame(sg_lsv_params), use_container_width=True)
                            
                    elif len(sg_cv_kinetics) > 1:
                        st.markdown("#### 🔋 CV Kinetics & ECSA ($C_{dl}$ y $b$-value)")
                        df_cv = pd.DataFrame(sg_cv_kinetics).sort_values("v (mV/s)")
                        
                        df_cv["v (V/s)"] = df_cv["v (mV/s)"] / 1000.0
                        df_cv_clean = df_cv.dropna(subset=["delta_j_half", "v (V/s)"])
                        
                        c_dl = np.nan
                        rf = np.nan
                        fig_cdl = go.Figure()
                        
                        if len(df_cv_clean) > 1:
                            def fit_cdl(v, c): return c * v
                            try:
                                popt_cdl, _ = curve_fit(fit_cdl, df_cv_clean["v (V/s)"], df_cv_clean["delta_j_half"])
                                c_dl = popt_cdl[0]
                                
                                ss_res = np.sum((df_cv_clean["delta_j_half"] - c_dl * df_cv_clean["v (V/s)"])**2)
                                ss_tot = np.sum((df_cv_clean["delta_j_half"] - np.mean(df_cv_clean["delta_j_half"]))**2)
                                r2_cdl = max(0, 1 - (ss_res / ss_tot)) if ss_tot > 0 else 0
                                
                                rf = c_dl / c_dl_ref if c_dl_ref > 0 else np.nan
                                df_cv["j_p,ECSA (mA/cm²_ECSA)"] = df_cv["j_p (mA/cm²)"] / rf
                                
                                fig_cdl.add_trace(go.Scatter(x=df_cv_clean["v (V/s)"], y=df_cv_clean["delta_j_half"], mode='markers', marker=dict(size=10, color='blue'), name="Data"))
                                fit_v = np.array([0, df_cv_clean["v (V/s)"].max() * 1.1])
                                fig_cdl.add_trace(go.Scatter(x=fit_v, y=c_dl * fit_v, mode='lines', line=dict(color='red', dash='dash'), name=f"Fit (C_dl={c_dl:.2f} mF/cm²)"))
                            except: pass
                            
                            fig_cdl.update_layout(xaxis_title="Scan Rate v (V/s)", yaxis_title="Δj/2 (mA/cm²)", height=450)
                            fig_cdl = apply_scientific_style(fig_cdl, scientific_style, lx, ly, lxa, lya)
                        
                        slope, intercept, r_value, p_value, std_err = linregress(df_cv["log_v"], df_cv["log_jp"])
                        
                        fig_b = go.Figure()
                        fig_b.add_trace(go.Scatter(x=df_cv["log_v"], y=df_cv["log_jp"], mode='markers', marker=dict(size=10, color='black'), name="Data points"))
                        fit_x = np.array([df_cv["log_v"].min() - 0.1, df_cv["log_v"].max() + 0.1])
                        fig_b.add_trace(go.Scatter(x=fit_x, y=slope * fit_x + intercept, mode='lines', line=dict(color='red', dash='dash'), name=f"Fit: b = {slope:.2f}"))
                        
                        fig_b.update_layout(xaxis_title="log₁₀(v) [mV/s]", yaxis_title="log₁₀(j_p) [mA cm⁻²]" if scientific_style else "log₁₀(j_p) [mA/cm²]", height=450)
                        fig_b = apply_scientific_style(fig_b, scientific_style, lx, ly, lxa, lya)
                        
                        c1, c2 = st.columns([1, 1])
                        with c1: 
                            st.plotly_chart(fig_b, use_container_width=True, config=dl_config)
                            st.metric("b-value (Slope ± SE)", f"{slope:.4f} ± {std_err:.4f}", f"R² = {r_value**2:.4f}", delta_color="off")
                            st.info("💡 **b = 0.5**: Diffusion-controlled. **b = 1.0**: Surface-controlled.")
                        with c2: 
                            if len(df_cv_clean) > 1 and not np.isnan(c_dl):
                                st.plotly_chart(fig_cdl, use_container_width=True, config=dl_config)
                                st.metric("Double Layer Capacitance (C_dl)", f"{c_dl:.3f} mF/cm²", f"R² = {r2_cdl:.4f} (at {e_non_faradaic} V)", delta_color="off")
                                st.metric("Roughness Factor (RF)", f"{rf:.1f}", f"Ref = {c_dl_ref} mF/cm²", delta_color="off")

                        cols_to_show = ["Curve", "v (mV/s)", "E_p (V)", "j_p (mA/cm²)"]
                        if not np.isnan(rf): cols_to_show.append("j_p,ECSA (mA/cm²_ECSA)")
                        st.dataframe(df_cv[cols_to_show].style.format({c: "{:.3f}" for c in cols_to_show if c != "Curve"}), use_container_width=True)

    # --- BOTTOM AREA: INDIVIDUAL ANALYSIS ---
    st.markdown("---")
    st.header("📄 Individual Analysis")

    actual_file_order = [item.replace("⋮⋮ ", "") for group in st.session_state.file_groups for item in group["items"]]

    for file_name in actual_file_order:
        if file_name not in file_dict: continue 
            
        file = file_dict[file_name]
        raw_bytes = file.getvalue()
        
        if file_name.lower().endswith(".plt"):
            meta, curves = parse_medusa_plt(raw_bytes)
        else:
            meta, curves = parse_pstrace_csv(raw_bytes)
            if not curves:
                try:
                    raw_str = raw_bytes.decode('utf-8', errors='ignore')
                    if instrument.startswith("Gamry"): meta, curves = parse_gamry_dta_multi_curve(raw_str)
                    else: meta, curves = parse_biologic_mpt(raw_str)
                except Exception as e:
                    pass
            
        technique = meta.get("TECHNIQUE", "Unknown Technique")
        sr = manual_scan_rate if manual_scan_rate > 0.0 else _to_float(meta.get("SCANRATE", meta.get("dE/dt")))

        if not curves: continue

        vinit, vlim1, vlim2 = None, None, None
        is_eis = "EIS" in technique
        is_cp = "CP" in technique or "Chronopotentiometry" in technique
        is_medusa = "Medusa" in technique
        if not is_eis and not is_cp and not is_medusa: vinit, vlim1, vlim2 = extract_limits_from_data(curves[0][1], technique)

        st.markdown(f"### {file.name}")
        col_title, col_btn = st.columns([4, 1])
        with col_title: st.markdown(f"🔬 **Technique Detected:** `{technique}`")
        with col_btn: st.download_button("📥 Export to Excel", convert_df_to_excel(curves), f"{file.name.split('.')[0]}_Data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_{file.name}")
        
        if is_eis:
            st.info("💡 **Electrochemical Impedance Spectroscopy (EIS) Data:** Generating Nyquist Plot (-Z'' vs Z')")
        elif is_cp:
            st.info("💡 **Chronopotentiometry (CP) Data:** Generating Galvanostatic Plot (E vs t)")
        elif is_medusa:
            st.info("💡 **Chemical Speciation Data:** Generating Distribution Diagram")
        else:
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
        y_axis_label_medusa_ind = "Fraction"
        
        for i, (cid, dfi) in enumerate(curves):
            if is_eis:
                if "Z_real" in dfi.columns and "neg_Z_imag" in dfi.columns:
                    _dfi = dfi.loc[:, ~dfi.columns.duplicated()]
                    dd = _dfi[["Z_real", "neg_Z_imag", "Frequency"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                    dd.columns = ["x", "y", "f"]
                    if crop_eis:
                        dd = dd[(dd["f"] >= eis_min_f) & (dd["f"] <= eis_max_f)]
                    if len(dd) >= 5: processed_curves.append((cid, dd))
            elif is_cp:
                if "Time" in dfi.columns and "Vf" in dfi.columns:
                    _dfi = dfi.loc[:, ~dfi.columns.duplicated()]
                    dd = _dfi[["Time", "Vf"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                    if convert_to_rhe: dd["Vf"] = dd["Vf"] + e0_ref + (0.0591 * ph_val)
                    dd.columns = ["x", "y"]
                    if len(dd) >= 2: processed_curves.append((cid, dd))
            elif is_medusa:
                if dfi["y"].min() < -1.0: y_axis_label_medusa_ind = "Log Concentration"
                processed_curves.append((cid, dfi))
            else:
                Ecol = "Vf" if "Vf" in dfi.columns else ("Vu" if "Vu" in dfi.columns else None)
                if Ecol is None or "Im" not in dfi.columns: continue
                _dfi = dfi.loc[:, ~dfi.columns.duplicated()]
                dd = _dfi[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                if apply_ir: dd[Ecol] = dd[Ecol] - dd["Im"] * ru_ohms * (comp_percent / 100.0)
                if convert_to_rhe: dd[Ecol] = dd[Ecol] + e0_ref + (0.0591 * ph_val)
                dd.columns = ["x", "y"]
                if len(dd) >= 10: processed_curves.append((cid, dd))

        if not processed_curves: continue

        fig, fig_tafel, fig_jeta = go.Figure(), go.Figure(), go.Figure()
        results_list, lsv_cat_list, max_log_I_ind = [], [], -10
        
        # Para rastrear los máximos de EIS individual
        ind_min_z, ind_max_z = 0.0, 1.0

        avg_cycles = st.toggle(f"🌟 Average {len(processed_curves)} Cycles/Scans", key=f"avg_{file.name}") if len(processed_curves) > 1 and not is_medusa else False

        if avg_cycles:
            if is_eis:
                common_f, zr_mean, zi_mean, zr_std, zi_std = get_averaged_eis_curve(processed_curves)
                ind_min_z = min(ind_min_z, np.min(zr_mean), np.min(zi_mean))
                ind_max_z = max(ind_max_z, np.max(zr_mean), np.max(zi_mean))
                
                mean_color = combined_palette[0]
                fig.add_trace(go.Scatter(x=zr_mean, y=zi_mean, mode='markers', name='Average', marker=dict(color=mean_color, size=6)))
            elif is_cp:
                E_mean, I_mean, I_std = get_averaged_curve(processed_curves)
                mean_color = combined_palette[0]
                if show_sd_shadow:
                    fig.add_trace(go.Scatter(x=E_mean, y=I_mean + I_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=E_mean, y=I_mean - I_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(mean_color, 0.2), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=E_mean, y=I_mean, mode='lines', name='Average', line=dict(color=mean_color, width=2.5)))
            else:
                E_mean, I_mean, I_std = get_averaged_curve(processed_curves)
                mean_color = combined_palette[0]
                
                j_mean = I_mean * 1000 / electrode_area
                j_std = I_std * 1000 / electrode_area
                
                if show_sd_shadow:
                    fig.add_trace(go.Scatter(x=E_mean, y=j_mean + j_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=E_mean, y=j_mean - j_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(mean_color, 0.2), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=E_mean, y=j_mean, mode='lines', name='Average', line=dict(color=mean_color, width=2.5)))
                
                df_mean = pd.DataFrame({"x": E_mean, "y": I_mean})
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
            for i, (cid, dd) in enumerate(processed_curves):
                line_color = combined_palette[i % len(combined_palette)]
                if is_eis:
                    ind_min_z = min(ind_min_z, np.min(dd["x"]), np.min(dd["y"]))
                    ind_max_z = max(ind_max_z, np.max(dd["x"]), np.max(dd["y"]))
                    
                    fig.add_trace(go.Scatter(x=dd["x"], y=dd["y"], mode='markers', name=cid, marker=dict(color=line_color, size=6)))
                elif is_cp:
                    fig.add_trace(go.Scatter(x=dd["x"], y=dd["y"], mode='lines', name=cid, line=dict(color=line_color, width=2)))
                elif is_medusa:
                    fig.add_trace(go.Scatter(x=dd["x"], y=dd["y"], mode='lines', name=cid, line=dict(color=line_color, width=2.5)))
                else:
                    j_y = dd["y"] * 1000 / electrode_area
                    fig.add_trace(go.Scatter(x=dd["x"], y=j_y, mode='lines', name=cid, line=dict(color=line_color, width=2)))
                    
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

        if is_eis:
            x_title = "Z' (Ω)"
            y_title = "-Z'' (Ω)"
            # Definir límites cuadrados
            axis_min = ind_min_z * 1.05 if ind_min_z < 0 else 0.0
            axis_max = ind_max_z * 1.05
        elif is_cp:
            x_title = "Time t (s)"
            y_title = x_axis_label
        elif is_medusa:
            x_title = "pH"
            y_title = y_axis_label_medusa_ind
        else:
            x_title = x_axis_label
            y_title = j_axis_label
            
        title_txt = "Nyquist Plot" if is_eis else "Galvanostatic Charge-Discharge" if is_cp else "Chemical Speciation" if is_medusa else "Raw Data" if not scientific_style else ""
        
        # Aumentamos la altura de los diagramas de EIS a 650 para que el cuadrado se vea bien proporcionado
        fig.update_layout(title=title_txt, xaxis_title=x_title, yaxis_title=y_title, height=650 if is_eis else 500)
        
        if is_eis: 
            fig.update_xaxes(range=[axis_min, axis_max], constrain="domain")
            fig.update_yaxes(range=[axis_min, axis_max], scaleanchor="x", scaleratio=1, constrain="domain")
            
        if is_medusa and crop_medusa:
            fig.update_xaxes(range=[medusa_xmin, medusa_xmax])
            fig.update_yaxes(range=[medusa_ymin, medusa_ymax])
            
        fig = apply_scientific_style(fig, scientific_style, lx, ly, lxa, lya)
        st.plotly_chart(fig, use_container_width=True, config=dl_config)
        
        if "LSV" in technique and lsv_cat_list and not is_eis and not is_cp and not is_medusa:
            c1, c2 = st.columns(2)
            with c1:
                fig_jeta.update_layout(title="Catalytic Performance" if not scientific_style else "", xaxis_title="Overpotential η (mV)", yaxis_title=j_axis_label, height=500)
                fig_jeta = apply_scientific_style(fig_jeta, scientific_style, lx, ly, lxa, lya)
                st.plotly_chart(fig_jeta, use_container_width=True, config=dl_config)
            with c2:
                fig_tafel.update_layout(title="Tafel Plot" if not scientific_style else "", xaxis_title="log₁₀|I| (A)", yaxis_title=x_axis_label, xaxis=dict(range=[max_log_I_ind - 4.5, max_log_I_ind + 0.2]), height=500)
                fig_tafel = apply_scientific_style(fig_tafel, scientific_style, lx, ly, lxa, lya)
                st.plotly_chart(fig_tafel, use_container_width=True, config=dl_config)
        
        if results_list and not is_eis and not is_cp and not is_medusa: st.write("**Recommended Operating Ranges:**"); st.dataframe(pd.DataFrame(results_list), use_container_width=True)
        if lsv_cat_list and not is_eis and not is_cp and not is_medusa: st.write("**🧪 Catalytic Parameters:**"); st.dataframe(pd.DataFrame(lsv_cat_list), use_container_width=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
