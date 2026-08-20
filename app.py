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
st.title("📊 Gamry & Biologic CV Analyzer")
st.markdown("Upload your **Gamry (.DTA)** or **Biologic (.mpt)** files to visualize potential sweeps and calculate safe operating ranges.")

# ============================================================
# INSTRUMENT SELECTION
# ============================================================
instrument = st.selectbox(
    "Select instrument format:",
    ["Gamry 1010B (.DTA)", "Biologic SP-50e (.mpt)"]
)

# ============================================================
# MATH & ANALYTICS
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
# CATALYTIC PARAMETERS & TAFEL FIT
# ============================================================
def extract_lsv_catalytic_parameters(df_curve: pd.DataFrame) -> Tuple[dict, dict]:
    Ecol = "Vf" if "Vf" in df_curve.columns else ("Vu" if "Vu" in df_curve.columns else None)
    if Ecol is None or "Im" not in df_curve.columns:
        return {}, {}
        
    E = df_curve[Ecol].values
    I = df_curve["Im"].values
    abs_I = np.abs(I)
    
    if len(abs_I) < 10:
        return {}, {}
        
    I_max = np.max(abs_I)
    
    onset_mask = abs_I >= 0.10 * I_max
    E_onset = E[onset_mask][0] if np.any(onset_mask) else np.nan
    
    tafel_mask = (abs_I >= 0.10 * I_max) & (abs_I <= 0.50 * I_max)
    E_tafel = E[tafel_mask]
    I_tafel = abs_I[tafel_mask]
    
    tafel_slope = np.nan
    slope = np.nan
    intercept = np.nan
    log_I_tafel = []
    
    if len(E_tafel) > 5:
        log_I_tafel = np.log10(I_tafel)
        try:
            slope, intercept = np.polyfit(log_I_tafel, E_tafel, 1)
            tafel_slope = abs(slope * 1000) 
        except Exception:
            pass
            
    params = {
        "|I_max| (A)": I_max,
        "E_onset (V)": E_onset,
        "Tafel Slope (mV/dec)": tafel_slope
    }
    
    valid_I = abs_I > 0
    log_I_full = np.full_like(abs_I, np.nan, dtype=float)
    log_I_full[valid_I] = np.log10(abs_I[valid_I])
    
    fit_data = {
        "E_full": E,
        "log_I_full": log_I_full,
        "log_I_fit": log_I_tafel,
        "slope": slope,
        "intercept": intercept
    }
    
    return params, fit_data

# ============================================================
# PARSERS
# ============================================================
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

def parse_biologic_mpt(raw: str):
    lines = raw.splitlines()
    meta = {}
    header_lines = 0

    for line in lines:
        if "Nb header lines" in line:
            try:
                header_lines = int(line.split(":")[-1].strip())
            except:
                header_lines = 0
            break

    for i in range(min(header_lines, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
            
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
        else:
            parts = re.split(r'\s{2,}|\t+', line)
            if len(parts) >= 2:
                meta[parts[0].strip()] = parts[1].strip()

    data_lines = [line for line in lines[header_lines:] if line.strip()]

    if not data_lines:
        return meta, []

    line_minus_1 = [c.strip() for c in lines[header_lines - 1].split('\t')] if header_lines >= 1 else []
    line_minus_2 = [c.strip() for c in lines[header_lines - 2].split('\t')] if header_lines >= 2 else []
    
    line_minus_1 = [c for c in line_minus_1 if c]
    line_minus_2 = [c for c in line_minus_2 if c]

    first_data_row = [c.strip() for c in data_lines[0].split('\t') if c.strip()]
    num_cols = len(first_data_row)

    if len(line_minus_1) == num_cols:
        cols = line_minus_1
    elif len(line_minus_2) + len(line_minus_1) == num_cols:
        cols = line_minus_2 + line_minus_1
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
        if len(parts) >= num_cols:
            rows.append(parts[:num_cols])
        elif len(parts) > 0:
            rows.append(parts + [np.nan] * (num_cols - len(parts)))

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

    for c in df.columns:
        col = df[c].astype(str)
        col = col.str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(col, errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all").reset_index(drop=True)

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "ewe" in cl or "potential" in cl or "voltage" in cl:
            col_map[c] = "Vf"
        elif "<i>" in cl or "current" in cl or "i/ma" in cl:
            col_map[c] = "Im"
        elif cl == "cycle number" or cl == "cycle":
            col_map[c] = "Cycle"

    df = df.rename(columns=col_map)

    if "Vf" not in df.columns or "Im" not in df.columns:
        return meta, []

    if df["Im"].abs().max() > 1:
        df["Im"] = df["Im"] / 1000

    curves = []
    if "Cycle" in df.columns:
        unique_cycles = sorted(df["Cycle"].dropna().unique())
        for cyc in unique_cycles:
            df_cyc = df[df["Cycle"] == cyc].copy()
            if len(df_cyc) > 0:
                c_num = int(cyc) if float(cyc).is_integer() else cyc
                curves.append((f"Cycle {c_num}", df_cyc.reset_index(drop=True)))
    else:
        curves.append(("Curve 1", df))

    return meta, curves

# ============================================================
# EXPORT
# ============================================================
def convert_df_to_excel(curves_list: List[Tuple[str, pd.DataFrame]]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for cid, df in curves_list:
            Ecol = "Vf" if "Vf" in df.columns else ("Vu" if "Vu" in df.columns else None)
            if Ecol is None or "Im" not in df.columns:
                continue
            clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[Ecol, "Im"])
            if len(clean_df) >= 10:
                clean_df.to_excel(writer, index=False, sheet_name=cid)
    return output.getvalue()

# ============================================================
# APP LOGIC
# ============================================================
uploaded_files = st.file_uploader("Upload CV/LSV files", type=["DTA", "dta", "mpt", "MPT"], accept_multiple_files=True)
default_colors = px.colors.qualitative.Plotly
combined_palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Plotly 

SUPER_PALETTES = [
    px.colors.sequential.Blues[::-1],  
    px.colors.sequential.Reds[::-1],   
    px.colors.sequential.Greens[::-1], 
    px.colors.sequential.Purples[::-1],
    px.colors.sequential.Oranges[::-1],
    px.colors.sequential.Greys[::-1]
]

if uploaded_files:
    file_dict = {f.name: f for f in uploaded_files}
    display_names = set([f"⋮⋮ {name}" for name in file_dict.keys()])
    
    if 'file_groups' not in st.session_state:
        st.session_state.file_groups = [
            {"header": "📥 Unassigned Files", "items": []},
            {"header": "📊 Group 1", "items": []}
        ]
        
    for group in st.session_state.file_groups:
        group["items"] = [item for item in group["items"] if item in display_names]
        
    existing_items = set([item for group in st.session_state.file_groups for item in group["items"]])
    new_items = display_names - existing_items
    if new_items:
        st.session_state.file_groups[0]["items"].extend(list(new_items))

    # --- SIDEBAR: DRAG AND DROP GROUPS ---
    with st.sidebar:
        st.header("🗂️ Drag & Drop Groups")
        st.markdown("Organize your files into groups.")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Add Group"):
                new_idx = len(st.session_state.file_groups)
                st.session_state.file_groups.append({"header": f"📊 Group {new_idx}", "items": []})
                st.rerun()
        with c2:
            if st.button("➖ Remove Group") and len(st.session_state.file_groups) > 1:
                orphans = st.session_state.file_groups[-1]["items"]
                st.session_state.file_groups[0]["items"].extend(orphans)
                st.session_state.file_groups.pop()
                st.rerun()
                
        st.session_state.file_groups = sort_items(st.session_state.file_groups, multi_containers=True)
        
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; margin-top: 50px;'>
                <p style='color: #888888; font-size: 0.85rem; font-family: sans-serif;'>
                    Developed by<br>
                    <b>PhD(c) Carlos A. Torres-Ramírez</b><br><br>
                    <i>Optimized for GAMRY 1010B & Biologic Formats</i>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- TOP AREA: GROUPED COMPARISONS ---
    has_groups_plotted = False
    valid_groups_for_super = []
    
    for g_idx, group in enumerate(st.session_state.file_groups):
        if g_idx == 0: continue 
        if not group["items"]: continue 
        
        valid_groups_for_super.append(group["header"])
            
        if not has_groups_plotted:
            st.header("📈 Group Comparisons")
            has_groups_plotted = True
            
        st.subheader(f"{group['header']}")
        
        fig_comp = go.Figure()
        fig_tafel_comp = go.Figure()
        trace_idx = 0
        group_lsv_params = [] 
        is_group_lsv = False
        
        for item in group["items"]:
            fname = item.replace("⋮⋮ ", "")
            if fname not in file_dict: continue
            
            file = file_dict[fname]
            raw_text = file.getvalue().decode("utf-8", errors="replace")
            
            technique_group = "Unknown Technique"
            
            if instrument.startswith("Gamry"):
                meta_g, curves_comp = parse_gamry_dta_multi_curve(raw_text)
                tag_g = meta_g.get("TAG", "").upper()
                title_g = meta_g.get("TITLE", "").upper()
                if "LSV" in tag_g or "LINEAR" in title_g:
                    technique_group = "LSV"
            else:
                meta_g, curves_comp = parse_biologic_mpt(raw_text)
                if not "E2 (V)" in meta_g:
                    technique_group = "LSV"
                
            for cid, df_comp in curves_comp:
                Ecol = "Vf" if "Vf" in df_comp.columns else ("Vu" if "Vu" in df_comp.columns else None)
                if Ecol and "Im" in df_comp.columns:
                    dd_comp = df_comp[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(dd_comp) >= 10:
                        trace_name = f"{fname}" if len(curves_comp) == 1 else f"{fname} ({cid})"
                        c_color = combined_palette[trace_idx % len(combined_palette)]
                        
                        fig_comp.add_trace(go.Scatter(
                            x=dd_comp[Ecol], 
                            y=dd_comp["Im"], 
                            mode='lines',
                            name=trace_name,
                            line=dict(color=c_color, width=2)
                        ))
                        trace_idx += 1
                        
                        if technique_group == "LSV":
                            is_group_lsv = True
                            cat_params, fit_data = extract_lsv_catalytic_parameters(dd_comp)
                            if cat_params:
                                cat_params = {"File": fname, "Curve": cid, **cat_params} 
                                group_lsv_params.append(cat_params)
                                
                                fig_tafel_comp.add_trace(go.Scatter(
                                    x=fit_data["log_I_full"],
                                    y=fit_data["E_full"],
                                    mode='lines',
                                    name=trace_name,
                                    line=dict(color=c_color, width=2)
                                ))
                                
                                # Trazar línea de ajuste extrapolada con valor en leyenda
                                if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                                    min_x = np.min(fit_data["log_I_fit"])
                                    max_x = np.max(fit_data["log_I_fit"])
                                    span = max_x - min_x
                                    
                                    # Extrapolar la línea un poco más allá de la zona de ajuste para visualizar la tangente
                                    fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                                    fit_y = fit_data["slope"] * fit_x + fit_data["intercept"]
                                    
                                    tafel_val = cat_params["Tafel Slope (mV/dec)"]
                                    
                                    fig_tafel_comp.add_trace(go.Scatter(
                                        x=fit_x, 
                                        y=fit_y, 
                                        mode='lines',
                                        name=f"Fit: {tafel_val:.1f} mV/dec",
                                        line=dict(color=c_color, width=2, dash='dot')
                                    ))
                        
        fig_comp.update_layout(
            xaxis_title="E (V vs Ref.)",
            yaxis_title="I (A)",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.5)"),
            height=600
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        if is_group_lsv:
            fig_tafel_comp.update_layout(
                title="Tafel Plot Comparison (log₁₀|I| vs E)",
                xaxis_title="log₁₀|I| (A)",
                yaxis_title="E (V vs Ref.)",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.5)"),
                height=600
            )
            st.plotly_chart(fig_tafel_comp, use_container_width=True)
        
        if group_lsv_params:
            st.markdown("#### 🧪 Group Catalytic Statistics (LSV)")
            st.info("ℹ️ **Heurística de Cálculo:** $E_{onset}$ se calcula dinámicamente al 10% de $|I_{max}|$. La Pendiente de Tafel se ajusta mediante regresión lineal en la ventana entre el 10% y el 50% de $|I_{max}|$ (representado por las líneas punteadas).")
            
            df_cat = pd.DataFrame(group_lsv_params)
            summary = []
            
            for col in ["|I_max| (A)", "E_onset (V)", "Tafel Slope (mV/dec)"]:
                if col in df_cat.columns:
                    mean_v = df_cat[col].mean()
                    std_v = df_cat[col].std()
                    n_v = df_cat[col].notna().sum()
                    rsd_v = (std_v / abs(mean_v) * 100) if (pd.notna(mean_v) and mean_v != 0) else np.nan
                    
                    summary.append({
                        "Parameter": col,
                        "Mean": round(mean_v, 6) if pd.notna(mean_v) else "N/A",
                        "Std Dev (±)": round(std_v, 6) if pd.notna(std_v) else "N/A",
                        "RSD (%)": round(rsd_v, 2) if pd.notna(rsd_v) else "N/A",
                        "Count (n)": int(n_v)
                    })
            
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
            with st.expander(f"View raw catalytic data for {group['header']}"):
                st.dataframe(df_cat, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)


    # --- ZONA: SUPER GROUPS ---
    st.markdown("---")
    st.header("🧬 Super Groups (Group of Groups)")
    st.markdown("Combine entire groups into a single plot. **Each group will be assigned a distinct base color** (e.g. Group 1 in Blues, Group 2 in Reds).")

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
                "Select basic groups to merge:", 
                options=valid_groups_for_super, 
                key=f"super_group_select_{sg}"
            )
            
            if selected_groups:
                fig_super = go.Figure()
                
                for g_idx, g_name in enumerate(selected_groups):
                    group_data = next(g for g in st.session_state.file_groups if g["header"] == g_name)
                    current_palette = SUPER_PALETTES[g_idx % len(SUPER_PALETTES)]
                    item_idx = 0
                    
                    for item in group_data["items"]:
                        fname = item.replace("⋮⋮ ", "")
                        if fname not in file_dict: continue
                        
                        file = file_dict[fname]
                        raw_text = file.getvalue().decode("utf-8", errors="replace")
                        
                        if instrument.startswith("Gamry"):
                            _, curves_comp = parse_gamry_dta_multi_curve(raw_text)
                        else:
                            _, curves_comp = parse_biologic_mpt(raw_text)
                            
                        for cid, df_comp in curves_comp:
                            Ecol = "Vf" if "Vf" in df_comp.columns else ("Vu" if "Vu" in df_comp.columns else None)
                            if Ecol and "Im" in df_comp.columns:
                                dd_comp = df_comp[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna()
                                if len(dd_comp) >= 10:
                                    trace_name = f"{g_name} | {fname}" if len(curves_comp) == 1 else f"{g_name} | {fname} ({cid})"
                                    color_shade = current_palette[(item_idx * 2) % len(current_palette)]
                                    
                                    fig_super.add_trace(go.Scatter(
                                        x=dd_comp[Ecol], 
                                        y=dd_comp["Im"], 
                                        mode='lines',
                                        name=trace_name,
                                        line=dict(color=color_shade, width=2)
                                    ))
                                    item_idx += 1
                                    
                fig_super.update_layout(
                    title=f"Combined Plot: {', '.join(selected_groups)}",
                    xaxis_title="E (V vs Ref.)",
                    yaxis_title="I (A)",
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.5)"),
                    height=700
                )
                
                st.plotly_chart(fig_super, use_container_width=True)

    # --- BOTTOM AREA: INDIVIDUAL ANALYSIS ---
    st.markdown("---")
    st.header("📄 Individual Analysis")

    actual_file_order = [item.replace("⋮⋮ ", "") for group in st.session_state.file_groups for item in group["items"]]

    for file_name in actual_file_order:
        if file_name not in file_dict:
            continue 
            
        file = file_dict[file_name]
        raw_text = file.getvalue().decode("utf-8", errors="replace")
        
        technique = "Unknown Technique"

        if instrument.startswith("Gamry"):
            meta, curves = parse_gamry_dta_multi_curve(raw_text)
            
            tag = meta.get("TAG", "").upper()
            title = meta.get("TITLE", "").upper()
            
            if "LSV" in tag or "LINEAR" in title:
                technique = "Linear Sweep Voltammetry (LSV)"
                vinit = _to_float(meta.get("VINIT"))
                vlim1 = _to_float(meta.get("VFINAL"))
                vlim2 = None
            else:
                technique = "Cyclic Voltammetry (CV)"
                vinit = _to_float(meta.get("VINIT"))
                vlim1 = _to_float(meta.get("VLIMIT1"))
                vlim2 = _to_float(meta.get("VLIMIT2"))
            
            sr = _to_float(meta.get("SCANRATE"))
            
        else:
            meta, curves = parse_biologic_mpt(raw_text)
            
            if "E2 (V)" in meta:
                technique = "Cyclic Voltammetry (CV)"
                vinit = _to_float(meta.get("Ei (V)"))
                vlim1 = _to_float(meta.get("E1 (V)"))
                vlim2 = _to_float(meta.get("E2 (V)"))
            else:
                technique = "Linear Sweep Voltammetry (LSV)"
                vinit = _to_float(meta.get("Ei (V)"))
                vlim1 = _to_float(meta.get("E1 (V)")) or _to_float(meta.get("Ef (V)"))
                vlim2 = None
                
            sr = _to_float(meta.get("dE/dt"))

        if not curves:
            st.error(f"❌ Could not parse {file.name}. Check format.")
            continue

        st.markdown(f"### {file.name}")
        col_title, col_btn = st.columns([4, 1])
        with col_title:
            st.markdown(f"🔬 **Technique Detected:** `{technique}`")
        with col_btn:
            excel_data = convert_df_to_excel(curves)
            st.download_button(
                label="📥 Export to Excel",
                help=f"Export curve data to Excel from {file.name}",
                data=excel_data,
                file_name=f"{file.name.split('.')[0]}_Data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{file.name}"
            )
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial Potential", f"{vinit} V" if vinit is not None else "N/A")
        
        if technique == "Linear Sweep Voltammetry (LSV)":
            c2.metric("Final Potential", f"{vlim1} V" if vlim1 is not None else "N/A")
            c3.metric("Scan Limit 2", "N/A")
        else:
            c2.metric("Scan Limit 1", f"{vlim1} V" if vlim1 is not None else "N/A")
            c3.metric("Scan Limit 2", f"{vlim2} V" if vlim2 is not None else "N/A")
            
        c4.metric("Scan Rate", f"{sr} mV/s" if sr is not None else "N/A")

        fig = go.Figure()
        fig_tafel = go.Figure()
        results_list = []
        lsv_cat_list = [] 
        
        for i, (cid, dfi) in enumerate(curves):
            Ecol = "Vf" if "Vf" in dfi.columns else ("Vu" if "Vu" in dfi.columns else None)
            if Ecol is None or "Im" not in dfi.columns:
                continue
                
            dd = dfi[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(dd) < 10:
                continue
            
            line_color = default_colors[i % len(default_colors)]
            
            fig.add_trace(go.Scatter(
                x=dd[Ecol], 
                y=dd["Im"], 
                mode='lines',
                name=cid,
                line=dict(color=line_color, width=2)
            ))
            
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
            
            if technique == "Linear Sweep Voltammetry (LSV)":
                cat_params, fit_data = extract_lsv_catalytic_parameters(dd)
                if cat_params:
                    cat_params = {"Curve": cid, **cat_params}
                    lsv_cat_list.append(cat_params)
                    
                    fig_tafel.add_trace(go.Scatter(
                        x=fit_data["log_I_full"], 
                        y=fit_data["E_full"], 
                        mode='lines',
                        name=f"{cid} (Log Curve)", 
                        line=dict(color=line_color, width=2)
                    ))
                    
                    # Trazar línea de ajuste extrapolada
                    if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                        min_x = np.min(fit_data["log_I_fit"])
                        max_x = np.max(fit_data["log_I_fit"])
                        span = max_x - min_x
                        
                        fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                        fit_y = fit_data["slope"] * fit_x + fit_data["intercept"]
                        
                        tafel_val = cat_params["Tafel Slope (mV/dec)"]
                        
                        fig_tafel.add_trace(go.Scatter(
                            x=fit_x, 
                            y=fit_y, 
                            mode='lines',
                            name=f"{cid} Fit ({tafel_val:.1f} mV/dec)", 
                            line=dict(color=line_color, width=2, dash='dot')
                        ))

        fig.update_layout(
            xaxis_title="E (V vs Ref.)",
            yaxis_title="I (A)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0)")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if technique == "Linear Sweep Voltammetry (LSV)" and lsv_cat_list:
            fig_tafel.update_layout(
                title="Tafel Plot (log₁₀|I| vs E)",
                xaxis_title="log₁₀|I| (A)",
                yaxis_title="E (V vs Ref.)",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.5)"),
                height=500
            )
            st.plotly_chart(fig_tafel, use_container_width=True)
        
        if results_list:
            st.write("**Recommended Operating Ranges:**")
            st.dataframe(pd.DataFrame(results_list), use_container_width=True)
            
        if lsv_cat_list:
            st.write("**🧪 Catalytic Parameters:**")
            st.dataframe(pd.DataFrame(lsv_cat_list), use_container_width=True)
            
        st.markdown("<br><br>", unsafe_allow_html=True)
