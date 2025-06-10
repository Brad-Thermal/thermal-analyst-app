# Sercomm Tool Suite v4.0 (featuring Viper & Cobra)
# Author: Gemini
# Description: A unified platform integrating the Viper Thermal Suite and the Cobra Thermal Analyzer.
# Version Notes: 
# - BUG FIX: Restored the complete UI code for the Viper Thermal Suite module.
# - Reverted Viper UI to Traditional Chinese.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import re
import io

# --- ======================================================================= ---
# ---                             SHARED CONSTANTS                            ---
# --- ======================================================================= ---

# Suppress specific Streamlit warnings
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

# Constants for Viper
STEFAN_BOLTZMANN_CONST = 5.67e-8
EPSILON = 1e-9
BUILT_IN_SAFETY_FACTOR = 0.9
AIR_DENSITY_RHO = 1.225
AIR_SPECIFIC_HEAT_CP = 1006
M3S_TO_CFM_CONVERSION = 2118.88
SOLAR_IRRADIANCE = 1000

# Constants for Cobra
DATA_COL_COMPONENT_IDX = 1
DATA_COL_FIRST_SERIES_TEMP_IDX = 2
SPEC_TYPE_TC_CALC = "Tc"
SPEC_TYPE_TJ_ONLY = "Tj"
SPEC_TYPE_TA_ONLY = "Ta"
SPEC_TYPES = [SPEC_TYPE_TC_CALC, SPEC_TYPE_TJ_ONLY, SPEC_TYPE_TA_ONLY]
DELTA_SYMBOL = "\u0394"

# --- ======================================================================= ---
# ---                     VIPER CALCULATION ENGINES                           ---
# --- ======================================================================= ---

def calculate_natural_convection(L, W, H, Ts_peak, Ta, material_props):
    if Ts_peak <= Ta: return { "error": "外殼允許溫度 (Ts) 必須高於環境溫度 (Ta)。" }
    if L <= 0 or W <= 0 or H <= 0: return { "error": "產品的長、寬、高尺寸必須大於零。" }
    try:
        epsilon, k_uniform = material_props["emissivity"], material_props["k_uniform"]
        Ts_eff = Ta + (Ts_peak - Ta) * k_uniform
        delta_T_eff = Ts_eff - Ta
        L_m, W_m, H_m = L/1000, W/1000, H/1000
        A_total = 2 * (L_m*W_m + L_m*H_m + W_m*H_m)
        T_film = (Ts_eff + Ta) / 2
        k_air, nu_air, pr_air, beta, g = 0.0275, 1.85e-5, 0.72, 1 / (T_film + 273.15 + EPSILON), 9.81
        Lc_vert, Lc_horiz = H_m, (L_m * W_m) / (2 * (L_m + W_m) + EPSILON)
        Ra_vert, Ra_horiz = (g*beta*delta_T_eff*Lc_vert**3)/(nu_air**2)*pr_air, (g*beta*delta_T_eff*Lc_horiz**3)/(nu_air**2)*pr_air
        Nu_vert = (0.825 + (0.387 * abs(Ra_vert)**(1/6)) / (1 + (0.492/pr_air)**(9/16))**(8/27))**2
        if 1e4 <= Ra_horiz <= 1e7: Nu_top = 0.54 * Ra_horiz**(1/4)
        elif Ra_horiz > 1e7: Nu_top = 0.15 * Ra_horiz**(1/3)
        else: Nu_top = 1.0
        Nu_bottom = 0.27 * abs(Ra_horiz)**(1/4)
        h_vert, h_top, h_bottom = (Nu_vert*k_air)/(Lc_vert+EPSILON), (Nu_top*k_air)/(Lc_horiz+EPSILON), (Nu_bottom*k_air)/(Lc_horiz+EPSILON)
        Q_conv_total = ((h_top*L_m*W_m)+(h_bottom*L_m*W_m)+h_vert*2*(L_m*H_m+W_m*H_m))*delta_T_eff
        h_avg = Q_conv_total / ((A_total+EPSILON)*delta_T_eff)
        Ts_eff_K, Ta_K = Ts_eff + 273.15, Ta + 273.15
        Q_rad = epsilon * STEFAN_BOLTZMANN_CONST * A_total * (Ts_eff_K**4 - Ta_K**4)
        Q_ideal_total = Q_conv_total + Q_rad
        Q_final = Q_ideal_total * BUILT_IN_SAFETY_FACTOR
        return {"total_power": Q_final, "error": None}
    except Exception as e: return {"error": f"計算過程中發生未預期的錯誤: {e}"}

def calculate_forced_convection(power_q, T_in, T_out):
    if T_out <= T_in: return {"error": "出風口溫度必須高於進風口溫度。"}
    if power_q <= 0: return {"error": "需散熱的功耗必須大於零。"}
    delta_T = T_out - T_in
    mass_flow_rate = power_q / (AIR_SPECIFIC_HEAT_CP * delta_T)
    volume_flow_rate_m3s = mass_flow_rate / AIR_DENSITY_RHO
    return {"cfm": volume_flow_rate_m3s * M3S_TO_CFM_CONVERSION, "error": None}

def calculate_solar_gain(projected_area_mm2, alpha, solar_irradiance):
    if projected_area_mm2 <= 0: return {"error": "曝曬投影面積必須大於零。"}
    try:
        projected_area_m2 = projected_area_mm2 / 1_000_000
        return {"solar_gain": alpha * projected_area_m2 * solar_irradiance, "error": None}
    except Exception as e: return {"error": f"計算過程中發生未預期的錯誤: {e}"}

# --- ======================================================================= ---
# ---                     COBRA DATA PROCESSING LOGIC                         ---
# --- ======================================================================= ---
def clean_series_header(raw_header: str) -> str:
    temp_name = str(raw_header).strip()
    if not temp_name: return "Unnamed Series"
    if temp_name.upper() in ["DEFAULT", "BASELINE"]: return temp_name.capitalize()
    
    bracket_match = re.search(r"\[(.*?)\]", temp_name)
    if bracket_match:
        content = bracket_match.group(1).strip()
        if content.upper() in ["DEFAULT", "BASELINE"]: return content.capitalize()
        if content and not any(k.upper() == content.upper() for k in ["CONFIGURATION", "CASE", "OPTION", "SERIES", "TEMP", "MAX"]):
            return content
        temp_name = temp_name.replace(bracket_match.group(0), "").strip()

    patterns = [r"Temperature \(Solid\) Max.*", r"\[°C\]", r"\(°C\)", r"°C", r"PoE mode_battery.*", r"k=10.*"]
    for p in patterns: temp_name = re.sub(p, "", temp_name, flags=re.IGNORECASE).strip()
    
    temp_name = temp_name.strip().replace("_", " ")
    temp_name = re.sub(r"\s+", " ", temp_name).strip()
    return temp_name if temp_name else "Unnamed Series"

def clean_component_display_name(raw_name: str) -> str:
    name = str(raw_name).strip()
    if not name: return "Unnamed Component"
    name = re.sub(r"^VG\s+", "", name, flags=re.IGNORECASE)
    suffixes = [r"Temperature \(Solid\) Max.*", r"Max \[°C\]", r"\[°C\]", r"\(°C\)", r"°C"]
    for suffix in suffixes: name = re.sub(suffix, "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"-\s*[\d\.\*\s\+\-/xX]+W", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"[\s_-]+$", "", name).strip()
    name = re.sub(r"[\s_]+", " ", name).strip()
    return name if name else "Unnamed Component"

def cobra_pre_study(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file)
        if not xls.sheet_names:
            return {"error": "The Excel file contains no sheets."}
        
        target_sheet = xls.sheet_names[-1]
        df_header = pd.read_excel(xls, sheet_name=target_sheet, header=None, nrows=20)

        header_row_idx = -1
        for i, row in df_header.iterrows():
            if str(row.iloc[DATA_COL_COMPONENT_IDX]).strip().upper().startswith("GOAL ("):
                header_row_idx = i
                break
        
        if header_row_idx == -1:
            return {"error": "Could not find 'Goal (Value)' marker in column B."}

        header_row = df_header.iloc[header_row_idx]
        raw_series_names = [str(name).strip() for name in header_row[DATA_COL_FIRST_SERIES_TEMP_IDX:] if str(name).strip() and str(name).strip().lower() != 'nan']
        series_excel_indices = {name: i for i, name in enumerate(header_row) if name in raw_series_names}

        cleaned_names = []
        counts = {}
        cleaned_to_raw_map = {}
        for raw_name in raw_series_names:
            clean_base = clean_series_header(raw_name)
            count = counts.get(clean_base, 0)
            final_name = f"{clean_base}_{count}" if count > 0 else clean_base
            counts[clean_base] = count + 1
            cleaned_names.append(final_name)
            cleaned_to_raw_map[final_name] = raw_name

        data_start_row = header_row_idx + 1
        df_components = pd.read_excel(xls, sheet_name=target_sheet, header=None, usecols=[DATA_COL_COMPONENT_IDX], skiprows=data_start_row, dtype=str)
        unique_original_components = df_components.iloc[:, 0].str.strip().replace('', np.nan).dropna().unique()
        
        cleaned_components_set = {clean_component_display_name(name) for name in unique_original_components if clean_component_display_name(name)}
        
        return {
            "error": None, "series_names": cleaned_names, "component_names": sorted(list(cleaned_components_set)),
            "series_excel_indices": series_excel_indices, "cleaned_to_raw_map": cleaned_to_raw_map,
            "header_row_idx": header_row_idx, "target_sheet": target_sheet
        }
    except Exception as e: return {"error": f"An error occurred during pre-study: {e}"}

def run_cobra_analysis(uploaded_file, cobra_data, selected_series, selected_ics, spec_df):
    try:
        df_full = pd.read_excel(uploaded_file, sheet_name=cobra_data['target_sheet'], header=None, dtype=str)
        df_data = df_full.iloc[cobra_data['header_row_idx'] + 1:].copy()
        analysis_data = {
            cleaned_name: pd.to_numeric(df_data[cobra_data['series_excel_indices'][cobra_data['cleaned_to_raw_map'][cleaned_name]]], errors='coerce')
            for cleaned_name in selected_series
        }
        component_names = df_data[DATA_COL_COMPONENT_IDX].apply(clean_component_display_name)
        
        table_data = []
        key_ic_data = {}
        for ic in selected_ics:
            match_indices = component_names[component_names == ic].index
            if not match_indices.empty:
                idx = match_indices[0]
                temps = {s_name: analysis_data[s_name].loc[idx] for s_name in selected_series}
                key_ic_data[ic] = temps
                table_data.append({"Component": ic, **temps})

        df_table = pd.DataFrame(table_data).set_index("Component")
        results = {}
        for _, spec_row in spec_df.iterrows():
            ic, spec_type = spec_row['Component'], spec_row['Spec Type']
            effective_spec = np.nan
            try:
                if spec_type == SPEC_TYPE_TC_CALC: effective_spec = float(spec_row['Tj (°C)']) - (float(spec_row['Pd (W)']) * float(spec_row['Rjc (°C/W)']))
                elif spec_type == SPEC_TYPE_TJ_ONLY: effective_spec = float(spec_row['Tj (°C)'])
                elif spec_type == SPEC_TYPE_TA_ONLY: effective_spec = float(spec_row['Ta Limit (°C)'])
            except (ValueError, TypeError): pass
            
            results[ic] = {"spec": effective_spec, "result": "PASS"}
            if pd.notna(effective_spec) and ic in key_ic_data:
                if any(pd.notna(temp) and temp > effective_spec for temp in key_ic_data[ic].values()):
                    results[ic]["result"] = "FAIL"
        
        df_table['Spec (°C)'] = df_table.index.map(lambda ic: f"{results.get(ic, {}).get('spec', 'N/A'):.1f}" if pd.notna(results.get(ic, {}).get('spec')) else 'N/A')
        df_table['Result'] = df_table.index.map(lambda ic: results.get(ic, {}).get('result', 'N/A'))

        fig, ax = plt.subplots(figsize=(10, 6))
        if not df_table.empty:
            df_table[selected_series].plot(kind='bar', ax=ax)
            ax.set_ylabel("Temperature (°C)"), ax.set_xlabel("Component"), ax.set_title("Key IC Temperature Comparison")
            plt.xticks(rotation=45, ha='right'), plt.tight_layout()

        conclusion_lines = ["[H1]COBRA THERMAL ANALYSIS REPORT", f"Analyzed {len(selected_series)} configurations for {len(selected_ics)} Key ICs."]
        for ic, res in results.items():
            spec_val = f"{res['spec']:.1f}°C" if pd.notna(res['spec']) else "N/A"
            conclusion_lines.extend([f"\n[H2]Component: {ic}", f"  - Spec Limit: {spec_val}", f"  - Overall Result: **{res['result']}**"])
        
        return {"table": df_table, "chart": fig, "conclusion": "\n".join(conclusion_lines)}
    except Exception as e: return {"error": f"An error occurred during analysis: {e}"}

# --- ======================================================================= ---
# ---                       APPLICATION UI FUNCTIONS                          ---
# --- ======================================================================= ---

def render_viper_ui():
    viper_logo_svg = """
    <svg width="50" height="50" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M50 10 L85 45 L50 90 L15 45 Z" fill="#1E1E1E" stroke="#FF5733" stroke-width="4"/>
      <path d="M50 25 C 40 35, 40 55, 50 65" stroke="#FFC300" stroke-width="5" stroke-linecap="round" fill="none"/>
      <path d="M50 25 C 60 35, 60 55, 50 65" stroke="#FFC300" stroke-width="5" stroke-linecap="round" fill="none"/>
      <path d="M42 45 L58 45" stroke="#FFC300" stroke-width="5" stroke-linecap="round"/>
      <circle cx="40" cy="35" r="4" fill="#FFFFFF"/>
      <circle cx="60" cy="35" r="4" fill="#FFFFFF"/>
    </svg>
    """
    st.markdown(f"""
        <div style="display: flex; align-items: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px;">
            <div style="margin-right: 15px;">{viper_logo_svg}</div>
            <div>
                <h1 style="margin-bottom: 0; color: #FFFFFF;">Viper Thermal Suite</h1>
                <p style="margin-top: 0; color: #AAAAAA;">延續 Cobra 系列的熱風險評估工具</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    natural_convection_materials = {
        "塑膠 (ABS/PC)": {"emissivity": 0.90, "k_uniform": 0.65},
        "鋁合金 (陽極處理)": {"emissivity": 0.85, "k_uniform": 0.90}
    }
    solar_absorptivity_materials = {
        "白色 (White Paint)": {"absorptivity": 0.25},
        "銀色 (Silver Paint)": {"absorptivity": 0.40},
        "深灰色 (Dark Gray)": {"absorptivity": 0.80},
        "黑色 (Black Plastic/Paint)": {"absorptivity": 0.95}
    }

    tab_nat, tab_force, tab_solar = st.tabs(["🍃 自然對流分析", "🌬️ 強制對流分析", "☀️ 太陽輻射分析"])
    
    with tab_nat:
        st.header("被動散熱功耗估算")
        col_nat_input, col_nat_result = st.columns(2, gap="large")
        with col_nat_input:
            st.subheader("輸入參數")
            nc_material_name = st.selectbox("外殼材質", options=list(natural_convection_materials.keys()), key="nc_mat")
            st.markdown("**產品尺寸 (mm)**")
            dim_col1, dim_col2, dim_col3 = st.columns(3)
            with dim_col1: nc_dim_L = st.number_input("長度 (L)", 1.0, 1000.0, 200.0, 10.0, "%.1f", key="nc_l")
            with dim_col2: nc_dim_W = st.number_input("寬度 (W)", 1.0, 1000.0, 150.0, 10.0, "%.1f", key="nc_w")
            with dim_col3: nc_dim_H = st.number_input("高度 (H)", 1.0, 500.0, 50.0, 5.0, "%.1f", key="nc_h")
            st.markdown("**運作條件 (°C)**")
            op_cond_col1, op_cond_col2 = st.columns(2)
            with op_cond_col1: nc_temp_ambient = st.number_input("環境溫度 (Ta)", 0, 60, 25, key="nc_ta")
            with op_cond_col2: nc_temp_surface_peak = st.number_input("外殼允許溫度 (Ts)", nc_temp_ambient + 1, 100, 50, key="nc_ts")
        with col_nat_result:
            st.subheader("評估結果")
            selected_material_props_nc = natural_convection_materials[nc_material_name]
            nc_results = calculate_natural_convection(nc_dim_L, nc_dim_W, nc_dim_H, nc_temp_surface_peak, nc_temp_ambient, selected_material_props_nc)
            if nc_results.get("error"): st.error(f"**錯誤:** {nc_results['error']}")
            else: st.metric(label="✅ 最大可解析瓦數", value=f"{nc_results['total_power']:.2f} W", help="此結果已內建材質的溫度均勻性及固定的工程安全係數 (0.9)。")

    with tab_force:
        st.header("主動散熱風量估算")
        col_force_input, col_force_result = st.columns(2, gap="large")
        with col_force_input:
            st.subheader("輸入參數")
            fc_param_col1, fc_param_col2 = st.columns(2, gap="medium")
            with fc_param_col1: fc_power_q = st.number_input("目標散熱功耗 (Q, W)", 0.1, value=50.0, step=1.0, format="%.1f", help="風扇需要帶走的總熱量 (單位: 瓦)。")
            with fc_param_col2:
                fc_temp_in = st.number_input("進風口溫度 (Tin, °C)", 0, 60, 25, key="fc_tin")
                fc_temp_out = st.number_input("最高允許出風口溫度 (Tout, °C)", fc_temp_in + 1, 100, 45, key="fc_tout")
            st.subheader("計算公式")
            st.latex(r"Q = \dot{m} \cdot C_p \cdot \Delta T")
        with col_force_result:
            st.subheader("評估結果")
            fc_results = calculate_forced_convection(fc_power_q, fc_temp_in, fc_temp_out)
            if fc_results.get("error"): st.error(f"**錯誤:** {fc_results['error']}")
            else: st.metric(label="🌬️ 所需風量", value=f"{fc_results['cfm']:.2f} CFM", help="CFM (Cubic Feet per Minute): 立方英尺/分鐘。")

    with tab_solar:
        st.header("太陽輻射熱增益估算")
        col_solar_input, col_solar_result = st.columns(2, gap="large")
        with col_solar_input:
            st.subheader("輸入參數")
            solar_material_name = st.selectbox("外殼顏色/表面處理", options=list(solar_absorptivity_materials.keys()) + ["其他..."], key="solar_mat")
            if solar_material_name == "其他...":
                alpha_val = st.number_input("自訂吸收率 (α)", 0.0, 1.0, 0.5, 0.05)
            else:
                alpha_val = solar_absorptivity_materials[solar_material_name]["absorptivity"]
                st.number_input("對應吸收率 (α)", value=alpha_val, disabled=True)
            projected_area_mm2 = st.number_input("曝曬投影面積 (mm²)", 0.0, value=30000.0, step=1000.0, format="%.1f")
            solar_irradiance_val = st.number_input("太陽輻射強度 (W/m²)", 0, value=1000, step=50)
            st.subheader("計算公式")
            st.latex(r"Q_{solar} = \alpha \cdot A_{proj} \cdot G_{solar}")
        with col_solar_result:
            st.subheader("評估結果")
            solar_results = calculate_solar_gain(projected_area_mm2, alpha_val, solar_irradiance_val)
            if solar_results.get("error"): st.error(f"**錯誤:** {solar_results['error']}")
            else: st.metric(label="☀️ 額外吸收的太陽輻射熱", value=f"{solar_results['solar_gain']:.2f} W")

def render_cobra_ui():
    cobra_logo_svg = """..."""
    st.markdown(f"...", unsafe_allow_html=True)

    st.header("Excel Data Post-Processing")
    uploaded_file = st.file_uploader("Upload an Excel file (.xlsx or .xls)", type=["xlsx", "xls"], key="cobra_file_uploader")

    if 'cobra_prestudy_data' not in st.session_state: st.session_state.cobra_prestudy_data = {}
    if 'cobra_analysis_results' not in st.session_state: st.session_state.cobra_analysis_results = None

    if uploaded_file and st.session_state.get('cobra_filename') != uploaded_file.name:
        st.session_state.cobra_filename = uploaded_file.name
        with st.spinner('Pre-analyzing Excel file...'):
            st.session_state.cobra_prestudy_data = cobra_pre_study(uploaded_file)
            st.session_state.cobra_analysis_results = None
    
    cobra_data = st.session_state.cobra_prestudy_data

    if cobra_data.get("error"): st.error(cobra_data["error"]); return
    if not cobra_data.get("series_names"): st.info("Upload an Excel file to see analysis options."); return
        
    st.subheader("Analysis Selections")
    col1, col2 = st.columns(2)
    with col1: selected_series = st.multiselect("Select Configurations:", options=cobra_data["series_names"], default=cobra_data["series_names"])
    with col2: selected_ics = st.multiselect("Select Key ICs:", options=cobra_data["component_names"])

    spec_df = None
    if selected_ics:
        st.divider()
        st.subheader("Key IC Specification Input")
        spec_data = [{"Component": ic, "Spec Type": SPEC_TYPE_TC_CALC, "Tj (°C)": 125.0, "Rjc (°C/W)": 1.5, "Pd (W)": 2.0, "Ta Limit (°C)": np.nan} for ic in selected_ics]
        spec_df = st.data_editor(pd.DataFrame(spec_data), key="spec_editor", hide_index=True)

    st.divider()
    if st.button("🚀 Analyze Selected Data", use_container_width=True, type="primary"):
        if not selected_series or not selected_ics: st.warning("Please select at least one configuration AND one Key IC.")
        else:
            with st.spinner("Processing data..."):
                st.session_state.cobra_analysis_results = run_cobra_analysis(uploaded_file, cobra_data, selected_series, selected_ics, spec_df)

    if st.session_state.cobra_analysis_results:
        results = st.session_state.cobra_analysis_results
        if results.get("error"): st.error(results['error'])
        else:
            st.header("Analysis Results")
            res_tab1, res_tab2, res_tab3 = st.tabs(["Conclusions", "Table", "Chart"])
            with res_tab1:
                conclusion_md = results.get("conclusion", "No conclusion generated.").replace("[H1]", "### ").replace("[H2]", "#### ").replace("**", "**")
                st.markdown(conclusion_md)
            with res_tab2: st.dataframe(results.get("table"))
            with res_tab3: st.pyplot(results.get("chart"))

# --- ======================================================================= ---
# ---                           MAIN APP ROUTER                             ---
# --- ======================================================================= ---

st.set_page_config(page_title="Sercomm Tool Suite", layout="wide")

st.sidebar.title("Sercomm Engineering Suite")
app_selection = st.sidebar.radio("Select a Tool:", ("Viper Thermal Suite", "Cobra Data Analyzer"))
st.sidebar.markdown("---")
st.sidebar.info("A unified platform for Sercomm's engineering analysis tools.")

if app_selection == "Viper Thermal Suite":
    render_viper_ui()
elif app_selection == "Cobra Data Analyzer":
    render_cobra_ui()

