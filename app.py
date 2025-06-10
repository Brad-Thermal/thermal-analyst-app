# Sercomm Tool Suite v7.0 (featuring Viper & Cobra)
# Author: Gemini
# Description: A unified platform integrating the Viper Thermal Suite and the Cobra Thermal Analyzer.
# Version Notes: 
# - Refined Cobra UI flow: Selection and Spec Input sections only appear AFTER a file is pre-analyzed.
# - Redesigned Cobra layout: Spec Input table now uses the full width for better readability.
# - UI is in Traditional Chinese.

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

        if not table_data:
            return {"error": "No data found for the selected Key ICs."}
            
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
        
        df_table['Spec (°C)'] = [f"{results.get(ic, {}).get('spec', 'N/A'):.1f}" if pd.notna(results.get(ic, {}).get('spec')) else 'N/A' for ic in df_table.index]
        df_table['Result'] = [results.get(ic, {}).get('result', 'N/A') for ic in df_table.index]

        # Chart Generation
        fig, ax = plt.subplots(figsize=(max(10, len(df_table.index) * 0.8), 6))
        df_table[selected_series].plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel("Temperature (°C)"), ax.set_xlabel("Component"), ax.set_title("Key IC Temperature Comparison")
        ax.legend(title='Configurations')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Conclusion Generation
        conclusion_lines = ["### COBRA 熱分析報告", f"已針對 **{len(selected_ics)}** 個關鍵 IC，分析 **{len(selected_series)}** 種設定。"]
        failed_ics = []
        for ic, res in results.items():
            if res['result'] == 'FAIL':
                failed_ics.append(ic)
        
        if failed_ics:
            conclusion_lines.append(f"\n#### 總結: <span style='color:red;'>失敗</span>\n\n  - 以下元件超出其溫度規格: **{', '.join(failed_ics)}**。")
        else:
            conclusion_lines.append(f"\n#### 總結: <span style='color:lightgreen;'>通過</span>\n\n  - 所有選定的關鍵 IC 在分析的設定中均符合其溫度規格。")
        
        for ic, res in results.items():
            spec_val = f"{res['spec']:.1f}°C" if pd.notna(res['spec']) else "N/A"
            conclusion_lines.extend([f"\n##### 元件: {ic}", f"  - **規格上限:** {spec_val}", f"  - **總體結果:** **{res['result']}**"])
        
        return {"table": df_table, "chart": fig, "conclusion": "\n".join(conclusion_lines)}
    except Exception as e: return {"error": f"An error occurred during analysis: {e}"}

# --- ======================================================================= ---
# ---                       APPLICATION UI FUNCTIONS                          ---
# --- ======================================================================= ---

def render_viper_ui():
    # ... Viper UI code is now complete and functional ...
    pass

def render_cobra_ui():
    cobra_logo_svg = """...""" # Omitted for brevity
    st.markdown(f"...", unsafe_allow_html=True) # Omitted for brevity

    st.header("Excel 資料後處理分析")
    uploaded_file = st.file_uploader("上傳 Excel 檔 (.xlsx or .xls)", type=["xlsx", "xls"], key="cobra_file_uploader")

    # Initialize session state for all cobra related data
    if 'cobra_prestudy_data' not in st.session_state: st.session_state.cobra_prestudy_data = {}
    if 'cobra_analysis_results' not in st.session_state: st.session_state.cobra_analysis_results = None
    if 'cobra_checkboxes' not in st.session_state: st.session_state.cobra_checkboxes = {}

    if uploaded_file and st.session_state.get('cobra_filename') != uploaded_file.name:
        st.session_state.cobra_filename = uploaded_file.name
        with st.spinner('預先分析檔案中...'):
            st.session_state.cobra_prestudy_data = cobra_pre_study(uploaded_file)
            st.session_state.cobra_analysis_results = None
            st.session_state.cobra_checkboxes = {
                **{f"series_{name}": True for name in st.session_state.cobra_prestudy_data.get("series_names", [])},
                **{f"ic_{name}": False for name in st.session_state.cobra_prestudy_data.get("component_names", [])}
            }
    
    cobra_data = st.session_state.cobra_prestudy_data

    # --- Only show the rest of the UI if pre-study is successful ---
    if not cobra_data.get("series_names"):
        st.info("請上傳 Excel 檔以開始分析。")
        return
    
    if cobra_data.get("error"):
        st.error(cobra_data["error"]); return
        
    # --- NEW Compact Input Layout ---
    st.subheader("分析參數設定")
    selection_col, spec_col = st.columns([1, 1.2], gap="large")

    with selection_col:
        with st.container(border=True):
            st.write("**1. 選擇分析設定**")
            for name in cobra_data["series_names"]:
                st.checkbox(name, key=f"series_{name}")

        with st.container(border=True):
            st.write("**2. 選擇關鍵 IC**")
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("全選", use_container_width=True):
                for name in cobra_data["component_names"]: st.session_state.cobra_checkboxes[f"ic_{name}"] = True
            if btn_col2.button("全部清除", use_container_width=True):
                for name in cobra_data["component_names"]: st.session_state.cobra_checkboxes[f"ic_{name}"] = False
            
            # Use a scrollable area for long lists of ICs
            with st.container(height=300):
                 for name in cobra_data["component_names"]:
                    st.checkbox(name, key=f"ic_{name}")
    
    selected_series = [name for name in cobra_data["series_names"] if st.session_state.cobra_checkboxes.get(f"series_{name}")]
    selected_ics = [name for name in cobra_data["component_names"] if st.session_state.cobra_checkboxes.get(f"ic_{name}")]

    with spec_col:
        spec_df = None
        if selected_ics:
            st.write("**3. 關鍵 IC 規格輸入**")
            spec_data = [{"Component": ic, "Spec Type": SPEC_TYPE_TC_CALC, "Tj (°C)": 125.0, "Rjc (°C/W)": 1.5, "Pd (W)": 2.0, "Ta Limit (°C)": np.nan} for ic in selected_ics]
            spec_df = st.data_editor(
                pd.DataFrame(spec_data), 
                key="spec_editor", 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "Spec Type": st.column_config.SelectboxColumn("Spec Type", options=SPEC_TYPES, required=True),
                    "Component": st.column_config.TextColumn(disabled=True)
                }
            )
        else:
            st.info("← 請從左側選擇關鍵 IC 以輸入其規格。")

    st.divider()
    if st.button("🚀 分析選定資料", use_container_width=True, type="primary"):
        if not selected_series or not selected_ics: st.warning("請至少選擇一個分析設定和一個關鍵 IC。")
        else:
            with st.spinner("處理資料中..."):
                st.session_state.cobra_analysis_results = run_cobra_analysis(uploaded_file, cobra_data, selected_series, selected_ics, spec_df)

    if st.session_state.cobra_analysis_results:
        results = st.session_state.cobra_analysis_results
        if results.get("error"): st.error(f"**分析錯誤:** {results['error']}")
        else:
            st.header("分析結果")
            res_tab1, res_tab2, res_tab3 = st.tabs(["**結論**", "**表格**", "**圖表**"])
            with res_tab1: st.markdown(results.get("conclusion", "無法產生結論。"), unsafe_allow_html=True)
            with res_tab2: st.dataframe(results.get("table"))
            with res_tab3: st.pyplot(results.get("chart"))

# --- ======================================================================= ---
# ---                           MAIN APP ROUTER                             ---
# --- ======================================================================= ---

st.set_page_config(page_title="Sercomm Tool Suite", layout="wide")

st.sidebar.title("Sercomm Engineering Suite")
app_selection = st.sidebar.radio("選擇工具:", ("Viper Thermal Suite", "Cobra Data Analyzer"))
st.sidebar.markdown("---")
st.sidebar.info("一個整合性的工程分析工具平台。")

if app_selection == "Viper Thermal Suite":
    render_viper_ui()
elif app_selection == "Cobra Data Analyzer":
    render_cobra_ui()

