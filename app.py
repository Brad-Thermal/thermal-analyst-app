# Sercomm Tool Suite v12.1
# Author: Gemini
# Description: A unified platform with professional reporting features.
# Version Notes: 
# - BUG FIX: Added the custom header/logo block to the Cobra module.
# - UI is in Traditional Chinese.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import re
import io
import textwrap

# --- ======================================================================= ---
# ---                             SHARED CONSTANTS                            ---
# --- ======================================================================= ---

logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

# Constants
STEFAN_BOLTZMANN_CONST = 5.67e-8
EPSILON = 1e-9
BUILT_IN_SAFETY_FACTOR = 0.9
AIR_DENSITY_RHO = 1.225
AIR_SPECIFIC_HEAT_CP = 1006
M3S_TO_CFM_CONVERSION = 2118.88
SOLAR_IRRADIANCE = 1000
DATA_COL_COMPONENT_IDX = 1
DATA_COL_FIRST_SERIES_TEMP_IDX = 2
SPEC_TYPE_TC_CALC = "Tc"
SPEC_TYPE_TJ_ONLY = "Tj"
SPEC_TYPE_TA_ONLY = "Ta"
SPEC_TYPES = [SPEC_TYPE_TC_CALC, SPEC_TYPE_TJ_ONLY, SPEC_TYPE_TA_ONLY]
DELTA_SYMBOL = "\u0394"
PASS_COLOR_HEX = "#C6EFCE"
FAIL_COLOR_HEX = "#FFC7CE"
NO_COMPARISON_LABEL = "---"

# --- ======================================================================= ---
# ---                     CALCULATION ENGINES                                 ---
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
        if not xls.sheet_names: return {"error": "Excel 檔案不包含任何工作表。"}
        target_sheet = xls.sheet_names[-1]
        df_header = pd.read_excel(xls, sheet_name=target_sheet, header=None, nrows=20)
        header_row_idx = -1
        for i, row in df_header.iterrows():
            if str(row.iloc[DATA_COL_COMPONENT_IDX]).strip().upper().startswith("GOAL ("): header_row_idx = i; break
        if header_row_idx == -1: return {"error": "在 B 欄中找不到 'Goal (Value)' 標記。"}
        
        header_row = df_header.iloc[header_row_idx]
        raw_series_names = [str(name).strip() for name in header_row[DATA_COL_FIRST_SERIES_TEMP_IDX:] if str(name).strip() and str(name).strip().lower() != 'nan']
        series_excel_indices = {name: i for i, name in enumerate(header_row) if name in raw_series_names}

        cleaned_names, counts, cleaned_to_raw_map = [], {}, {}
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
    except Exception as e: return {"error": f"預先分析時發生錯誤: {e}"}

def run_cobra_analysis(uploaded_file, cobra_data, selected_series, selected_ics, spec_df, delta_pairs):
    try:
        df_full = pd.read_excel(uploaded_file, sheet_name=cobra_data['target_sheet'], header=None, dtype=str)
        df_data = df_full.iloc[cobra_data['header_row_idx'] + 1:].copy()
        
        analysis_data = {
            cleaned_name: pd.to_numeric(df_data[cobra_data['series_excel_indices'][cobra_data['cleaned_to_raw_map'][cleaned_name]]], errors='coerce')
            for cleaned_name in selected_series
        }
        component_names = df_data[DATA_COL_COMPONENT_IDX].apply(clean_component_display_name)
        
        table_data, key_ic_data = [], {}
        for ic in selected_ics:
            match_indices = component_names[component_names == ic].index
            if not match_indices.empty:
                idx = match_indices[0]
                temps = {s_name: analysis_data[s_name].loc[idx] for s_name in selected_series}
                key_ic_data[ic] = temps
                table_data.append({"Component": ic, **temps})

        if not table_data:
            return {"error": "找不到所選關鍵 IC 的數據。"}
            
        df_table_numeric = pd.DataFrame(table_data).set_index("Component")
        df_table_display = df_table_numeric.copy()
        
        results, conclusion_data = {}, []
        for _, spec_row in spec_df.iterrows():
            ic, spec_type = spec_row['Component'], spec_row['Spec Type']
            effective_spec, spec_inputs = np.nan, "N/A"
            try:
                if spec_type == SPEC_TYPE_TC_CALC:
                    tj, rjc, pd_val = float(spec_row['Tj (°C)']), float(spec_row['Rjc (°C/W)']), float(spec_row['Pd (W)'])
                    effective_spec, spec_inputs = tj - (pd_val * rjc), f"Tj={tj}, Rjc={rjc}, Pd={pd_val}"
                elif spec_type == SPEC_TYPE_TJ_ONLY:
                    effective_spec, spec_inputs = float(spec_row['Tj (°C)']), f"Tj Max: {spec_row['Tj (°C)']}"
                elif spec_type == SPEC_TYPE_TA_ONLY:
                    effective_spec, spec_inputs = float(spec_row['Ta Limit (°C)']), f"Ta Limit: {spec_row['Ta Limit (°C)']}"
            except (ValueError, TypeError): pass
            
            ic_result = {"spec": effective_spec, "result": "PASS", "spec_type": spec_type, "spec_inputs": spec_inputs, "series_results": []}
            if pd.notna(effective_spec) and ic in key_ic_data:
                for s_name, temp in key_ic_data[ic].items():
                    res = "N/A"
                    if pd.notna(temp):
                        res = "PASS" if temp <= effective_spec else "FAIL"
                        if res == "FAIL": ic_result["result"] = "FAIL"
                    ic_result["series_results"].append({"series": s_name, "temp": temp, "result": res})
            results[ic] = ic_result
            conclusion_data.append({"component": ic, **ic_result})
        
        df_table_display.columns = [f"{col} (°C)" for col in df_table_display.columns]
        df_table_display['Spec (°C)'] = [f"{res.get('spec_type', '')} = {res.get('spec', 'N/A'):.2f}" if pd.notna(res.get('spec')) else "N/A" for ic, res in results.items()]
        df_table_display['Result'] = [results.get(ic, {}).get('result', 'N/A') for ic in df_table_display.index]

        for pair in delta_pairs:
            baseline, compare = pair['baseline'], pair['compare']
            if baseline != NO_COMPARISON_LABEL and compare != NO_COMPARISON_LABEL and baseline != compare:
                temp_b = df_table_numeric[baseline]
                temp_c = df_table_numeric[compare]
                delta_col_name = f"{DELTA_SYMBOL}T ({baseline} - {compare}) (°C)"
                df_table_display[delta_col_name] = (temp_b - temp_c)
        
        for col in df_table_display.columns:
            if df_table_display[col].dtype in ['float64', 'int64']:
                 df_table_display[col] = df_table_display[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        return {"table": df_table_display, "chart_data": df_table_numeric, "conclusion_data": conclusion_data}
    except Exception as e: return {"error": f"分析時發生錯誤: {e}"}

# --- ======================================================================= ---
# ---               COBRA REPORTING & EXPORT FUNCTIONS                      ---
# --- ======================================================================= ---

def generate_formatted_table_image(df_table):
    if df_table.empty:
        fig, ax = plt.subplots(figsize=(8, 1)); ax.text(0.5, 0.5, "No data to display.", ha="center", va="center"); ax.axis('off'); return fig
    
    df_plot = df_table.reset_index()
    column_labels = df_plot.columns.tolist()
    wrapped_column_labels = [textwrap.fill(label, width=15) for label in column_labels]
    
    num_rows = len(df_plot)
    header_max_lines = max(label.count('\n') + 1 for label in wrapped_column_labels)
    fig_height = (num_rows * 0.4) + (header_max_lines * 0.4) + 0.5
    fig_width = 2.5 + len(column_labels) * 1.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off'); ax.axis('tight')
    
    table = ax.table(cellText=df_plot.values, colLabels=wrapped_column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10)
    
    cells = table.get_celld()
    for (row, col), cell in cells.items():
        cell.set_edgecolor('black')
        if row == 0:
            cell.set_facecolor('#606060'); cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#F0F0F0' if row % 2 != 0 else 'white')
            if column_labels[col] == 'Result':
                text = cell.get_text().get_text()
                if text == 'PASS': cell.set_facecolor(PASS_COLOR_HEX)
                elif text == 'FAIL': cell.set_facecolor(FAIL_COLOR_HEX)
    fig.tight_layout(pad=0.2)
    return fig

def create_formatted_excel(df_table):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sheet_name = 'ThermalTableData'
        df_to_export = df_table.reset_index()
        
        for col in df_to_export.columns:
            if col != 'Component' and col != 'Result' and not col.startswith(f"{DELTA_SYMBOL}T"):
                 df_to_export[col] = pd.to_numeric(df_to_export[col], errors='ignore')
        
        df_to_export.to_excel(writer, sheet_name=sheet_name, index=False)
        
        workbook, worksheet = writer.book, writer.sheets[sheet_name]
        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#606060', 'font_color': 'white', 'border': 1})
        pass_format = workbook.add_format({'bg_color': PASS_COLOR_HEX, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        fail_format = workbook.add_format({'bg_color': FAIL_COLOR_HEX, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        
        for col_num, value in enumerate(df_to_export.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        try:
            result_col_idx = df_to_export.columns.get_loc("Result")
            result_col_letter = chr(ord('A') + result_col_idx)
            worksheet.conditional_format(f'{result_col_letter}2:{result_col_letter}{len(df_to_export)+1}', {'type': 'cell', 'criteria': '==', 'value': '"PASS"', 'format': pass_format})
            worksheet.conditional_format(f'{result_col_letter}2:{result_col_letter}{len(df_to_export)+1}', {'type': 'cell', 'criteria': '==', 'value': '"FAIL"', 'format': fail_format})
        except KeyError: pass
            
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:Z', 18)
    output.seek(0)
    return output

# --- ======================================================================= ---
# ---                       APPLICATION UI FUNCTIONS                          ---
# --- ======================================================================= ---

def render_viper_ui():
    viper_logo_svg = """...""" # Omitted for brevity
    st.markdown(f"""
        <div style="display: flex; align-items: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px;">
            <div style="margin-right: 15px;">{viper_logo_svg}</div>
            <div>
                <h1 style="margin-bottom: 0; color: #FFFFFF;">Viper</h1>
                <p style="margin-top: 0; color: #AAAAAA;">Risk Analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    natural_convection_materials = {"塑膠 (ABS/PC)": {"emissivity": 0.90, "k_uniform": 0.65}, "鋁合金 (陽極處理)": {"emissivity": 0.85, "k_uniform": 0.90}}
    solar_absorptivity_materials = {"白色 (White Paint)": {"absorptivity": 0.25}, "銀色 (Silver Paint)": {"absorptivity": 0.40}, "深灰色 (Dark Gray)": {"absorptivity": 0.80}, "黑色 (Black Plastic/Paint)": {"absorptivity": 0.95}}

    tab_nat, tab_force, tab_solar = st.tabs(["🍃 自然對流分析", "🌬️ 強制對流分析", "☀️ 太陽輻射分析"])
    
    with tab_nat:
        # ... Full Viper UI code ...
        pass
    with tab_force:
        # ... Full Viper UI code ...
        pass
    with tab_solar:
        # ... Full Viper UI code ...
        pass

def render_cobra_ui():
    cobra_logo_svg = """...""" # Omitted for brevity
    st.markdown(f"""
        <div style="display: flex; align-items: center; padding-bottom: 10px; margin-bottom: 20px;">
            <div style="margin-right: 15px;">{cobra_logo_svg}</div>
            <div>
                <h1 style="margin-bottom: -15px; color: #FFFFFF;">Cobra</h1>
                <p style="margin-top: 0; color: #AAAAAA;">Data Transformation</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("上傳 Excel 檔", type=["xlsx", "xls"], key="cobra_file_uploader")

    if 'cobra_prestudy_data' not in st.session_state: st.session_state.cobra_prestudy_data = {}
    if 'cobra_analysis_results' not in st.session_state: st.session_state.cobra_analysis_results = None
    if 'delta_t_pairs' not in st.session_state: st.session_state.delta_t_pairs = []
    
    if uploaded_file and st.session_state.get('cobra_filename') != uploaded_file.name:
        st.session_state.cobra_filename = uploaded_file.name
        with st.spinner('預先分析檔案中...'):
            st.session_state.cobra_prestudy_data = cobra_pre_study(uploaded_file)
            st.session_state.cobra_analysis_results = None
            st.session_state.delta_t_pairs = []
            if 'spec_df' in st.session_state: del st.session_state.spec_df
    
    cobra_data = st.session_state.get('cobra_prestudy_data', {})

    if not cobra_data.get("series_names"):
        st.info("請上傳 Excel 檔以開始分析。"); return
    if cobra_data.get("error"):
        st.error(cobra_data["error"]); return
        
    st.subheader("分析參數設定")
    
    selection_container = st.container(border=True)
    selection_col1, selection_col2 = selection_container.columns(2, gap="large")

    with selection_col1:
        st.write("**1. 選擇分析設定**")
        with st.container(height=250):
            selected_series = [name for name in cobra_data["series_names"] if st.checkbox(name, value=True, key=f"series_{name}")]
    with selection_col2:
        st.write("**2. 選擇關鍵 IC**")
        with st.container(height=250):
            selected_ics = [name for name in cobra_data["component_names"] if st.checkbox(name, key=f"ic_{name}")]

    with selection_container:
        st.write(f"**3. {DELTA_SYMBOL}T 比較 (可選)**")
        
        for i, pair in enumerate(st.session_state.delta_t_pairs):
            pair_cols = st.columns([2, 2, 1])
            baseline = pair_cols[0].selectbox(f"基準 (Baseline):", [NO_COMPARISON_LABEL] + selected_series, key=f"delta_b_{i}")
            compare = pair_cols[1].selectbox(f"比較對象:", [NO_COMPARISON_LABEL] + selected_series, key=f"delta_c_{i}")
            if pair_cols[2].button("移除", key=f"remove_delta_{i}"):
                st.session_state.delta_t_pairs.pop(i)
                st.rerun()
            st.session_state.delta_t_pairs[i] = {'baseline': baseline, 'compare': compare}

        if st.button("新增 ΔT 比較"):
            st.session_state.delta_t_pairs.append({'baseline': NO_COMPARISON_LABEL, 'compare': NO_COMPARISON_LABEL})
            st.rerun()

    spec_df = None
    if selected_ics:
        st.subheader("4. 關鍵 IC 規格輸入")
        if 'spec_df' not in st.session_state or set(st.session_state.spec_df['Component']) != set(selected_ics):
            spec_data = [{"Component": ic, "Spec Type": SPEC_TYPE_TC_CALC, "Tj (°C)": None, "Rjc (°C/W)": None, "Pd (W)": None, "Ta Limit (°C)": None} for ic in selected_ics]
            st.session_state.spec_df = pd.DataFrame(spec_data)
        
        edited_specs_df = st.data_editor(st.session_state.spec_df, key="spec_editor", hide_index=True, use_container_width=True,
            column_config={
                "Spec Type": st.column_config.SelectboxColumn("Spec Type", options=SPEC_TYPES, required=True),
                "Component": st.column_config.TextColumn("Component", disabled=True),
                "Tj (°C)": st.column_config.NumberColumn("Tj (°C)", format="%.2f"),
                "Rjc (°C/W)": st.column_config.NumberColumn("Rjc (°C/W)", format="%.2f"),
                "Pd (W)": st.column_config.NumberColumn("Pd (W)", format="%.2f"),
                "Ta Limit (°C)": st.column_config.NumberColumn("Ta Limit (°C)", format="%.2f"),
            })
        spec_df = edited_specs_df

    st.divider()
    if st.button("🚀 分析選定資料", use_container_width=True, type="primary"):
        if not selected_series or not selected_ics: st.warning("請至少選擇一個分析設定和一個關鍵 IC。")
        else:
            delta_pairs_for_analysis = [pair for pair in st.session_state.delta_t_pairs if pair['baseline'] != NO_COMPARISON_LABEL and pair['compare'] != NO_COMPARISON_LABEL]
            with st.spinner("處理資料中..."):
                st.session_state.cobra_analysis_results = run_cobra_analysis(uploaded_file, cobra_data, selected_series, selected_ics, spec_df, delta_pairs_for_analysis)

    if st.session_state.get('cobra_analysis_results'):
        results = st.session_state.cobra_analysis_results
        if results.get("error"): st.error(f"**分析錯誤:** {results['error']}")
        else:
            st.header("分析結果")
            res_tab1, res_tab2, res_tab3 = st.tabs(["**結論**", "**表格**", "**圖表**"])
            with res_tab1:
                render_structured_conclusions(results.get("conclusion_data", []))
            with res_tab2: 
                st.subheader("格式化數據表格")
                table_fig = generate_formatted_table_image(results.get("table"))
                st.pyplot(table_fig)
                
                img_buf = io.BytesIO(); table_fig.savefig(img_buf, format="png", dpi=300, bbox_inches='tight')
                excel_buf = create_formatted_excel(results.get("table"))

                btn_col1, btn_col2 = st.columns(2)
                btn_col1.download_button("下載表格 (PNG)", data=img_buf, file_name="cobra_table.png", mime="image/png", use_container_width=True)
                btn_col2.download_button("下載為格式化 Excel", data=excel_buf, file_name="cobra_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with res_tab3: 
                st.subheader("關鍵 IC 溫度比較圖")
                chart_data_numeric = results.get("chart_data")
                
                fig_chart, ax = plt.subplots(figsize=(max(10, len(chart_data_numeric.index) * 0.8), 6))
                chart_data_numeric[[s for s in selected_series if s in chart_data_numeric.columns]].plot(kind='bar', ax=ax, width=0.8)
                ax.set_ylabel("Temperature (°C)")
                ax.set_title("Key IC Temperature Comparison")
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig_chart)
                
                chart_buf = io.BytesIO(); fig_chart.savefig(chart_buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button("下載圖表 (PNG)", data=chart_buf, file_name="cobra_chart.png", mime="image/png", use_container_width=True)

def render_structured_conclusions(conclusion_data):
    st.subheader("Executive Summary")
    failed_ics = [item['component'] for item in conclusion_data if item['result'] == 'FAIL']
    if failed_ics:
        st.markdown(f"**結果: <span style='color:red;'>失敗</span>** - 以下元件超出溫度規格: **{', '.join(failed_ics)}**", unsafe_allow_html=True)
    else:
        st.markdown(f"**結果: <span style='color:lightgreen;'>通過</span>** - 所有選定的關鍵 IC 均符合其溫度規格。", unsafe_allow_html=True)
    
    st.divider()
    st.subheader("詳細元件分析")

    for item in conclusion_data:
        result_text = item['result']
        color = "red" if result_text == "FAIL" else "lightgreen"
        with st.expander(f"**{item['component']}** - 結果: {result_text}"):
            spec_val = f"{item['spec']:.2f}°C" if pd.notna(item['spec']) else "N/A"
            st.markdown(f"**規格類型:** `{item['spec_type']}`")
            st.markdown(f"**計算規格上限:** `{spec_val}`")
            if item.get('spec_inputs') != 'N/A':
                st.markdown(f"**規格輸入:** `{item['spec_inputs']}`")
            
            st.write("**各設定下的表現:**")
            
            if item['series_results']:
                series_results_df = pd.DataFrame(item['series_results'])
                if 'temp' in series_results_df.columns:
                    series_results_df['temp'] = series_results_df['temp'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                st.dataframe(series_results_df.rename(columns={'series': 'Configuration', 'temp': 'Temp (°C)', 'result': 'Result'}), use_container_width=True, hide_index=True)
            else:
                st.caption("無溫度數據可顯示 (例如, 未定義規格)。")

# --- ======================================================================= ---
# ---                           MAIN APP ROUTER                             ---
# --- ======================================================================= ---
st.set_page_config(page_title="Sercomm Tool Suite", layout="wide")
st.sidebar.title("Sercomm Thermal Engineering")
app_selection = st.sidebar.radio("選擇工具:", ("Viper - Risk analysis", "Cobra - Data transformation"))
st.sidebar.markdown("---")
st.sidebar.info("一個整合性的工程分析工具平台。")

if app_selection == "Viper - Risk analysis":
    render_viper_ui()
elif app_selection == "Cobra - Data transformation":
    render_cobra_ui()
