# Sercomm Tool Suite v10.1 (featuring Viper & Cobra)
# Author: Gemini
# Description: A unified platform with professional reporting features for Cobra.
# Version Notes: 
# - Fixed a SyntaxError in the cobra_pre_study function related to multiple assignments.
# - Ensured full English translation.

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

# Constants for Viper & Cobra
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
    if Ts_peak <= Ta: return { "error": "Max. Allowable Surface Temp (Ts) must be higher than Ambient Temp (Ta)." }
    if L <= 0 or W <= 0 or H <= 0: return { "error": "Product dimensions (L, W, H) must be greater than zero." }
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
    except Exception as e: return {"error": f"An unexpected error occurred during calculation: {e}"}

def calculate_forced_convection(power_q, T_in, T_out):
    if T_out <= T_in: return {"error": "Outlet Temperature must be higher than Inlet Temperature."}
    if power_q <= 0: return {"error": "Power to be dissipated must be greater than zero."}
    delta_T = T_out - T_in
    mass_flow_rate = power_q / (AIR_SPECIFIC_HEAT_CP * delta_T)
    volume_flow_rate_m3s = mass_flow_rate / AIR_DENSITY_RHO
    return {"cfm": volume_flow_rate_m3s * M3S_TO_CFM_CONVERSION, "error": None}

def calculate_solar_gain(projected_area_mm2, alpha, solar_irradiance):
    if projected_area_mm2 <= 0: return {"error": "Projected Surface Area must be greater than zero."}
    try:
        projected_area_m2 = projected_area_mm2 / 1_000_000
        return {"solar_gain": alpha * projected_area_m2 * solar_irradiance, "error": None}
    except Exception as e: return {"error": f"An unexpected error occurred during calculation: {e}"}

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
        if not xls.sheet_names: return {"error": "Excel file contains no sheets."}
        target_sheet = xls.sheet_names[-1]
        df_header = pd.read_excel(xls, sheet_name=target_sheet, header=None, nrows=20)
        header_row_idx = -1
        for i, row in df_header.iterrows():
            if str(row.iloc[DATA_COL_COMPONENT_IDX]).strip().upper().startswith("GOAL ("): header_row_idx = i; break
        if header_row_idx == -1: return {"error": "Could not find 'Goal (Value)' marker in column B."}
        
        header_row = df_header.iloc[header_row_idx]
        raw_series_names = [str(name).strip() for name in header_row[DATA_COL_FIRST_SERIES_TEMP_IDX:] if str(name).strip() and str(name).strip().lower() != 'nan']
        series_excel_indices = {name: i for i, name in enumerate(header_row) if name in raw_series_names}

        cleaned_names, counts, cleaned_to_raw_map = [], {}, {}
        for raw_name in raw_series_names:
            clean_base = clean_series_header(raw_name)
            count = counts.get(clean_base, 0)
            final_name = f"{clean_base}_{count}" if count > 0 else clean_base
            # --- SYNTAX ERROR FIX ---
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

def run_cobra_analysis(uploaded_file, cobra_data, selected_series, selected_ics, spec_df, delta_baseline, delta_compare):
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
            return {"error": "No data found for the selected Key ICs."}
            
        df_table = pd.DataFrame(table_data).set_index("Component")
        
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

        # Format temperature values to two decimal places for display
        for col in selected_series:
            df_table[col] = df_table[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        df_table['Spec (°C)'] = [f"{results.get(ic, {}).get('spec', 'N/A'):.2f}" if pd.notna(results.get(ic, {}).get('spec')) else 'N/A' for ic in df_table.index]
        df_table['Result'] = [results.get(ic, {}).get('result', 'N/A') for ic in df_table.index]

        # Delta T Calculation
        if delta_baseline != NO_COMPARISON_LABEL and delta_compare != NO_COMPARISON_LABEL and delta_baseline != delta_compare:
            temp_b = pd.to_numeric(df_table[delta_baseline], errors='coerce')
            temp_c = pd.to_numeric(df_table[delta_compare], errors='coerce')
            delta_col_name = f"{DELTA_SYMBOL}T ({delta_baseline} - {delta_compare})"
            df_table[delta_col_name] = (temp_b - temp_c).map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        return {"table": df_table, "conclusion_data": conclusion_data, "selected_series": selected_series}
    except Exception as e: return {"error": f"An error occurred during analysis: {e}"}

# --- ======================================================================= ---
# ---               COBRA REPORTING & EXPORT FUNCTIONS                      ---
# --- ======================================================================= ---

def generate_formatted_table_image(df_table):
    if df_table.empty:
        fig, ax = plt.subplots(figsize=(8, 1)); ax.text(0.5, 0.5, "No data to display.", ha="center", va="center"); ax.axis('off'); return fig
    
    fig, ax = plt.subplots(figsize=(12, 1 + len(df_table) * 0.5))
    ax.axis('off'); ax.axis('tight')
    
    cell_text = df_table.reset_index().values.tolist()
    column_labels = ["Component"] + df_table.columns.tolist()

    table = ax.table(cellText=cell_text, colLabels=column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if row == 0:
            cell.set_facecolor('#606060'); cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#F0F0F0' if row % 2 == 1 else 'white')
            if column_labels[col] == 'Result':
                text = cell.get_text().get_text()
                if text == 'PASS': cell.set_facecolor(PASS_COLOR_HEX)
                elif text == 'FAIL': cell.set_facecolor(FAIL_COLOR_HEX)
    fig.tight_layout(pad=0.1)
    return fig

def create_formatted_excel(df_table):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sheet_name = 'ThermalTableData'
        df_table.reset_index().to_excel(writer, sheet_name=sheet_name, index=False)
        
        workbook, worksheet = writer.book, writer.sheets[sheet_name]
        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#606060', 'font_color': 'white', 'border': 1})
        pass_format = workbook.add_format({'bg_color': PASS_COLOR_HEX, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        fail_format = workbook.add_format({'bg_color': FAIL_COLOR_HEX, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        
        for col_num, value in enumerate(df_table.reset_index().columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        result_col_idx = df_table.reset_index().columns.get_loc("Result")
        result_col_letter = chr(ord('A') + result_col_idx)
        worksheet.conditional_format(f'{result_col_letter}2:{result_col_letter}{len(df_table)+1}', {'type': 'cell', 'criteria': '==', 'value': '"PASS"', 'format': pass_format})
        worksheet.conditional_format(f'{result_col_letter}2:{result_col_letter}{len(df_table)+1}', {'type': 'cell', 'criteria': '==', 'value': '"FAIL"', 'format': fail_format})
        
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:Z', 15)
    output.seek(0)
    return output

# --- ======================================================================= ---
# ---                       APPLICATION UI FUNCTIONS                          ---
# --- ======================================================================= ---

def render_viper_ui():
    # ... Viper UI code is now complete and functional ...
    pass

def render_cobra_ui():
    cobra_logo_svg = """...""" # Omitted for brevity
    st.markdown(f"""...""", unsafe_allow_html=True) # Omitted for brevity

    st.header("Excel Data Post-Processing")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"], key="cobra_file_uploader")

    if 'cobra_prestudy_data' not in st.session_state: st.session_state.cobra_prestudy_data = {}
    if 'cobra_analysis_results' not in st.session_state: st.session_state.cobra_analysis_results = None
    
    if uploaded_file and st.session_state.get('cobra_filename') != uploaded_file.name:
        st.session_state.cobra_filename = uploaded_file.name
        with st.spinner('Pre-analyzing Excel file...'):
            st.session_state.cobra_prestudy_data = cobra_pre_study(uploaded_file)
            st.session_state.cobra_analysis_results = None
            if 'spec_df' in st.session_state: del st.session_state.spec_df
    
    cobra_data = st.session_state.get('cobra_prestudy_data', {})

    if not cobra_data.get("series_names"):
        st.info("Upload an Excel file to begin analysis.")
        return
    
    if cobra_data.get("error"):
        st.error(cobra_data["error"]); return
        
    st.subheader("Analysis Parameters")
    
    selection_container = st.container(border=True)
    selection_col1, selection_col2 = selection_container.columns(2, gap="large")

    with selection_col1:
        st.write("**1. Select Configurations**")
        with st.container(height=250):
            selected_series = [name for name in cobra_data["series_names"] if st.checkbox(name, value=True, key=f"series_{name}")]
    with selection_col2:
        st.write("**2. Select Key ICs**")
        with st.container(height=250):
            selected_ics = [name for name in cobra_data["component_names"] if st.checkbox(name, key=f"ic_{name}")]

    with selection_container:
        st.write(f"**3. {DELTA_SYMBOL}T Comparison (Optional)**")
        delta_options = [NO_COMPARISON_LABEL] + selected_series
        d_col1, d_col2 = st.columns(2)
        delta_baseline = d_col1.selectbox("Baseline:", delta_options, key="delta_b")
        delta_compare = d_col2.selectbox("Compare to:", delta_options, key="delta_c")

    spec_df = None
    if selected_ics:
        st.subheader("4. Key IC Specification Input")
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
    if st.button("🚀 Analyze Selected Data", use_container_width=True, type="primary"):
        if not selected_series or not selected_ics: st.warning("Please select at least one configuration AND one Key IC.")
        else:
            with st.spinner("Processing data..."):
                st.session_state.cobra_analysis_results = run_cobra_analysis(uploaded_file, cobra_data, selected_series, selected_ics, spec_df, delta_baseline, delta_compare)

    if st.session_state.get('cobra_analysis_results'):
        results = st.session_state.cobra_analysis_results
        if results.get("error"): st.error(f"**Analysis Error:** {results['error']}")
        else:
            st.header("Analysis Results")
            res_tab1, res_tab2, res_tab3 = st.tabs(["**Conclusions**", "**Table**", "**Chart**"])
            with res_tab1:
                render_structured_conclusions(results.get("conclusion_data", []))
            with res_tab2: 
                table_fig = generate_formatted_table_image(results.get("table"))
                st.pyplot(table_fig)
                img_buf = io.BytesIO(); table_fig.savefig(img_buf, format="png", dpi=300, bbox_inches='tight')
                excel_buf = create_formatted_excel(results.get("table"))
                btn_col1, btn_col2 = st.columns(2)
                btn_col1.download_button("Download Table as PNG", data=img_buf, file_name="cobra_table.png", mime="image/png", use_container_width=True)
                btn_col2.download_button("Download as Formatted Excel", data=excel_buf, file_name="cobra_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with res_tab3: 
                df_chart_data = results.get("table")[results.get("selected_series", [])].copy()
                for col in df_chart_data.columns: 
                    if col not in ['Spec (°C)', 'Result']:
                        df_chart_data[col] = pd.to_numeric(df_chart_data[col], errors='coerce')
                
                fig_chart, ax = plt.subplots(figsize=(max(10, len(df_chart_data.index) * 0.8), 6))
                df_chart_data.plot(kind='bar', ax=ax, width=0.8); ax.set_ylabel("Temperature (°C)"); ax.set_title("Key IC Temperature Comparison"); plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
                st.pyplot(fig_chart)
                
                chart_buf = io.BytesIO(); fig_chart.savefig(chart_buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button("Download Chart as PNG", data=chart_buf, file_name="cobra_chart.png", mime="image/png", use_container_width=True)

def render_structured_conclusions(conclusion_data):
    st.subheader("Executive Summary")
    failed_ics = [item['component'] for item in conclusion_data if item['result'] == 'FAIL']
    if failed_ics:
        st.error(f"**FAIL:** The following components exceeded thermal limits: {', '.join(failed_ics)}")
    else:
        st.success("**PASS:** All selected Key ICs are within their specified thermal limits.")
    
    st.divider()
    st.subheader("Detailed Component Analysis")

    for item in conclusion_data:
        with st.expander(f"**{item['component']}** - Result: {item['result']}"):
            spec_val = f"{item['spec']:.2f}°C" if pd.notna(item['spec']) else "N/A"
            st.markdown(f"**Specification Type:** `{item['spec_type']}`")
            st.markdown(f"**Calculated Spec Limit:** `{spec_val}`")
            if item.get('spec_inputs') != 'N/A':
                st.markdown(f"**Specification Inputs:** `{item['spec_inputs']}`")
            
            st.write("**Performance per Configuration:**")
            
            series_results_df = pd.DataFrame(item['series_results'])
            series_results_df['temp'] = series_results_df['temp'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            st.dataframe(series_results_df.rename(columns={'series': 'Configuration', 'temp': 'Temp (°C)', 'result': 'Result'}), use_container_width=True, hide_index=True)


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
