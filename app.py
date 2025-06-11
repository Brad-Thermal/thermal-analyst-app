# Sercomm Tool Suite v19.4
# Author: Gemini
# Description: A unified platform with professional reporting features.
# Version Notes:
# - v19.4: CRITICAL FIX: Corrected the 'Remove File' button logic to avoid a StreamlitAPIException by removing direct manipulation of the file_uploader state.
# - v19.3: Added a 'Remove File' button to the Cobra UI to clear all loaded data and reset the state.
# - v19.2: Added data labels to the top of each bar in the Cobra results chart for readability.
# - v19.2: Added 'Surface Area' output (in m²) to the Viper natural convection results.

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
# ---                          CALCULATION ENGINES                            ---
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
        return {"total_power": Q_final, "surface_area": A_total, "error": None}
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
# ---                       COBRA DATA PROCESSING LOGIC                       ---
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
            final_name = f"{clean_base}_{count+1}" if clean_base in counts else clean_base
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

def run_cobra_analysis(file_buffer, cobra_data, selected_series, selected_ics, spec_df, delta_pairs):
    try:
        df_full = pd.read_excel(file_buffer, sheet_name=cobra_data['target_sheet'], header=None, dtype=str)
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

        df_table_numeric = pd.DataFrame(table_data).set_index("Component")
        df_table_display = df_table_numeric.copy()

        results, conclusion_data = {}, []
        if spec_df is not None:
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
        if results:
            df_table_display['Spec (°C)'] = [f"{results.get(ic, {}).get('spec_type', '')} = {results.get(ic, {}).get('spec', 'N/A'):.2f}" if pd.notna(results.get(ic, {}).get('spec')) else "N/A" for ic in df_table_display.index]
            df_table_display['Result'] = [results.get(ic, {}).get('result', 'N/A') for ic in df_table_display.index]

        for pair in delta_pairs:
            baseline, compare = pair['baseline'], pair['compare']
            if baseline != NO_COMPARISON_LABEL and compare != NO_COMPARISON_LABEL and baseline != compare:
                if f"{baseline} (°C)" in df_table_display.columns and f"{compare} (°C)" in df_table_display.columns:
                    temp_b = pd.to_numeric(df_table_display[f"{baseline} (°C)"], errors='coerce')
                    temp_c = pd.to_numeric(df_table_display[f"{compare} (°C)"], errors='coerce')
                    delta_col_name = f"{DELTA_SYMBOL}T ({baseline} - {compare}) (°C)"
                    df_table_display[delta_col_name] = (temp_b - temp_c)

        spec_col = 'Spec (°C)'
        result_col = 'Result'
        if spec_col in df_table_display.columns:
            temp_cols = [col for col in df_table_display.columns if col.endswith(' (°C)') and col not in [spec_col, result_col] and not col.startswith(DELTA_SYMBOL)]
            delta_cols = sorted([col for col in df_table_display.columns if col.startswith(DELTA_SYMBOL)])
            
            new_order = [spec_col] + temp_cols + delta_cols
            if result_col in df_table_display.columns:
                new_order.append(result_col)
            
            df_table_display = df_table_display[new_order]

        df_for_excel = df_table_display.copy()
        
        df_for_image = df_table_display.copy()
        for col in df_for_image.columns:
            if df_for_image[col].dtype in ['float64', 'int64']:
                df_for_image[col] = df_for_image[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        return {"table_for_image": df_for_image, "table_for_excel": df_for_excel, "chart_data": df_table_numeric, "conclusion_data": conclusion_data}
    except Exception as e:
        return {"error": f"An error occurred during analysis: {e}"}


# --- ======================================================================= ---
# ---                   COBRA REPORTING & EXPORT FUNCTIONS                    ---
# --- ======================================================================= ---

def generate_formatted_table_image(df_table):
    if df_table.empty:
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.text(0.5, 0.5, "No data to display.", ha="center", va="center")
        ax.axis('off')
        return fig

    df_plot = df_table.reset_index()
    column_labels = df_plot.columns.tolist()
    wrapped_column_labels = [textwrap.fill(label, width=15) for label in column_labels]

    fig_width = 3.0 * len(column_labels) 
    
    fig, ax = plt.subplots(figsize=(fig_width, 1))
    ax.axis('off')

    table = ax.table(cellText=df_plot.values, colLabels=wrapped_column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(13.5)

    cells = table.get_celld()
    for (row, col), cell in cells.items():
        cell.set_edgecolor('black')
        if row == 0:
            cell.set_height(0.5)
            cell.set_facecolor('#606060')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_height(0.5)
            cell.set_facecolor('#F0F0F0' if row % 2 != 0 else 'white')
            if 'Result' in df_plot.columns and column_labels[col] == 'Result':
                text = cell.get_text().get_text()
                if text == 'PASS':
                    cell.set_facecolor(PASS_COLOR_HEX)
                elif text == 'FAIL':
                    cell.set_facecolor(FAIL_COLOR_HEX)
    
    fig.canvas.draw()
    bbox = table.get_window_extent(ax.figure.canvas.get_renderer())
    fig.set_size_inches(fig_width, bbox.height/fig.dpi + 0.75)

    fig.tight_layout()
    return fig

def create_formatted_excel(df_table):
    output = io.BytesIO()
    df_to_export = df_table.reset_index()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sheet_name = 'ThermalTableData'
        workbook = writer.book
        worksheet = workbook.add_worksheet(sheet_name)

        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#606060', 'font_color': 'white', 'border': 1})
        pass_format = workbook.add_format({'bg_color': PASS_COLOR_HEX, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        fail_format = workbook.add_format({'bg_color': FAIL_COLOR_HEX, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
        
        for col_num, value in enumerate(df_to_export.columns.values):
            worksheet.write(0, col_num, value, header_format)

        for row_num, row_data in df_to_export.iterrows():
            for col_num, cell_value in enumerate(row_data):
                col_name = df_to_export.columns[col_num]
                
                current_format = center_format
                if col_name == 'Result':
                    if cell_value == 'PASS':
                        current_format = pass_format
                    elif cell_value == 'FAIL':
                        current_format = fail_format
                
                try:
                    num_value = float(cell_value)
                    if pd.notna(num_value):
                        worksheet.write_number(row_num + 1, col_num, num_value, current_format)
                    else:
                        worksheet.write_blank(row_num + 1, col_num, None, current_format)
                except (ValueError, TypeError):
                    worksheet.write_string(row_num + 1, col_num, str(cell_value), current_format)

        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:Z', 18)

    output.seek(0)
    return output


# --- ======================================================================= ---
# ---                        APPLICATION UI FUNCTIONS                         ---
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
                <h1 style="margin-bottom: 0; color: #FFFFFF;">Viper</h1>
                <p style="margin-top: 0; color: #AAAAAA;">Risk Analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    natural_convection_materials = {"Plastic (ABS/PC)": {"emissivity": 0.90, "k_uniform": 0.65}, "Aluminum (Anodized)": {"emissivity": 0.85, "k_uniform": 0.90}}
    solar_absorptivity_materials = {"White (Paint)": {"absorptivity": 0.25}, "Silver (Paint)": {"absorptivity": 0.40}, "Dark Gray": {"absorptivity": 0.80}, "Black (Plastic/Paint)": {"absorptivity": 0.95}}

    tab_nat, tab_force, tab_solar = st.tabs(["🍃 Natural Convection", "🌬️ Forced Convection", "☀️ Solar Radiation"])

    with tab_nat:
        st.header("Passive Cooling Power Estimator")
        col_nat_input, col_nat_result = st.columns(2, gap="large")
        with col_nat_input:
            st.subheader("Input Parameters")
            nc_material_name = st.selectbox("Enclosure Material", options=list(natural_convection_materials.keys()), key="nc_mat")
            st.markdown("**Product Dimensions (mm)**")
            dim_col1, dim_col2, dim_col3 = st.columns(3)
            with dim_col1: nc_dim_L = st.number_input("Length (L)", 1.0, 1000.0, 200.0, 10.0, "%.1f", key="nc_l")
            with dim_col2: nc_dim_W = st.number_input("Width (W)", 1.0, 1000.0, 150.0, 10.0, "%.1f", key="nc_w")
            with dim_col3: nc_dim_H = st.number_input("Height (H)", 1.0, 500.0, 50.0, 5.0, "%.1f", key="nc_h")
            st.markdown("**Operating Conditions (°C)**")
            op_cond_col1, op_cond_col2 = st.columns(2)
            with op_cond_col1: nc_temp_ambient = st.number_input("Ambient Temp (Ta)", 0, 60, 25, key="nc_ta")
            with op_cond_col2: nc_temp_surface_peak = st.number_input("Max. Surface Temp (Ts)", nc_temp_ambient + 1, 100, 50, key="nc_ts")
        with col_nat_result:
            st.subheader("Evaluation Result")
            selected_material_props_nc = natural_convection_materials[nc_material_name]
            nc_results = calculate_natural_convection(nc_dim_L, nc_dim_W, nc_dim_H, nc_temp_surface_peak, nc_temp_ambient, selected_material_props_nc)
            if nc_results.get("error"): 
                st.error(f"**Error:** {nc_results['error']}")
            else:
                st.metric(label="📐 Surface Area", value=f"{nc_results['surface_area']:.4f} m²")
                st.metric(label="✅ Max. Dissipatable Power", value=f"{nc_results['total_power']:.2f} W", help="This result includes built-in material uniformity and a fixed engineering safety factor (0.9).")

    with tab_force:
        st.header("Active Cooling Airflow Estimator")
        col_force_input, col_force_result = st.columns(2, gap="large")
        with col_force_input:
            st.subheader("Input Parameters")
            fc_power_q = st.number_input("Power to Dissipate (Q, W)", 0.1, value=50.0, step=1.0, format="%.1f", help="The total heat (in Watts) that the fan must remove.")
            fc_temp_in = st.number_input("Inlet Air Temp (Tin, °C)", 0, 60, 25, key="fc_tin")
            fc_temp_out = st.number_input("Max. Outlet Temp (Tout, °C)", fc_temp_in + 1, 100, 45, key="fc_tout")
            st.subheader("Governing Equation")
            st.latex(r"Q = \dot{m} \cdot C_p \cdot \Delta T")
        with col_force_result:
            st.subheader("Evaluation Result")
            fc_results = calculate_forced_convection(fc_power_q, fc_temp_in, fc_temp_out)
            if fc_results.get("error"): st.error(f"**Error:** {fc_results['error']}")
            else: st.metric(label="🌬️ Required Airflow", value=f"{fc_results['cfm']:.2f} CFM", help="CFM: Cubic Feet per Minute.")

    with tab_solar:
        st.header("Solar Heat Gain Estimator")
        col_solar_input, col_solar_result = st.columns(2, gap="large")
        with col_solar_input:
            st.subheader("Input Parameters")
            solar_material_name = st.selectbox("Enclosure Color/Finish", options=list(solar_absorptivity_materials.keys()) + ["Other..."], key="solar_mat")
            if solar_material_name == "Other...":
                alpha_val = st.number_input("Custom Absorptivity (α)", 0.0, 1.0, 0.5, 0.05)
            else:
                alpha_val = solar_absorptivity_materials[solar_material_name]["absorptivity"]
                st.number_input("Corresponding Absorptivity (α)", value=alpha_val, disabled=True)
            projected_area_mm2 = st.number_input("Projected Surface Area (mm²)", 0.0, value=30000.0, step=1000.0, format="%.1f")
            solar_irradiance_val = st.number_input("Solar Irradiance (W/m²)", 0, value=1000, step=50)
            st.subheader("Governing Equation")
            st.latex(r"Q_{solar} = \alpha \cdot A_{proj} \cdot G_{solar}")
        with col_solar_result:
            st.subheader("Evaluation Result")
            solar_results = calculate_solar_gain(projected_area_mm2, alpha_val, solar_irradiance_val)
            if solar_results.get("error"): st.error(f"**Error:** {solar_results['error']}")
            else: st.metric(label="☀️ Absorbed Solar Heat Gain", value=f"{solar_results['solar_gain']:.2f} W")

def render_cobra_ui():
    cobra_logo_svg = """
    <svg width="50" height="50" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M40 95 C 20 95, 10 75, 25 60 C 35 50, 65 50, 75 60 C 90 75, 80 95, 60 95 L 40 95 Z" fill="#2C7873"/>
        <path d="M50 70 C 25 70, 25 40, 50 20 C 75 40, 75 70, 50 70 Z" fill="#4A938E" stroke="#FFFFFF" stroke-width="2"/>
        <path d="M50 20 C 40 30, 40 50, 50 60 C 60 50, 60 30, 50 20" fill="#1E1E1E"/>
        <circle cx="46" cy="45" r="4" fill="red"/>
        <circle cx="54" cy="45" r="4" fill="red"/>
        <path d="M48 62 L52 62 L50 70 Z" fill="#FFD700"/>
    </svg>
    """
    st.markdown(f"""
        <div style="display: flex; align-items: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px;">
            <div style="margin-right: 15px;">{cobra_logo_svg}</div>
            <div>
                <h1 style="margin-bottom: 0; color: #FFFFFF;">Cobra</h1>
                <p style="margin-top: 0; color: #AAAAAA;">Data Transformation & Analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    main_cols = st.columns([0.4, 0.6], gap="large")

    with main_cols[0]:
        st.subheader("Upload Excel File")
        uploader_cols = st.columns([0.7, 0.3])
        uploaded_file = uploader_cols[0].file_uploader("Drag and drop file here", type=["xlsx", "xls"], key="cobra_file_uploader", label_visibility="collapsed")
        
        # FIX: "Remove File" button logic
        if uploader_cols[1].button("Remove", use_container_width=True):
            keys_to_clear = [
                'cobra_prestudy_data', 'cobra_analysis_results', 'delta_t_pairs', 
                'uploaded_file_buffer', 'cobra_filename', 'series_selection', 
                'ic_selection', 'spec_df'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            # No need to touch cobra_file_uploader, Streamlit handles it.
            st.rerun()


    if 'cobra_prestudy_data' not in st.session_state: st.session_state.cobra_prestudy_data = {}
    if 'cobra_analysis_results' not in st.session_state: st.session_state.cobra_analysis_results = None
    if 'delta_t_pairs' not in st.session_state: st.session_state.delta_t_pairs = []
    if 'uploaded_file_buffer' not in st.session_state: st.session_state.uploaded_file_buffer = None


    if uploaded_file and st.session_state.get('cobra_filename') != uploaded_file.name:
        st.session_state.cobra_filename = uploaded_file.name
        st.session_state.uploaded_file_buffer = uploaded_file.getvalue()
        with st.spinner('Pre-analyzing Excel file...'):
            st.session_state.cobra_prestudy_data = cobra_pre_study(io.BytesIO(st.session_state.uploaded_file_buffer))
            st.session_state.cobra_analysis_results = None
            st.session_state.delta_t_pairs = []
            if 'series_selection' in st.session_state: del st.session_state.series_selection
            if 'ic_selection' in st.session_state: del st.session_state.ic_selection
            if 'spec_df' in st.session_state: del st.session_state.spec_df
    
    cobra_data = st.session_state.get('cobra_prestudy_data', {})

    with main_cols[1]:
        if not cobra_data.get("series_names"):
            st.info("Please upload an Excel file to begin analysis.")
            st.stop()
        if cobra_data.get("error"):
            st.error(cobra_data["error"])
            st.stop()

        if 'series_selection' not in st.session_state:
            st.session_state.series_selection = cobra_data["series_names"]
        if 'ic_selection' not in st.session_state:
            st.session_state.ic_selection = []
        
        st.subheader("Analysis Parameters")
        tab1, tab2, tab3 = st.tabs(["1. Select Configurations", "2. Select Key ICs", f"3. {DELTA_SYMBOL}T Comparison (Optional)"])

        with tab1:
            st.write("Select configurations to analyze:")
            btn_cols = st.columns(2)
            if btn_cols[0].button("Select All", key="select_all_series", use_container_width=True):
                st.session_state.series_selection = cobra_data["series_names"]
            if btn_cols[1].button("Clear All", key="deselect_all_series", use_container_width=True):
                st.session_state.series_selection = []
            
            selected_series = st.multiselect(
                "Select configurations to analyze (multiple allowed):",
                options=cobra_data["series_names"],
                key="series_selection",
                label_visibility="collapsed"
            )

        with tab2:
            st.write("Select Key ICs to analyze:")
            btn_cols = st.columns(2)
            if btn_cols[0].button("Select All", key="select_all_ics", use_container_width=True):
                st.session_state.ic_selection = cobra_data["component_names"]
            if btn_cols[1].button("Clear All", key="deselect_all_ics", use_container_width=True):
                st.session_state.ic_selection = []

            selected_ics = st.multiselect(
                "Select Key ICs to analyze (multiple allowed):",
                options=cobra_data["component_names"],
                key="ic_selection",
                label_visibility="collapsed"
            )
            
        with tab3:
            st.write(f"Set up {DELTA_SYMBOL}T comparisons (optional):")
            with st.container(height=280):
                for i, pair in enumerate(st.session_state.delta_t_pairs):
                    pair_cols = st.columns([2, 2, 1])
                    baseline_options = [NO_COMPARISON_LABEL] + selected_series if selected_series else [NO_COMPARISON_LABEL]
                    baseline = pair_cols[0].selectbox(f"Baseline:", baseline_options, key=f"delta_b_{i}")
                    compare = pair_cols[1].selectbox(f"Compare to:", baseline_options, key=f"delta_c_{i}")
                    if pair_cols[2].button("Remove", key=f"remove_delta_{i}"):
                        st.session_state.delta_t_pairs.pop(i)
                        st.rerun()
                    st.session_state.delta_t_pairs[i] = {'baseline': baseline, 'compare': compare}

                if st.button("Add ΔT Pair"):
                    st.session_state.delta_t_pairs.append({'baseline': NO_COMPARISON_LABEL, 'compare': NO_COMPARISON_LABEL})
                    st.rerun()

    spec_df = None
    if 'selected_ics' in locals() and selected_ics:
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
        elif st.session_state.uploaded_file_buffer is None:
            st.error("File buffer is missing. Please re-upload the file.")
        else:
            delta_pairs_for_analysis = [pair for pair in st.session_state.delta_t_pairs if pair['baseline'] != NO_COMPARISON_LABEL and pair['compare'] != NO_COMPARISON_LABEL]
            with st.spinner("Processing data..."):
                file_buffer = io.BytesIO(st.session_state.uploaded_file_buffer)
                st.session_state.cobra_analysis_results = run_cobra_analysis(file_buffer, cobra_data, selected_series, selected_ics, spec_df, delta_pairs_for_analysis)

    if st.session_state.get('cobra_analysis_results'):
        results = st.session_state.cobra_analysis_results
        if results.get("error"): st.error(f"**Analysis Error:** {results['error']}")
        else:
            st.header("Analysis Results")
            res_tab1, res_tab2, res_tab3 = st.tabs(["**Conclusions**", "**Table**", "**Chart**"])
            with res_tab1:
                render_structured_conclusions(results.get("conclusion_data", []))
            with res_tab2:
                st.subheader("Formatted Data Table")
                table_fig = generate_formatted_table_image(results.get("table_for_image"))
                st.pyplot(table_fig)

                img_buf = io.BytesIO(); table_fig.savefig(img_buf, format="png", dpi=300, bbox_inches='tight')
                excel_buf = create_formatted_excel(results.get("table_for_excel"))

                btn_col1, btn_col2 = st.columns(2)
                btn_col1.download_button("Download Table as PNG", data=img_buf, file_name="cobra_table.png", mime="image/png", use_container_width=True)
                btn_col2.download_button("Download as Formatted Excel", data=excel_buf, file_name="cobra_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with res_tab3:
                st.subheader("Temperature Comparison Chart")
                chart_data_numeric = results.get("chart_data")

                if chart_data_numeric is not None and not chart_data_numeric.empty:
                    fig_chart, ax = plt.subplots(figsize=(max(10, len(chart_data_numeric.index) * 0.8), 6))
                    df_chart_data_to_plot = chart_data_numeric[[s for s in selected_series if s in chart_data_numeric.columns]].copy()
                    
                    df_chart_data_to_plot.plot(kind='bar', ax=ax, width=0.8)
                    ax.set_ylabel("Temperature (°C)")
                    ax.set_title("Key IC Temperature Comparison")

                    for patch in ax.patches:
                        height = patch.get_height()
                        if pd.notna(height):
                            ax.text(patch.get_x() + patch.get_width() / 2., height,
                                    f'{height:.2f}',
                                    ha='center', va='bottom',
                                    xytext=(0, 3), textcoords='offset points')

                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig_chart)

                    chart_buf = io.BytesIO()
                    fig_chart.savefig(chart_buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("Download Chart as PNG", data=chart_buf, file_name="cobra_chart.png", mime="image/png", use_container_width=True)
                else:
                    st.warning("No data available to plot for the selected configurations.")


def render_structured_conclusions(conclusion_data):
    st.subheader("Executive Summary")
    if not conclusion_data:
        st.info("No conclusion data to display.")
        return

    failed_ics = [item['component'] for item in conclusion_data if item['result'] == 'FAIL']
    if failed_ics:
        st.markdown(f"**Result: <span style='color:red;'>FAIL</span>** - The following components exceeded thermal limits: **{', '.join(failed_ics)}**", unsafe_allow_html=True)
    else:
        st.markdown(f"**Result: <span style='color:lightgreen;'>PASS</span>** - All selected Key ICs are within their specified thermal limits.", unsafe_allow_html=True)

    st.divider()
    st.subheader("Detailed Component Analysis")

    for item in conclusion_data:
        result_text = item['result']
        status_emoji = "🔴" if result_text == "FAIL" else "🟢" if result_text == "PASS" else "⚪️"
        expander_title = f"**{item['component']}** — Result: {status_emoji} {result_text}"

        with st.expander(expander_title, expanded=True):
            spec_val = f"{item['spec']:.2f}°C" if pd.notna(item['spec']) else "N/A"
            st.markdown(f"**Specification Type:** `{item['spec_type']}`")
            st.markdown(f"**Calculated Spec Limit:** `{spec_val}`")
            if item.get('spec_inputs') != 'N/A':
                st.markdown(f"**Specification Inputs:** `{item['spec_inputs']}`")

            st.write("**Performance per Configuration:**")

            if item['series_results']:
                series_results_df = pd.DataFrame(item['series_results'])

                html_table = "<table><tr><th style='text-align:left; padding-right: 1em;'>Configuration</th><th style='text-align:left; padding-right: 1em;'>Temp (°C)</th><th style='text-align:left;'>Result</th></tr>"
                for _, row in series_results_df.iterrows():
                    res_text_inner = row['result']
                    temp_text = f"{row['temp']:.2f}" if pd.notna(row['temp']) else "N/A"
                    html_table += f"<tr><td>{row['series']}</td><td>{temp_text}</td><td style='font-weight:bold; color:{'#FF2B2B' if res_text_inner == 'FAIL' else 'inherit'};'>{res_text_inner}</td></tr>"
                html_table += "</table>"
                st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.caption("No temperature data to display (e.g., spec was not defined).")

# --- ======================================================================= ---
# ---                             MAIN APP ROUTER                             ---
# --- ======================================================================= ---
st.set_page_config(page_title="Sercomm Tool Suite", layout="wide")
st.sidebar.title("Sercomm Thermal Engineering")
app_selection = st.sidebar.radio("Select a Tool:", ("Cobra - Data Transformation", "Viper - Risk Analysis"))
st.sidebar.markdown("---")
st.sidebar.info("A unified platform for Sercomm's engineering analysis tools.")

if "Data Transformation" in app_selection:
    render_cobra_ui()
elif "Risk Analysis" in app_selection:
    render_viper_ui()

