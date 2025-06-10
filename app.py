# Sercomm Tool Suite v2.0 (featuring Viper & Cobra)
# Author: Gemini
# Description: A unified platform integrating the Viper Thermal Suite and the new Cobra Thermal Analyzer.
# Version Notes: 
# - Major architectural update to integrate Cobra.
# - Refactored Cobra's tkinter UI into Streamlit widgets.
# - Implemented Cobra's file upload and pre-analysis logic.

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

# Constants for Cobra
DATA_COL_COMPONENT_IDX = 1
DATA_COL_FIRST_SERIES_TEMP_IDX = 2
SPEC_TYPE_TC_CALC = "Tc"
SPEC_TYPE_TJ_ONLY = "Tj"
SPEC_TYPE_TA_ONLY = "Ta"

# --- ======================================================================= ---
# ---                     VIPER CALCULATION ENGINES                           ---
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
    except Exception as e: return {"error": f"An unexpected error occurred: {e}"}

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

        # --- Process Series ---
        raw_series_names = []
        series_excel_indices = {}
        header_row = df_header.iloc[header_row_idx]
        for i in range(DATA_COL_FIRST_SERIES_TEMP_IDX, len(header_row)):
            name = str(header_row.iloc[i]).strip()
            if name and name.lower() != 'nan':
                raw_series_names.append(name)
                series_excel_indices[name] = i
        
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

        # --- Process Components ---
        data_start_row = header_row_idx + 1
        df_components = pd.read_excel(xls, sheet_name=target_sheet, header=None, usecols=[DATA_COL_COMPONENT_IDX], skiprows=data_start_row, dtype=str)
        unique_original_components = df_components.iloc[:, 0].str.strip().replace('', np.nan).dropna().unique()
        
        cleaned_components_set = set()
        for orig_name in unique_original_components:
            cleaned_name = clean_component_display_name(orig_name)
            if cleaned_name:
                cleaned_components_set.add(cleaned_name)
        
        sorted_cleaned_components = sorted(list(cleaned_components_set))

        return {
            "error": None,
            "series_names": cleaned_names,
            "component_names": sorted_cleaned_components,
            "series_excel_indices": series_excel_indices,
            "cleaned_to_raw_map": cleaned_to_raw_map,
            "header_row_idx": header_row_idx
        }
    except Exception as e:
        return {"error": f"An error occurred during pre-study: {e}"}

# --- ======================================================================= ---
# ---                       APPLICATION UI FUNCTIONS                          ---
# --- ======================================================================= ---

def render_viper_ui():
    # ... (Viper UI code remains the same, but now it's a function)
    viper_logo_svg = """...""" # Keep SVG definition
    st.markdown(f"...", unsafe_allow_html=True) # Keep Viper title
    
    natural_convection_materials = {
        "Plastic (ABS/PC)": {"emissivity": 0.90, "k_uniform": 0.65},
        "Aluminum (Anodized)": {"emissivity": 0.85, "k_uniform": 0.90}
    }
    solar_absorptivity_materials = {
        "White (Paint)": {"absorptivity": 0.25},
        "Silver (Paint)": {"absorptivity": 0.40},
        "Dark Gray": {"absorptivity": 0.80},
        "Black (Plastic/Paint)": {"absorptivity": 0.95}
    }

    tab_nat, tab_force, tab_solar = st.tabs(["🍃 Natural Convection", "🌬️ Forced Convection", "☀️ Solar Radiation"])
    
    # --- Natural Convection Tab ---
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
            if nc_results.get("error"): st.error(f"**Error:** {nc_results['error']}")
            else: st.metric(label="✅ Max. Dissipatable Power", value=f"{nc_results['total_power']:.2f} W", help="...")

    # --- Forced Convection Tab ---
    with tab_force:
        st.header("Active Cooling Airflow Estimator")
        col_force_input, col_force_result = st.columns(2, gap="large")
        with col_force_input:
            st.subheader("Input Parameters")
            fc_param_col1, fc_param_col2 = st.columns(2, gap="medium")
            with fc_param_col1: fc_power_q = st.number_input("Power to Dissipate (Q, W)", min_value=0.1, value=50.0, step=1.0, format="%.1f", help="...")
            with fc_param_col2:
                fc_temp_in = st.number_input("Inlet Air Temp (Tin, °C)", 0, 60, 25, key="fc_tin")
                fc_temp_out = st.number_input("Max. Outlet Temp (Tout, °C)", fc_temp_in + 1, 100, 45, key="fc_tout")
            st.subheader("Governing Equation")
            st.latex(r"Q = \dot{m} \cdot C_p \cdot \Delta T")
        with col_force_result:
            st.subheader("Evaluation Result")
            fc_results = calculate_forced_convection(fc_power_q, fc_temp_in, fc_temp_out)
            if fc_results.get("error"): st.error(f"**Error:** {fc_results['error']}")
            else: st.metric(label="🌬️ Required Airflow", value=f"{fc_results['cfm']:.2f} CFM", help="...")

    # --- Solar Radiation Tab ---
    with tab_solar:
        st.header("Solar Heat Gain Estimator")
        col_solar_input, col_solar_result = st.columns(2, gap="large")
        with col_solar_input:
            st.subheader("Input Parameters")
            solar_material_name = st.selectbox("1. Enclosure Color/Finish", options=list(solar_absorptivity_materials.keys()) + ["Other..."], key="solar_mat")
            if solar_material_name == "Other...":
                alpha_val = st.number_input("Custom Absorptivity (α)", 0.0, 1.0, 0.5, 0.05, help="...")
            else:
                alpha_val = solar_absorptivity_materials[solar_material_name]["absorptivity"]
                st.number_input("Corresponding Absorptivity (α)", value=alpha_val, disabled=True)
            projected_area_mm2 = st.number_input("2. Projected Surface Area (mm²)", 0.0, value=30000.0, step=1000.0, format="%.1f", help="...")
            solar_irradiance_val = st.number_input("3. Solar Irradiance (G, W/m²)", 0, value=1000, step=50, help="...")
            st.subheader("Governing Equation")
            st.latex(r"Q_{solar} = \alpha \cdot A_{proj} \cdot G_{solar}")
        with col_solar_result:
            st.subheader("Evaluation Result")
            solar_results = calculate_solar_gain(projected_area_mm2, alpha_val, solar_irradiance_val)
            if solar_results.get("error"): st.error(f"**Error:** {solar_results['error']}")
            else: st.metric(label="☀️ Absorbed Solar Heat Gain", value=f"{solar_results['solar_gain']:.2f} W", help="...")

def render_cobra_ui():
    # --- Custom Cobra Logo and Title ---
    cobra_logo_svg = """...""" # Keep SVG definition
    st.markdown(f"...", unsafe_allow_html=True) # Keep Cobra title

    st.header("Excel Data Post-Processing")
    
    uploaded_file = st.file_uploader(
        "Upload an Excel file (.xlsx or .xls)",
        type=["xlsx", "xls"],
        key="cobra_file_uploader"
    )

    # Initialize session state for cobra data
    if 'cobra_data' not in st.session_state:
        st.session_state.cobra_data = {}

    if uploaded_file is not None:
        # If a new file is uploaded, reset the state and run pre-study
        if st.session_state.get('cobra_filename') != uploaded_file.name:
            st.session_state.cobra_filename = uploaded_file.name
            with st.spinner('Pre-analyzing Excel file...'):
                st.session_state.cobra_data = cobra_pre_study(uploaded_file)
    
    cobra_data = st.session_state.cobra_data

    if cobra_data.get("error"):
        st.error(cobra_data["error"])
        return

    if not cobra_data.get("series_names"):
        st.info("Upload an Excel file to see analysis options.")
        return
        
    st.subheader("Analysis Selections")
    col1, col2 = st.columns(2)

    with col1:
        selected_series = st.multiselect(
            "Select Configurations for Analysis:",
            options=cobra_data["series_names"],
            default=cobra_data["series_names"] # Default to all selected
        )
    
    with col2:
        selected_ics = st.multiselect(
            "Select Key ICs for Table/Spec Check:",
            options=cobra_data["component_names"]
        )

    st.divider()

    if selected_ics:
        st.subheader("Key IC Specification Input")
        st.info("This section for entering Tj, Rjc, etc., is under construction.")
        # Placeholder for data editor
        spec_data = []
        for ic in selected_ics:
            spec_data.append({"Component": ic, "Spec Type": "Tc", "Tj (°C)": 125.0, "Rjc (°C/W)": 1.5, "Pd (W)": 2.0, "Ta Limit (°C)": "N/A"})
        
        edited_specs = st.data_editor(spec_data, key="spec_editor")

    st.divider()
    
    if st.button("🚀 Analyze Selected Data", use_container_width=True):
        if not selected_series:
            st.warning("Please select at least one configuration to analyze.")
        else:
            with st.spinner("Processing data..."):
                st.write("Analysis logic will be implemented here.")
                st.write("Selected Series:", selected_series)
                st.write("Selected Key ICs:", selected_ics)
                if selected_ics:
                    st.write("With Specs:")
                    st.dataframe(edited_specs)
                st.success("Analysis would be complete!")


# --- ======================================================================= ---
# ---                           MAIN APP ROUTER                             ---
# --- ======================================================================= ---

st.set_page_config(page_title="Sercomm Tool Suite", layout="wide")

st.sidebar.title("Sercomm Engineering Suite")
app_selection = st.sidebar.radio(
    "Select a Tool:",
    ("Viper Thermal Suite", "Cobra Data Analyzer")
)
st.sidebar.markdown("---")
st.sidebar.info("A unified platform for Sercomm's engineering analysis tools.")

if app_selection == "Viper Thermal Suite":
    render_viper_ui()
elif app_selection == "Cobra Data Analyzer":
    render_cobra_ui()

