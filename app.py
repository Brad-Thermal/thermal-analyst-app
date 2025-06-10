# Viper Thermal Suite v9.2
# Author: Gemini
# Description: The successor to the Cobra series, a branded thermal analysis tool for the Sercomm Team.
# Version Notes: 
# - Integrated Solar Radiation analysis directly into the Natural Convection tab via a toggle switch.
# - The primary result now shows "Max. Internal Dissipatable Power" when solar load is included.
# - Added a "Thermal Budget Breakdown" to clarify results.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# --- Suppress specific Streamlit warnings ---
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

# --- Constants ---
STEFAN_BOLTZMANN_CONST = 5.67e-8  # W/(m^2*K^4)
EPSILON = 1e-9 # A small number to prevent division by zero
BUILT_IN_SAFETY_FACTOR = 0.9 # Fixed safety factor for natural convection
AIR_DENSITY_RHO = 1.225 # kg/m^3
AIR_SPECIFIC_HEAT_CP = 1006 # J/kg°C
M3S_TO_CFM_CONVERSION = 2118.88 # m^3/s to CFM
SOLAR_IRRADIANCE = 1000 # Standard solar irradiance in W/m^2

# --- Calculation Engines ---

def calculate_natural_convection(L, W, H, Ts_peak, Ta, material_props):
    """Calculates the TOTAL dissipatable power for a given set of conditions."""
    if Ts_peak <= Ta: return { "error": "Max. Allowable Surface Temp (Ts) must be higher than Ambient Temp (Ta)." }
    if L <= 0 or W <= 0 or H <= 0: return { "error": "Product dimensions (L, W, H) must be greater than zero." }
    try:
        epsilon = material_props["emissivity"]
        k_uniform = material_props["k_uniform"]
        Ts_eff = Ta + (Ts_peak - Ta) * k_uniform
        delta_T_eff = Ts_eff - Ta
        L_m, W_m, H_m = L/1000, W/1000, H/1000
        A_total = 2 * (L_m*W_m + L_m*H_m + W_m*H_m)
        T_film = (Ts_eff + Ta) / 2
        k_air = 0.0275; nu_air = 1.85e-5; pr_air = 0.72; g = 9.81
        beta = 1 / (T_film + 273.15 + EPSILON)
        Lc_vert = H_m
        Lc_horiz = (L_m * W_m) / (2 * (L_m + W_m) + EPSILON)
        Ra_vert = (g * beta * delta_T_eff * Lc_vert**3) / (nu_air**2) * pr_air
        Ra_horiz = (g * beta * delta_T_eff * Lc_horiz**3) / (nu_air**2) * pr_air
        Nu_vert = (0.825 + (0.387 * abs(Ra_vert)**(1/6)) / (1 + (0.492/pr_air)**(9/16))**(8/27))**2
        if 1e4 <= Ra_horiz <= 1e7: Nu_top = 0.54 * Ra_horiz**(1/4)
        elif Ra_horiz > 1e7: Nu_top = 0.15 * Ra_horiz**(1/3)
        else: Nu_top = 1.0
        Nu_bottom = 0.27 * abs(Ra_horiz)**(1/4)
        h_vert = (Nu_vert * k_air) / (Lc_vert + EPSILON)
        h_top = (Nu_top * k_air) / (Lc_horiz + EPSILON)
        h_bottom = (Nu_bottom * k_air) / (Lc_horiz + EPSILON)
        Q_conv_total = ((h_top * L_m*W_m) + (h_bottom * L_m*W_m) + h_vert * 2 * (L_m*H_m + W_m*H_m)) * delta_T_eff
        h_avg = Q_conv_total / ((A_total + EPSILON) * delta_T_eff)
        Ts_eff_K, Ta_K = Ts_eff + 273.15, Ta + 273.15
        Q_rad = epsilon * STEFAN_BOLTZMANN_CONST * A_total * (Ts_eff_K**4 - Ta_K**4)
        Q_ideal_total = Q_conv_total + Q_rad
        Q_final = Q_ideal_total * BUILT_IN_SAFETY_FACTOR
        results = {
            "total_power": Q_final, "convection_power": Q_conv_total, "radiation_power": Q_rad,
            "h_avg": h_avg, "surface_area": A_total, "Ts_eff": Ts_eff, "k_uniform": k_uniform, "error": None
        }
        for k, v in results.items():
            if isinstance(v, (float, int)) and (np.isnan(v) or np.isinf(v)):
                return {"error": f"Invalid number (NaN/inf) encountered during calculation."}
        return results
    except Exception as e: return { "error": f"An unexpected error occurred: {e}" }

def calculate_forced_convection(power_q, T_in, T_out):
    """Calculates the required airflow in CFM based on Q = m*Cp*dT."""
    if T_out <= T_in: return {"error": "Outlet Temperature must be higher than Inlet Temperature."}
    if power_q <= 0: return {"error": "Power to be dissipated must be greater than zero."}
    delta_T = T_out - T_in
    mass_flow_rate = power_q / (AIR_SPECIFIC_HEAT_CP * delta_T) # kg/s
    volume_flow_rate_m3s = mass_flow_rate / AIR_DENSITY_RHO # m^3/s
    volume_flow_rate_cfm = volume_flow_rate_m3s * M3S_TO_CFM_CONVERSION # CFM
    return {"cfm": volume_flow_rate_cfm, "error": None}

def calculate_solar_gain(L, W, H, solar_props):
    """Calculates the absorbed solar heat based on Q = α * A_proj * G."""
    if L <= 0 or W <= 0 or H <= 0: return {"error": "Product dimensions must be greater than zero."}
    try:
        alpha = solar_props["absorptivity"]
        # Assuming sun is directly overhead, projected area is LxW.
        # A more complex model could consider side-facing sun.
        projected_area = (L / 1000) * (W / 1000) # m^2
        solar_gain = alpha * projected_area * SOLAR_IRRADIANCE
        return {"solar_gain": solar_gain, "projected_area": projected_area, "alpha": alpha, "error": None}
    except Exception as e: return { "error": f"An unexpected error occurred during calculation: {e}" }

# --- Main Application UI ---
st.set_page_config(page_title="Viper Thermal Suite", layout="wide")

# --- Custom Viper Logo and Title ---
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
st.markdown(
    f"""
    <div style="display: flex; align-items: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px;">
        <div style="margin-right: 15px;">{viper_logo_svg}</div>
        <div>
            <h1 style="margin-bottom: 0; color: #FFFFFF;">Viper Thermal Suite</h1>
            <p style="margin-top: 0; color: #AAAAAA;">A Thermal Assessment Tool for the Sercomm Thermal Team</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Material Properties Definition ---
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

# --- Main UI Tabs ---
tab_nat, tab_force = st.tabs(["🍃 Natural Convection", "🌬️ Forced Convection"])

# --- Natural Convection Tab ---
with tab_nat:
    st.header("Passive Cooling Power Estimator")
    col_nat_input, col_nat_result = st.columns([1, 1], gap="large")

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
        
        st.divider()
        include_solar = st.toggle("Include Solar Radiation Effects?")
        solar_color = ""
        if include_solar:
            solar_color = st.selectbox("Enclosure Color/Finish", options=list(solar_absorptivity_materials.keys()), key="solar_mat")

    with col_nat_result:
        st.subheader("Evaluation Result")
        selected_material_props_nc = natural_convection_materials[nc_material_name]
        nc_results = calculate_natural_convection(nc_dim_L, nc_dim_W, nc_dim_H, nc_temp_surface_peak, nc_temp_ambient, selected_material_props_nc)
        
        solar_gain = 0
        if include_solar and solar_color:
            selected_solar_props = solar_absorptivity_materials[solar_color]
            solar_results = calculate_solar_gain(nc_dim_L, nc_dim_W, nc_dim_H, selected_solar_props)
            if not solar_results.get("error"):
                solar_gain = solar_results["solar_gain"]

        if nc_results.get("error"):
            st.error(f"**Error:** {nc_results['error']}")
        else:
            total_dissipatable_power = nc_results['total_power']
            internal_power_budget = total_dissipatable_power - solar_gain

            if include_solar:
                st.metric(
                    label="✅ Max. Internal Dissipatable Power",
                    value=f"{internal_power_budget:.2f} W",
                    help="This is the power your internal electronics can generate after accounting for the sun's heat."
                )
                st.markdown("---")
                st.markdown("##### Thermal Budget Breakdown")
                st.markdown(f"☀️ **Solar Heat Gain:** `{solar_gain:.2f} W`")
                st.markdown(f"🔌 **Internal Power Budget:** `{internal_power_budget:.2f} W`")
                st.markdown(f"---")
                st.markdown(f"**Total Dissipated by Enclosure:** `{total_dissipatable_power:.2f} W`")
            else:
                 st.metric(label="✅ Max. Dissipatable Power", value=f"{total_dissipatable_power:.2f} W")
                 st.info("This result includes built-in material uniformity and a fixed engineering safety factor (0.9).")

# --- Forced Convection Tab ---
with tab_force:
    st.header("Active Cooling Airflow Estimator")
    col_force_input, col_force_result = st.columns([1, 1], gap="large")

    with col_force_input:
        st.subheader("Input Parameters")
        fc_param_col1, fc_param_col2 = st.columns(2, gap="medium")
        with fc_param_col1: fc_power_q = st.number_input("Power to Dissipate (Q, W)", min_value=0.1, value=50.0, step=1.0, format="%.1f", help="The total heat (in Watts) that the fan must remove.")
        with fc_param_col2:
            fc_temp_in = st.number_input("Inlet Air Temp (Tin, °C)", 0, 60, 25, key="fc_tin")
            fc_temp_out = st.number_input("Max. Outlet Temp (Tout, °C)", fc_temp_in + 1, 100, 45, key="fc_tout")
        st.subheader("Governing Equation")
        st.latex(r"Q = \dot{m} \cdot C_p \cdot \Delta T")

    with col_force_result:
        st.subheader("Evaluation Result")
        fc_results = calculate_forced_convection(fc_power_q, fc_temp_in, fc_temp_out)
        if fc_results.get("error"):
            st.error(f"**Error:** {fc_results['error']}")
        else:
            st.metric(
                label="🌬️ Required Airflow",
                value=f"{fc_results['cfm']:.2f} CFM",
                help="CFM: Cubic Feet per Minute. This is the minimum airflow required to dissipate the specified power under the given temperature constraints."
            )
            st.info("This calculation assumes standard air density (1.225 kg/m³) and specific heat (1006 J/kg°C).")
