# ThermoPower Analyst (TPA) v4.2
# Author: Gemini
# Description: A standardized evaluation tool with a minimalist UI and enhanced chart interpretation.
# Version Notes: 
# - Switched to matplotlib for charting to add clear X/Y axis labels.
# - Added a detailed "How to Interpret" section with actionable advice below the chart.

import streamlit as st
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

# --- Suppress specific Streamlit warnings ---
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

# --- Constants ---
STEFAN_BOLTZMANN_CONST = 5.67e-8  # W/(m^2*K^4)
EPSILON = 1e-9 # A small number to prevent division by zero
BUILT_IN_SAFETY_FACTOR = 0.9 # A fixed safety factor is now part of the model

# --- Physics & Calculation Engine ---

def get_air_properties(T_film_C):
    """Estimate air properties at a given film temperature in Celsius."""
    k_air = 0.0275; nu_air = 1.85e-5; pr_air = 0.72
    beta = 1 / (T_film_C + 273.15 + EPSILON)
    return k_air, nu_air, pr_air, beta

def calculate_heat_dissipation(L, W, H, Ts_peak, Ta, material_props):
    """Core physics engine. Coefficients are derived from material_props."""
    # 1. Input validation
    if Ts_peak <= Ta: return { "error": "Surface Temperature (Ts) must be higher than Ambient Temperature (Ta)." }
    if L <= 0 or W <= 0 or H <= 0: return { "error": "Product dimensions (L, W, H) must be greater than zero." }

    try:
        # 2. Get properties from material selection
        epsilon = material_props["emissivity"]
        k_uniform = material_props["k_uniform"] # Get k_uniform from material dict

        # 3. Effective Surface Temperature based on built-in k_uniform
        Ts_eff = Ta + (Ts_peak - Ta) * k_uniform
        delta_T_eff = Ts_eff - Ta

        # 4. Geometry
        L_m, W_m, H_m = L / 1000, W / 1000, H / 1000
        A_total = 2 * (L_m*W_m + L_m*H_m + W_m*H_m)

        # 5. Convection Calculation (Detailed Physics Model ONLY)
        T_film = (Ts_eff + Ta) / 2
        k_air, nu_air, pr_air, beta = get_air_properties(T_film)
        g = 9.81
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
       
        # 6. Radiation Calculation
        Ts_eff_K, Ta_K = Ts_eff + 273.15, Ta + 273.15
        Q_rad = epsilon * STEFAN_BOLTZMANN_CONST * A_total * (Ts_eff_K**4 - Ta_K**4)
        
        # 7. Total Power with built-in safety factor
        Q_ideal_total = Q_conv_total + Q_rad
        Q_final = Q_ideal_total * BUILT_IN_SAFETY_FACTOR

        results_dict = {
            "total_power": Q_final, "convection_power": Q_conv_total, "radiation_power": Q_rad,
            "h_avg": h_avg, "surface_area": A_total, "Ts_eff": Ts_eff, "k_uniform": k_uniform, "error": None
        }
        for key, value in results_dict.items():
            if isinstance(value, (float, int)) and (np.isnan(value) or np.isinf(value)):
                return {"error": f"Invalid number (NaN/inf) encountered during calculation."}
        return results_dict
    except Exception as e:
        return { "error": f"An unexpected error occurred during calculation: {e}" }

# --- Simplified User Interface (v4.2) ---
st.set_page_config(page_title="ThermoPower Analyst", layout="wide")
st.title("🌡️ ThermoPower Analyst v4.2")
st.markdown("A Standardized Thermal Risk Assessment Tool")

# --- Material Properties Definition ---
materials_dict = {
    "Plastic (ABS/PC)":                     {"emissivity": 0.90, "k_uniform": 0.65},
    "Aluminum (ADC-12, Anodized)":          {"emissivity": 0.85, "k_uniform": 0.90},
    "Aluminum (ADC-12, Unfinished)":        {"emissivity": 0.10, "k_uniform": 0.95}
}

# --- UI Tabs ---
tab1, tab2 = st.tabs(["📊 Main Evaluation", "📈 Detailed Analysis"])

with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("1. Operating Conditions (°C)")
        temp_ambient = st.slider("Ambient Temperature (Ta)", 0, 60, 25)
        temp_surface_peak = st.slider("Max. Allowable Surface Temp (Ts)", temp_ambient + 1, 100, 50)
        
        st.subheader("2. Product Dimensions (mm)")
        dim_col1, dim_col2, dim_col3 = st.columns(3)
        with dim_col1:
            dim_L = st.number_input("Length (L)", 1.0, 1000.0, 200.0, 10.0, "%.1f")
        with dim_col2:
            dim_W = st.number_input("Width (W)", 1.0, 1000.0, 150.0, 10.0, "%.1f")
        with dim_col3:
            dim_H = st.number_input("Height (H)", 1.0, 500.0, 50.0, 5.0, "%.1f")

        st.subheader("3. Enclosure Material")
        material_name = st.selectbox("Select Material", options=list(materials_dict.keys()))
        
    # --- Perform Calculation ---
    selected_material_props = materials_dict[material_name]
    results = calculate_heat_dissipation(dim_L, dim_W, dim_H, temp_surface_peak, temp_ambient, selected_material_props)

    with col_result:
        st.header("Final Evaluation Result")
        if results.get("error"):
            st.error(f"**Error:** {results['error']}")
        else:
            st.metric(
                label="✅ Dissipatable Power",
                value=f"{results['total_power']:.2f} W",
                help="This result includes built-in considerations for temperature uniformity and a safety factor."
            )
            st.info("This result incorporates the material's temperature uniformity and a fixed engineering safety factor (0.9).")

with tab2:
    st.header("Detailed Calculation Parameters")
    if results.get("error"):
        st.warning("Please enter valid parameters in the Main Evaluation tab to see the analysis.")
    else:
        # --- Parameter Display ---
        st.markdown(f"""
        - **Avg. Heat Transfer Coefficient (h_avg):** `{results['h_avg']:.2f} W/m²K`
        - **Effective Surface Temperature (Ts_eff):** `{results['Ts_eff']:.1f} °C`
        - **Total Surface Area:** `{results['surface_area'] * 10000:.1f} cm²`
        - **Peak Surface Temperature (Ts_peak):** `{temp_surface_peak}°C`
        - **Ambient Temperature (Ta):** `{temp_ambient}°C`
        - **Material Emissivity:** `{selected_material_props['emissivity']}`
        - **Built-in Temperature Uniformity Factor (k_uniform):** `{results['k_uniform']}`
        - **Built-in Engineering Safety Factor:** `{BUILT_IN_SAFETY_FACTOR}`
        """)
        
        st.divider()

        # --- Chart and Interpretation Section ---
        st.subheader("Heat Dissipation Breakdown")
        ideal_total_power = results['total_power'] / BUILT_IN_SAFETY_FACTOR
        
        # --- Chart Generation with Matplotlib ---
        fig, ax = plt.subplots(figsize=(8, 4))
        modes = ["Convection", "Radiation"]
        powers = [results["convection_power"], results["radiation_power"]]
        bars = ax.bar(modes, powers, color=['#1f77b4', '#ff7f0e'])
        
        # Style for dark theme
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('#0E1117')
        ax.spines['right'].set_color('#0E1117')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Add labels
        ax.set_ylabel("Ideal Power (W)", color='white')
        ax.set_xlabel("Heat Transfer Mode", color='white')

        # Add text labels on bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f} W', va='bottom', ha='center', color='white')

        st.pyplot(fig)

        # --- Interpretation Section ---
        st.subheader("How to Interpret This Chart & Next Steps")
        st.markdown("""
        This chart shows how the total heat is dissipated through two primary methods:
        - **Convection:** Heat transferred to the surrounding air. This is driven by airflow over the product's surface.
        - **Radiation:** Heat transferred via electromagnetic waves. This is primarily driven by the material's surface properties (emissivity) and temperature.

        Use this breakdown to guide your design improvements:
        """)

        conv_p = results['convection_power'] / (ideal_total_power + EPSILON)
        rad_p = results['radiation_power'] / (ideal_total_power + EPSILON)

        if rad_p > conv_p:
            st.warning("""
            **Insight: Radiation is the dominant heat transfer mode.**
            - **Next Steps:** To improve thermal performance, focus on the material surface. Using a material with high emissivity (like anodized aluminum or standard plastics) is highly effective. Avoid unfinished, polished metals if passive cooling is critical.
            """)
        else:
            st.info("""
            **Insight: Convection is the dominant heat transfer mode.**
            - **Next Steps:** To improve thermal performance, focus on increasing airflow and surface area. Consider adding ventilation holes to the enclosure, ensuring the device isn't placed in a confined space, or increasing the overall product dimensions. Adding fins is a classic way to dramatically increase convective surface area.
            """)
