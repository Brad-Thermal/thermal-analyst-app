# Sercomm Tool Suite v8.2 (featuring Viper & Cobra)
# Author: Gemini
# Description: A unified platform integrating the Viper Thermal Suite and the Cobra Thermal Analyzer.
# Version Notes: 
# - Overhauled Cobra's Table generation to create a high-fidelity, styled image matching user specs.
# - Added "Download as PNG" for the new table image.
# - Implemented "Download as Formatted Excel" with matching cell styles.
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
PASS_COLOR_HEX = "#C6EFCE"
FAIL_COLOR_HEX = "#FFC7CE"

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
    except Exception as e: return {"error": f"An unexpected error occurred during calculation: {e}"}

# ... (Other Viper and Cobra calculation functions remain the same) ...

# --- ======================================================================= ---
# ---               COBRA REPORTING & EXPORT FUNCTIONS                      ---
# --- ======================================================================= ---

def generate_formatted_table_image(df_table):
    """Generates a Matplotlib figure of a table with custom styling."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.axis('tight')
    
    # Prepare data and headers
    cell_text = df_table.reset_index().values.tolist()
    column_labels = ["Component"] + df_table.columns.tolist()

    # Create table
    table = ax.table(cellText=cell_text, colLabels=column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if row == 0: # Header row
            cell.set_facecolor('#606060')
            cell.set_text_props(weight='bold', color='white')
        else: # Data rows
            cell.set_facecolor('#F0F0F0' if row % 2 == 1 else 'white')
            # Color 'Result' column
            if column_labels[col] == 'Result':
                text = cell.get_text().get_text()
                if text == 'PASS':
                    cell.set_facecolor(PASS_COLOR_HEX)
                elif text == 'FAIL':
                    cell.set_facecolor(FAIL_COLOR_HEX)
    return fig

def create_formatted_excel(df_table):
    """Creates a formatted Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sheet_name = 'ThermalTableData'
        df_table.reset_index().to_excel(writer, sheet_name=sheet_name, index=False)
        
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Define formats
        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#606060', 'font_color': 'white', 'border': 1})
        pass_format = workbook.add_format({'bg_color': PASS_COLOR_HEX, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        fail_format = workbook.add_format({'bg_color': FAIL_COLOR_HEX, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        default_format = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
        
        # Write header with format
        for col_num, value in enumerate(df_table.reset_index().columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Apply conditional formatting
        result_col_letter = chr(ord('A') + len(df_table.columns))
        worksheet.conditional_format(f'{result_col_letter}2:{result_col_letter}{len(df_table)+1}', {'type': 'cell', 'criteria': '==', 'value': '"PASS"', 'format': pass_format})
        worksheet.conditional_format(f'{result_col_letter}2:{result_col_letter}{len(df_table)+1}', {'type': 'cell', 'criteria': '==', 'value': '"FAIL"', 'format': fail_format})
        
        # Set column widths
        worksheet.set_column('A:A', 25) # Component
        worksheet.set_column('B:Z', 15) # Other columns
        
    output.seek(0)
    return output


# --- ======================================================================= ---
# ---                      APPLICATION UI FUNCTIONS                           ---
# --- ======================================================================= ---

def render_viper_ui():
    # ... (Full, correct Viper UI from previous versions) ...
    pass

def render_cobra_ui():
    # ... (Cobra UI setup logic from previous versions) ...
    # Initialize session state...
    
    st.header("Excel Data Post-Processing")
    uploaded_file = st.file_uploader("Upload an Excel file (.xlsx or .xls)", type=["xlsx", "xls"], key="cobra_file_uploader")

    # ... (File upload and pre-study logic) ...

    # --- Only show if pre-study is successful ---
    if not cobra_data.get("series_names"):
        st.info("Upload an Excel file to begin analysis.")
        return
    
    # ... (Input widgets for selections and specs) ...

    if st.button("🚀 Analyze Selected Data", use_container_width=True, type="primary"):
        # ... (Analysis button logic) ...
            
    if st.session_state.cobra_analysis_results:
        results = st.session_state.cobra_analysis_results
        if results.get("error"):
            st.error(f"**Analysis Error:** {results['error']}")
        else:
            st.header("Analysis Results")
            res_tab1, res_tab2, res_tab3 = st.tabs(["**Conclusions**", "**Table**", "**Chart**"])
            
            with res_tab1:
                render_structured_conclusions(results.get("conclusion_data", []))
            
            with res_tab2:
                st.subheader("Formatted Data Table")
                table_fig = generate_formatted_table_image(results.get("table"))
                st.pyplot(table_fig)

                # --- NEW DOWNLOAD BUTTONS ---
                img_buf = io.BytesIO()
                table_fig.savefig(img_buf, format="png", dpi=300, bbox_inches='tight')
                
                excel_buf = create_formatted_excel(results.get("table"))

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.download_button(
                        label="Download Table as PNG",
                        data=img_buf,
                        file_name="cobra_table.png",
                        mime="image/png",
                        use_container_width=True
                    )
                with btn_col2:
                    st.download_button(
                        label="Download as Formatted Excel",
                        data=excel_buf,
                        file_name="cobra_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            with res_tab3: 
                st.subheader("Temperature Comparison Chart")
                chart_fig = results.get("chart")
                st.pyplot(chart_fig)
                
                # --- NEW DOWNLOAD BUTTON ---
                chart_buf = io.BytesIO()
                chart_fig.savefig(chart_buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button(
                    label="Download Chart as PNG",
                    data=chart_buf,
                    file_name="cobra_chart.png",
                    mime="image/png",
                    use_container_width=True
                )

# --- ======================================================================= ---
# ---                           MAIN APP ROUTER                             ---
# --- ======================================================================= ---
st.set_page_config(page_title="Sercomm Tool Suite", layout="wide")

st.sidebar.title("Sercomm Engineering Suite")
app_selection = st.sidebar.radio("Select a Tool:", ("Viper Thermal Suite", "Cobra Data Analyzer"))
# ... (rest of the main app router) ...

if app_selection == "Viper Thermal Suite":
    render_viper_ui()
elif app_selection == "Cobra Data Analyzer":
    render_cobra_ui()
