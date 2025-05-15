import streamlit as st
import pandas as pd

# --- Load Data ---
df = pd.read_csv("df_sensors_with_status.csv", parse_dates=["Timestamp"])

# --- Page Title ---
st.title("Predictive Maintenance Dashboard")
st.markdown("View equipment status over time to support preventive maintenance planning.")

# --- Sidebar: Time Selection ---
st.sidebar.subheader("1. Select Timestamp")
selected_time = st.sidebar.selectbox(
    "Timestamp:",
    sorted(df["Timestamp"].unique())
)

# Filter by selected timestamp
filtered_df = df[df["Timestamp"] == selected_time]

# --- Sidebar: Risk Filter ---
st.sidebar.subheader("2. Filter by Maintenance Status")
risk_levels = ["All", "Normal", "Needs Attention"]
selected_level = st.sidebar.selectbox("Show:", risk_levels)

if selected_level != "All":
    filtered_df = filtered_df[filtered_df["Maintenance_Status"] == selected_level]

# --- Now that filters are applied, show metrics ---
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Normal Machines", value=(filtered_df['Maintenance_Status'] == 'Normal').sum())

with col2:
    st.metric(label="Machines Needing Attention", value=(filtered_df['Maintenance_Status'] == 'Needs Attention').sum())

# --- Display Machine Status Table ---
st.subheader(f"Machine Status at {selected_time.strftime('%Y-%m-%d %H:%M')}")
st.dataframe(
    filtered_df[[  # Removed 'reconstruction_error'
        "Machine_ID", "Temperature", "Temperature_delta", "Vibration", "Vibration_delta",
        "Pressure", "Pressure_delta", "Maintenance_Status"
    ]]
)

# --- Error Trend for a Selected Machine ---
st.sidebar.subheader("3. View Machine Trend")
machine_options = df["Machine_ID"].unique()
selected_machine = st.sidebar.selectbox("Machine:", machine_options)

machine_df = df[df["Machine_ID"] == selected_machine].sort_values("Timestamp")

st.subheader(f"Sensor Readings Over Time: {selected_machine}")
st.line_chart(machine_df.set_index("Timestamp")[["Temperature", "Vibration", "Pressure"]])

# --- Footer ---
st.markdown("---")
st.caption("Model: LSTM Autoencoder | Labels: Predictive (5-hour lookahead)")
