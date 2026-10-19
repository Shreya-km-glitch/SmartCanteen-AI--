import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from data_generator import generate_canteen_data
from forecasting import train_and_predict


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="SmartCanteen AI",
    page_icon="🍱",
    layout="wide"
)


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #121212;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #123B32;
        font-weight: 700;
    }

    h1, h2, h3 {
        color: #123B32;
    }

    div[data-testid="stSidebar"] {
        background-color: #123B32;
    }

    div[data-testid="stSidebar"] * {
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------

DATA_PATH = Path("data/canteen_data.csv")


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    return generate_canteen_data()


@st.cache_data
def get_forecast(data):
    return train_and_predict(data)


df = load_data()
df["date"] = pd.to_datetime(df["date"])

forecasts, metrics = get_forecast(df)


# ---------------- SIDEBAR ----------------

st.sidebar.title("🍱 SmartCanteen AI")
st.sidebar.caption("INTELLIGENT FOOD MANAGEMENT")

st.sidebar.markdown("---")

st.sidebar.write("📊 Dashboard")
st.sidebar.write("🔮 Demand Forecast")
st.sidebar.write("♻️ Waste Analytics")
st.sidebar.write("📥 Data Export")

st.sidebar.markdown("---")

dish_options = ["All dishes"] + sorted(df["dish"].unique().tolist())

selected_dish = st.sidebar.selectbox(
    "Select a dish",
    dish_options
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Prototype mode: Current dataset is simulated."
)


# ---------------- HEADER ----------------

st.title("🍱 SmartCanteen AI")

st.markdown(
    "### Predict demand. Reduce waste. Make smarter decisions."
)

st.write(
    "An AI-powered prototype designed to help canteens "
    "estimate food demand and understand leftover costs."
)

st.warning(
    "DEMO NOTICE: This prototype uses simulated data. "
    "Forecasts and estimated costs are demonstrations, "
    "not verified real-world results."
)


# ---------------- FILTER DATA ----------------

if selected_dish != "All dishes":

    filtered_df = df[df["dish"] == selected_dish].copy()

    filtered_forecast = forecasts[
        forecasts["dish"] == selected_dish
    ].copy()

    displayed_metrics = metrics[
        metrics["dish"] == selected_dish
    ].copy()

else:

    filtered_df = df.copy()
    filtered_forecast = forecasts.copy()
    displayed_metrics = metrics.copy()


# ---------------- KPI CALCULATIONS ----------------

total_prepared = int(filtered_df["prepared"].sum())

total_sold = int(filtered_df["sold"].sum())

total_waste = float(filtered_df["leftover_kg"].sum())

waste_cost = float(
    (
        filtered_df["leftover_portions"]
        * filtered_df["cost_per_portion"]
    ).sum()
)

waste_percentage = (
    filtered_df["leftover_portions"].sum()
    / total_prepared * 100
    if total_prepared > 0 else 0
)


# ---------------- KPI CARDS ----------------

st.markdown("## 📊 Performance Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Portions Prepared",
    f"{total_prepared:,}"
)

c2.metric(
    "Portions Sold",
    f"{total_sold:,}"
)

c3.metric(
    "Food Waste",
    f"{total_waste:,.1f} kg"
)

c4.metric(
    "Waste Cost",
    f"₹{waste_cost:,.0f}"
)

st.caption(
    f"Simulated leftover rate: {waste_percentage:.2f}% "
    "of prepared portions."
)

st.divider()


# ---------------- DEMAND FORECAST ----------------

st.markdown("## 🔮 Demand Forecast")

st.write(
    "Estimated demand for the next seven days, based on "
    "the historical records in this prototype."
)

forecast_chart = px.line(
    filtered_forecast,
    x="date",
    y="predicted_demand",
    color="dish",
    markers=True,
    template="plotly_white",
    labels={
        "date": "Date",
        "predicted_demand": "Predicted portions",
        "dish": "Dish"
    }
)

forecast_chart.update_layout(
    height=420,
    legend_title_text="Menu item",
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(
    forecast_chart,
    use_container_width=True
)


# ---------------- WASTE ANALYTICS ----------------

st.markdown("## ♻️ Waste Analytics")

left, right = st.columns(2)

with left:

    st.markdown("### Waste by Dish")

    waste_by_dish = (
        filtered_df.groupby("dish", as_index=False)
        ["leftover_kg"].sum()
        .sort_values("leftover_kg", ascending=False)
    )

    waste_chart = px.bar(
        waste_by_dish,
        x="dish",
        y="leftover_kg",
        color="dish",
        template="plotly_white",
        labels={
            "dish": "Menu item",
            "leftover_kg": "Leftover food (kg)"
        }
    )

    waste_chart.update_layout(
        showlegend=False,
        height=350
    )

    st.plotly_chart(
        waste_chart,
        use_container_width=True
    )


with right:

    st.markdown("### Estimated Waste Cost")

    cost_by_dish = (
        filtered_df.assign(
            estimated_cost=(
                filtered_df["leftover_portions"]
                * filtered_df["cost_per_portion"]
            )
        )
        .groupby("dish", as_index=False)["estimated_cost"]
        .sum()
        .sort_values("estimated_cost", ascending=False)
    )

    cost_chart = px.bar(
        cost_by_dish,
        x="dish",
        y="estimated_cost",
        color="dish",
        template="plotly_white",
        labels={
            "dish": "Menu item",
            "estimated_cost": "Estimated cost (₹)"
        }
    )

    cost_chart.update_layout(
        showlegend=False,
        height=350
    )

    st.plotly_chart(
        cost_chart,
        use_container_width=True
    )


st.divider()
# ---------------- WHAT-IF SIMULATOR ----------------

st.markdown("## 🧮 What-If Food Preparation Simulator")

st.write(
    "Explore how reducing preparation quantities might "
    "affect leftovers and their estimated ingredient cost."
)

reduction_percent = st.slider(
    "Reduce preparation quantity by (%)",
    min_value=0,
    max_value=30,
    value=10,
    step=5
)

# Use the historical data to create a simple scenario.
# This is an estimate, not a guaranteed outcome.

simulated_prepared = (
    filtered_df["prepared"] * (1 - reduction_percent / 100)
).round()

simulated_demand = filtered_df["sold"]

simulated_leftovers = (
    simulated_prepared - simulated_demand
).clip(lower=0)

simulated_waste_kg = (
    simulated_leftovers * 0.25
).sum()

simulated_waste_cost = (
    simulated_leftovers
    * filtered_df["cost_per_portion"]
).sum()

original_waste_cost = (
    filtered_df["leftover_portions"]
    * filtered_df["cost_per_portion"]
).sum()

original_waste_kg = filtered_df["leftover_kg"].sum()

cost_saved = original_waste_cost - simulated_waste_cost

waste_avoided = original_waste_kg - simulated_waste_kg

# Show scenario metrics.

sim1, sim2, sim3 = st.columns(3)

sim1.metric(
    "Scenario Leftovers",
    f"{simulated_waste_kg:,.1f} kg",
    delta=f"{waste_avoided:,.1f} kg avoided",
    delta_color="normal"
)

sim2.metric(
    "Scenario Waste Cost",
    f"₹{simulated_waste_cost:,.0f}"
)

sim3.metric(
    "Estimated Cost Difference",
    f"₹{cost_saved:,.0f}"
)

st.caption(
    "Scenario estimate only: this calculation assumes historical "
    "sales remain unchanged after reducing preparation. In reality, "
    "lower preparation could cause stockouts and lost sales. "
    "Current data is simulated, so these are not verified savings."
)

# Compare current and scenario waste.

comparison_df = pd.DataFrame({
    "Scenario": [
        "Current recorded leftovers",
        "What-if scenario"
    ],
    "Leftovers (kg)": [
        original_waste_kg,
        simulated_waste_kg
    ]
})

comparison_chart = px.bar(
    comparison_df,
    x="Scenario",
    y="Leftovers (kg)",
    color="Scenario",
    template="plotly_white",
    title="Current vs Simulated Leftovers"
)

st.plotly_chart(
    comparison_chart,
    use_container_width=True
)


# ---------------- UPCOMING DEMAND TABLE ----------------

st.markdown("## 📅 Upcoming Demand")

forecast_table = filtered_forecast[
    ["date", "dish", "predicted_demand"]
].copy()

forecast_table["date"] = (
    forecast_table["date"].dt.strftime("%d %b %Y")
)

forecast_table = forecast_table.rename(
    columns={
        "date": "Forecast Date",
        "dish": "Menu Item",
        "predicted_demand": "Predicted Portions"
    }
)

st.dataframe(
    forecast_table.sort_values(
        ["Forecast Date", "Menu Item"]
    ),
    use_container_width=True,
    hide_index=True
)


# ---------------- MODEL EVALUATION ----------------

st.markdown("## 🧪 Forecast Model Evaluation")

st.caption(
    "MAE = Mean Absolute Error. It measures the average "
    "absolute difference between predictions and actual "
    "held-out historical values. Lower is better, but "
    "the result alone does not prove real-world accuracy."
)

st.dataframe(
    displayed_metrics,
    use_container_width=True,
    hide_index=True
)


# ---------------- WASTE DETECTIVE ----------------

st.markdown("## 🕵️ Waste Detective")

latest_date = filtered_df["date"].max()

recent = filtered_df[
    filtered_df["date"] >= latest_date - pd.Timedelta(days=6)
]

previous = filtered_df[
    (filtered_df["date"] < latest_date - pd.Timedelta(days=6))
    & (
        filtered_df["date"]
        >= latest_date - pd.Timedelta(days=13)
    )
]

recent_waste = recent["leftover_kg"].sum()

previous_waste = previous["leftover_kg"].sum()

if previous_waste > 0:

    change = (
        (recent_waste - previous_waste)
        / previous_waste
    ) * 100

    if change > 10:

        st.warning(
            f"Simulated food waste is {change:.1f}% higher "
            "in the latest seven-day period. Review the "
            "dish-wise preparation quantities."
        )

    elif change < -10:

        st.success(
            f"Simulated food waste is {abs(change):.1f}% "
            "lower than in the previous seven-day period. "
            "Continue monitoring before drawing conclusions."
        )

    else:

        st.info(
            "Simulated food waste is relatively stable "
            "between these two periods."
        )

else:

    st.info(
        "There is not enough historical data for this comparison."
    )


# ---------------- EXPORT ----------------

st.divider()

st.markdown("## 📥 Export Your Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Canteen Data (CSV)",
    data=csv_data,
    file_name="smartcanteen_data.csv",
    mime="text/csv"
)


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "SmartCanteen AI | Innovation Challenge Prototype"
)