import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Auction Model Dashboard", layout="wide")
st.title("Auction Model Dashboard")

METRICS_CSV = Path("newProject/reports/metrics.csv")
FIG_DIR = Path("reports/figs")

MODEL_IMAGES = {
    "Naive Baseline": [FIG_DIR/"scatter_baseline.png"],
    "Linear Regression": [FIG_DIR/"scatter_linear.png"],
    "Random Forest": [FIG_DIR/"scatter_rf.png"],
    "Prophet (daily demand)": [
        FIG_DIR/"image.png",
        FIG_DIR/"image1.png"
    ],
}


if not METRICS_CSV.exists():
    st.error(f"Missing {METRICS_CSV}. Generate it from your notebook first.")
    st.stop()

metrics = pd.read_csv(METRICS_CSV)
st.subheader("Model Performance (MAE, RMSE, MAPE)")
st.dataframe(metrics)

bar_path = FIG_DIR/"metrics_bars.png"
if bar_path.exists():
    st.subheader("Comparison Chart")
    st.image(Image.open(bar_path), use_container_width=True)
else:
    st.info(f"Generate {bar_path} in your notebook to show the comparison chart.")

st.sidebar.header("Viewer")
choices = list(MODEL_IMAGES.keys())
model_choice = st.sidebar.selectbox("Select model", choices)

st.subheader(f"Plots for: {model_choice}")

imgs = MODEL_IMAGES.get(model_choice, [])
found_any = False
cols = st.columns(2)

for i, p in enumerate(imgs):
    if p.exists():
        with cols[i % 2]:
            st.image(Image.open(p), caption=p.name, use_container_width=True)
        found_any = True

if not found_any:
    st.warning("")

pred_csv = Path(f"{model_choice.lower().replace(' ', '_').replace('(', '').replace(')', '')}_predictions.csv")
if pred_csv.exists():
    st.download_button(
        label=f"Download {model_choice} predictions CSV",
        data=pred_csv.read_bytes(),
        file_name=pred_csv.name,
        mime="text/csv"
    )
