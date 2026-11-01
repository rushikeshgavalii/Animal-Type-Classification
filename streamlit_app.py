"""Streamlit demo. Run: streamlit run app/streamlit_app.py"""
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from ipshakti.model import load_checkpoint
from ipshakti.predict import classify
from ipshakti.segment import overlay_mask, segment_animal
from ipshakti.traits import estimate_traits

st.set_page_config(page_title="IP Shakti Sahayak", page_icon="🐃", layout="wide")
st.title("🐃 IP Shakti Sahayak")
st.caption("Image-based Animal Type Classification for cattle and buffaloes")

device = "cuda" if torch.cuda.is_available() else "cpu"

with st.sidebar:
    ckpt_path = st.text_input("Breed model checkpoint", "models/breed_classifier.pt")
    pixels_per_cm = st.number_input("Pixels per cm (0 = unknown)", min_value=0.0, value=0.0, step=0.1,
                                    help="Place a ruler or known-size marker beside the animal to get this.")
    facing = st.selectbox("Head direction", ["auto", "left", "right"])
    st.markdown("**Tip:** use a clear, full-body **side-view** photo on level ground.")


@st.cache_resource
def get_model(path):
    return load_checkpoint(path, device)


up = st.file_uploader("Upload a side-view photo", type=["jpg", "jpeg", "png"])
cam = st.camera_input("...or take a photo")
file = up or cam

if file:
    img = Image.open(file).convert("RGB")
    left, right = st.columns(2)
    report = {}

    with st.spinner("Analysing..."):
        seg = segment_animal(img, device=device)
        shown = img
        if seg is None:
            right.warning("No cattle or buffalo detected. Try a clearer full-body side view.")
        else:
            mask, conf = seg
            shown = overlay_mask(img, mask)
            try:
                res = estimate_traits(mask, pixels_per_cm or None, facing)
                report["traits"] = res.to_dict()
                right.subheader("Body traits (1-9 scale)")
                right.json(res.to_dict())
            except ValueError as e:
                right.warning(f"Could not score traits: {e}")

        if Path(ckpt_path).exists():
            model, names = get_model(ckpt_path)
            preds = classify(model, names, img, device=device)
            report["breed_prediction"] = preds
            top = preds[0]
            right.subheader("Breed")
            right.metric(f"{top['breed']} ({top['species']})", f"{top['confidence']*100:.1f}%")
            right.table(preds)
        else:
            right.info("No trained checkpoint found. Train one with `python -m ipshakti.train`.")

    left.image(shown, caption="Input (detected animal highlighted)", use_container_width=True)
    st.download_button("Download report (JSON)", json.dumps(report, indent=2), "report.json")
