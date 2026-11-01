# 🐃 IP Shakti Sahayak

**Image-based Animal Type Classification (ATC) for cattle and buffaloes.**

Upload a side-view photo of an animal and get:

1. **Species and breed** from a transfer-learning classifier (EfficientNet-B0).
2. **Body traits on a 1-9 scale** (body length, body depth, rump angle, and stature when a scale is known), measured from the animal's silhouette.
3. A downloadable JSON report, through a Streamlit web app.

## How it works

```
photo ──► Mask R-CNN (pretrained) ──► animal silhouette ──► trait measurements ──► 1-9 scores
   └────► EfficientNet-B0 (your trained model) ──────────► breed + species + confidence
```

| Module | Purpose |
|---|---|
| `src/ipshakti/train.py` | Train the breed classifier |
| `src/ipshakti/predict.py` | CLI inference (breed + traits) |
| `src/ipshakti/segment.py` | Animal segmentation (pretrained, no training needed) |
| `src/ipshakti/traits.py` | Silhouette to measurements to 1-9 scores |
| `app/streamlit_app.py` | Web demo with camera/upload |
| `scripts/split_dataset.py` | Split raw images into train/val |

## Quick start

```bash
git clone https://github.com/<your-username>/ip-shakti-sahayak.git
cd ip-shakti-sahayak
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### 1. Prepare data
Follow [`data/README.md`](data/README.md). You need labelled images per breed (`data/train/<breed>/`, `data/val/<breed>/`).
No dataset is bundled. Use your own field photos, preferably full-body side views.

### 2. Train
```bash
python -m ipshakti.train --data-dir data --epochs 15 --out models/breed_classifier.pt
```

### 3. Run
```bash
# Web app
streamlit run app/streamlit_app.py

# Command line
python -m ipshakti.predict --image sample.jpg --pixels-per-cm 2.5 --save-overlay out.png
```

### 4. Test
```bash
pytest
```

## Photo guidelines (for good trait scores)

- Full body in frame, **true side view**, camera at mid-body height.
- Animal standing square on level ground, head up.
- For **stature in cm**, place a ruler or known-size marker beside the animal and compute `pixels_per_cm`. Without it, stature is skipped.

## Limitations (please read)

- Trait scoring is a **rule-based prototype**. The 1-9 ranges in `traits.py` (`DEFAULT_RANGES`) are placeholders and must be **calibrated against expert-scored animals** before the results are used for breeding or registry decisions.
- Segmentation uses the COCO "cow" class, which works for cattle and buffaloes but can fail with heavy occlusion, multiple animals, or poor lighting.
- Breed accuracy depends entirely on your dataset's size and quality. Look-alike breeds and crossbreeds need more data.
- No trained model or dataset ships with this repo.

## Roadmap ideas

- Keypoint model (withers, hooks, pins, hocks) for more accurate traits
- Species-specific score ranges
- Add udder and teat traits from rear/side views
- Mobile-friendly export (ONNX / TFLite)
- Hindi and Marathi UI

## License

MIT, see [LICENSE](LICENSE).
