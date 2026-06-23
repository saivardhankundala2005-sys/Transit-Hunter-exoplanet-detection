# Transit Hunter: AI Exoplanet Detection Pipeline from Noisy TESS Light Curves

An end-to-end, production-grade exoplanet detection pipeline developed for the **Bharatiya Antariksh Hackathon 2026**. This project cleans noisy TESS light curves, detects periodic transits, extracts physical features, classifies candidates using a hybrid deep-learning + machine-learning ensemble, fits physical orbit models, and provides explainable AI attributions in an interactive, space-themed dashboard.

---

## 🚀 Features
- **Data Acquisition:** Automatically downloads light curves from MAST using `lightkurve`.
- **Systematic Cleaning:** Removes low-frequency stellar rotation and systematics using `wotan` biweight window filters and sigma clipping.
- **Signal Search:** Runs Box Least Squares (BLS) and Transit Least Squares (TLS) to isolate periodic dips.
- **Feature Engineering:** Extracts 11 physics-based shape, slope, symmetry, and noise metrics.
- **Hybrid AI Classifier:** Combines a 1D Dual-View Convolutional Neural Network (PyTorch) with an XGBoost/Random Forest physics classifier.
- **Keplerian Orbital Fitting:** Optimizes physical transit models (using `batman` or trapezoids) with Bootstrap uncertainty boundaries.
- **Scientific Vetting:** Performs centroid shift tests, odd-even depth t-tests, and model significance F-tests to calculate an overall reliability score.
- **Explainable AI:** Grad-CAM overlays (CNN branch) and feature sensitivity analysis (ML branch).
- **Streamlit Dashboard:** Interactive 7-page analytical viewer.

---

## 📂 Project Structure
```text
project/
│
├── configs/
│   └── config.yaml          # Pipeline hyperparameters & directories config
│
├── data/                    # Local cache and synthetic data folder
│
├── models/                  # Saved CNN and ML weights checkpoints
│
├── outputs/                 # Diagnostic JSON reports, fit curves, explainability plots
│
├── reports/
│   └── scientific_report.md # Technical report layout for the hackathon
│
├── src/
│   ├── acquisition/         # Downloader and synthetic transit injector
│   ├── preprocessing/       # Outlier clipping & Wotan detrending
│   ├── transit_search/      # BLS, TLS refinement, and binned phase folding
│   ├── feature_engineering/ # 11 Physics feature calculations
│   ├── models/              # DualViewCNN & Hybrid Ensemble models
│   ├── fitting/             # batman-package orbit fitting & bootstrap
│   ├── validation/          # Centroid, Odd-Even, F-test vettings
│   ├── explainability/      # Grad-CAM 1D & feature attributions
│   └── dashboard/           # Space-themed 7-page Streamlit application
│
├── tests/
│   └── test_pipeline.py     # Automated unit tests suite
│
├── Dockerfile               # Production containerization build
├── requirements.txt         # Project package requirements list
├── train_pipeline.py        # Synthetic dataset generator & model training runner
├── run_pipeline.py          # End-to-end command-line runner for target TIC IDs
└── README.md                # Installation and usage instructions
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8, 3.9, or 3.10
- Compilers (`gcc`, `g++`, `make`) to compile C-extensions (`batman-package`, `wotan`)

### 1. Install Dependencies
```bash
# Clone the repository
git clone https://github.com/saivardhankundala2005-sys/Transit-Hunter-exoplanet-detection.git
cd Transit-Hunter-exoplanet-detection

# Install python dependencies
pip install -r requirements.txt
```

---

## 📦 Running the Pipeline

### Step 1: Train the Models
Generate synthetic datasets (injecting exoplanet transits, binaries, blends, and artifacts) and train the CNN and ML classifiers:
```bash
python train_pipeline.py
```
This will compile training sets and write `cnn_model.pt` and `physics_model.pkl` to the `models/` directory.

### Step 2: Execute for a Target TIC ID
Run the end-to-end exoplanet analysis on a real TESS target (e.g. TIC 261108232):
```bash
python run_pipeline.py --tic 261108232
```
To test robustness, you can also inject a fake transit into a real light curve:
```bash
python run_pipeline.py --tic 261108232 --inject --period 4.25 --depth 0.007 --duration 0.15
```
Outputs (JSON reports, fitting graphs, Grad-CAM overlays, before-after detrending charts) will be exported to `outputs/tic_<ID>/`.

### Step 3: Run the Streamlit Dashboard
Launch the interactive 7-page analysis dashboard:
```bash
streamlit run src/dashboard/app.py
```

### Step 4: Run Automated Tests
Execute the unit test suite to verify code integrity:
```bash
python -m unittest tests/test_pipeline.py
```

---

## 🐳 Docker Deployment
To build and run the pipeline inside a container:
```bash
# Build the Docker image
docker build -t transit-hunter .

# Run the container (hosts the Streamlit app at http://localhost:8501)
docker run -p 8501:8501 transit-hunter
```
