# DAIN AI: Component-Based Pharmacodynamic Prediction System

Welcome to the **DAIN AI** project! 

Imagine being able to test a new chemical compound on thousands of virtual patients before ever running a real-world clinical trial. DAIN AI is an *in silico* (computer-simulated) clinical trial platform. Instead of relying on traditional, drug-specific empirical models, DAIN AI leverages deep learning to predict how any given chemical compound might change a patient's lab biomarkers (like cholesterol, glucose, or calcium levels) over time.

Whether you are a researcher, a developer, or an AI enthusiast, this guide will walk you through what the project does, how it works under the hood, and how to run it yourself.

---

## 🧠 How It Works (The AI Pipeline)

DAIN AI works by combining four distinct neural network models into a single, cohesive timeline. Here is a high-level overview of the pipeline:

### 1. Generating Virtual Patients (Conditional GAN)
Before we can test a drug, we need patients. We use a **Generative Adversarial Network (GAN)** trained on real-world demographic data from the NHANES dataset (approx. 23,000 patients). 
* **What it does:** You provide basic demographics (Age, Sex, Height, Weight), and the GAN generates a highly realistic "virtual patient" complete with 41 distinct features, including baseline lab results and body measurements.

### 2. Understanding the Drug (Hybrid Drug Encoder)
To predict what a drug will do, the AI needs to "read" its chemical structure. 
* **What it does:** We take the drug's SMILES string (a text representation of its chemical structure) and pass it through a **Transformer** model combined with traditional chemoinformatics capabilities (RDKit). This creates a dense, 768-dimensional mathematical representation of the drug's exact properties and functional groups.

### 3. Predicting the Effect (Cross-Attention Transformer)
Now we have our virtual patient and our mathematical drug representation. 
* **What it does:** We use a **Cross-Attention mechanism** (similar to how modern language models work) where the "Drug" attends to the 41 "Patient Features". This model calculates the static shift (the delta) in 22 key lab biomarkers caused by the administered drug.

### 4. Simulating Over Time (Time Series Transformer)
A drug's effect changes over time depending on dosage, half-life, and patient adherence.
* **What it does:** Our final **Transformer** model converts the static biomarker predictions into a temporal trajectory (e.g., predicting biomarker levels at day 10, 30, or 180). This data is then formatted to be visualized on our web dashboard.

---

## 🛠️ Tech Stack

- **Backend / API**: FastAPI (`app.py`), Uvicorn
- **Machine Learning**: PyTorch, Transformers (Hugging Face)
- **Chemoinformatics**: RDKit 
- **Data Processing**: Polars, Pandas, NumPy, SciPy, Scikit-learn
- **Frontend**: Vanilla JS, HTML, CSS (served from the `website/` directory)

---

## 📂 Project Structure

```plaintext
DAIN-AI/
├── data/                      # Raw and auxiliary data files
├── preprocessed_nhanes/       # Directory containing preprocessed patient Generation data
├── models/                    # Saved PyTorch checkpoint weights (.pt files)
├── src/
│   ├── models/                # GAN, PD Predictor, and Time Series Predictor implementations
│   ├── encoders/              # Drug Encoder implementations
│   ├── training/              # Training routines and scripts
│   └── utils/                 # Utility scripts
├── website/                   # Frontend dashboard (HTML/JS/CSS)
├── app.py                     # Main FastAPI application and API routing
├── main.py                    # Legacy/Alternative entry point
├── test.py                    # Test suites and simulation verifications
├── requirements.txt           # Python dependencies
└── MODEL_ARCHITECTURES.md     # Deep dive into network layers, dims, and parameters
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **RDKit compatibility**: We recommend using a Conda/Mamba environment or simply `pip install rdkit`.
- If you have an NVIDIA GPU, ensure your PyTorch installation matches your CUDA toolkit version for hardware acceleration. If no GPU is found, the system will seamlessly fall back to CPU.

### 1. Clone the Repository
```bash
git clone <repository-url>
cd DAIN-AI
```

### 2. Create a Virtual Environment
It is highly recommended to isolate your dependencies:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Model Weights
Before running the server, ensure that your pre-trained PyTorch weights (`.pt` files) are located inside the `models/` directory, and that the `preprocessed_nhanes/nhanes_final_complete.csv` data source is present.

---

## 🏃 Running the Application

### The Server and Web Dashboard
Start the FastAPI application by running:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Once the server is running, open your web browser and navigate to [http://localhost:8000](http://localhost:8000) to access the interactive simulation dashboard.

*(Windows users can also simply double-click or run the `.\run_web.ps1` script)*

---

## 🔌 API Reference

DAIN AI exposes a RESTful API if you wish to run pipeline simulations programmatically without using the web UI.

### `POST /api/simulate`
Executes the full pipeline for a single patient or a batch cohort.

**Example Request:**
```json
{
  "age": 45,
  "sex": "M",
  "height": 175.0,
  "weight": 80.0,
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "drug_name": "Aspirin",
  "dosage": 10.0,
  "total_hours": 3.0,
  "interval_seconds": 30,
  "patient_count": 1, 
  "volume_l": 50.0,
  "half_life_min": 60.0
}
```

### `POST /api/simulate/stream`
A Streaming Interface (Server-Sent Events) designed for large cohort predictions. It streams the generation progress back to the client continuously to prevent browser timeouts when generating thousands of patients.

---

## 📖 Further Reading
If you are modifying the deep learning architecture or want to dive entirely into the mathematics, please read the provided `MODEL_ARCHITECTURES.md` document. It contains detailed layer-by-layer specifications for all the PyTorch models used in this project.
