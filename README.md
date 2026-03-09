# Bilingual AI Essay Grading System

An end-to-end, Bilingual (English & Arabic) Artificial Intelligence system for Automated Essay Scoring (AES) and Handwritten Text Recognition (HTR).

This system bridges the gap between physical paper exams and digital AI assessment. It is capable of reading noisy, handwritten essays on ruled paper, extracting linguistic and semantic features, and generating both quantitative scores (0-10) and qualitative pedagogical feedback in real-time.

## 🌟 Key Features
- **High-Performance Scaling:** Achieves a near-human **0.91 Quadratic Weighted Kappa (QWK)** using a Support Vector Regression (SVR) engine working on 1550-dimensional Transformer embeddings.
- **Bilingual OCR pipeline:** 
  - English: Uses `microsoft/trocr-base-handwritten` combined with a horizontal projection-profile line segmentation algorithm to natively handle paragraph structures and ruled notebook lines.
  - Arabic: Utilizes `EasyOCR` to handle Right-to-Left (RTL) cursive scripts quickly and efficiently on CPU.
- **Formative Feedback:** Generates category-specific advice on Grammar, Structure, Content, and Vocabulary dynamically depending on the detected language.
- **Interactive UI:** A highly polished, responsive Streamlit dashboard.

---

## 📂 Project Structure

```text
Bilingual_AES_System/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── config.py                  # Global configurations (Model names, paths)
├── README.md                  # This documentation
├── ocr/                       # Handwritten Text Recognition Module
│   └── handwriting_recognizer.py
├── scoring/                   # Automated Essay Scoring Engine
│   └── scoring_agent_wrapper.py
├── feedback/                  # Rule-based Bilingual Feedback Engine
│   └── feedback_generator.py
├── agents/                    # Feature Extraction & Model Definition
│   ├── feature_agent.py       # MPNet Transformer embeddings & Linguistics
│   ├── novelty_agent.py       # TTR and Centroid novelty calculation
│   └── scoring_agent.py       # ML Architectures (SVR, MLP, Ridge)
└── checkpoints/               # Trained model weights
    └── best_model_exp_20260307_224545_e487fada.pkl  # 0.91 QWK SVR Matrix
```

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
Open your terminal/command prompt, navigate to the project directory, and install the required modules:

```bash
pip install -r requirements.txt
```

*(Note: PyTorch and Transformers must be installed. It is highly recommended to install the PyTorch version that corresponds to your specific GPU/CUDA setup for faster English OCR inference. If you do not have a GPU, the CPU version will work, though TrOCR will be slower).*

### 3. Launch the Streamlit App
Run the following command from the root of the project:

```bash
python -m streamlit run app.py
```

Streamlit will start a local server, usually accessible at `http://localhost:8501`.

---

## 📖 Walkthrough & Usage Guide

### A. Input Methods
The application supports three modes of input via the left navigation panel:
1. **Paste Text:** Directly paste digital essays.
2. **Upload File:** Upload raw `.txt` files.
3. **Handwriting OCR:** Upload a `.jpg` or `.png` file of a handwritten essay.

### B. Language Selection (Crucial!)
To guarantee accurate OCR processing, check the **Settings Sidebar** on the left:
- **Auto-Detect / English:** Uses TrOCR, tuned specifically for English handwriting. 
- **Arabic:** You MUST select Arabic from the dropdown if you are uploading an Arabic handwriting image. This bypasses TrOCR and uses EasyOCR alongside a specific right-to-left layout handler to process the script correctly. 

### C. Extracting Text & Grading
1. Upon uploading an image, click **"🚀 Extract Text from Image"**.
2. Wait for the engine to denoise the image, segment the paragraph lines, and perform sequence decoding.
3. Review the extracted text in the "Results & Feedback" panel.
4. Click **"📈 Grade Essay"** to marshal the text through the 1550-dimensional feature pipeline and receive your final score and detailed feedback.

---

## 🛠️ Architecture Methodology

This project operates on a three-stage pipeline:
1. **Morphological Pipeline:** A custom hybrid algorithm handles denoise, horizontal/vertical rule-line removal, text-smearing, and box-projection to smartly feed single-line crops to the Vision Transformers.
2. **Feature Engineering:** Student text is embedded against "Model Answers" using Multilingual Sentence-Transformers. The vectors are enriched with deep linguistic metadata (Structural Entropy, Lexical Density, Stem Ratios).
3. **Regression:** Features are passed through an RBF-Kernel Support Vector Regressor which standardizes the inputs and maps them accurately against known human-graded rubrics.
