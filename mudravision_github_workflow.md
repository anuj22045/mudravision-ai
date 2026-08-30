# 🇮🇳 MudraVision AI — GitHub Workflow Guide

> This document tells you exactly which **branches to create**, which **issues to open**, and how to work through the project on GitHub following the implementation plan.

---

## 🌿 Branching Strategy

Use a **feature-branch workflow**:

```
main
  └── dev                    ← integration branch (merge phases here)
        ├── phase-1/dataset-exploration
        ├── phase-1/cnn-preprocessing
        ├── phase-1/cnn-training
        ├── phase-1/model-evaluation
        ├── phase-2/prediction-pipeline
        ├── phase-3/ocr-setup
        ├── phase-3/nlp-processing
        ├── phase-4/coin-knowledge-base
        └── phase-4/streamlit-app
```

### Rules
- `main` → only holds stable, working code. Merge here only after a phase is 100% done and tested.
- `dev` → active development. Merge feature branches here first.
- Each branch is for **one specific task** from the implementation plan.
- When a task is done → open a **Pull Request** from feature branch → `dev`.
- After a phase is complete → merge `dev` → `main`.

---

## 📌 How to Create Branches

```bash
# Create and switch to a new branch from dev
git checkout dev
git pull origin dev
git checkout -b phase-1/dataset-exploration
```

When done with work on a branch:
```bash
git add .
git commit -m "feat: complete data exploration notebook (#1)"
git push origin phase-1/dataset-exploration
# Then open a Pull Request on GitHub: phase-1/dataset-exploration → dev
```

---

## 🐛 GitHub Issues to Create

Create all the issues below in your GitHub repository. Each issue maps to a step in the implementation plan. You will **close each issue by referencing it in your commit message** using `closes #<issue-number>`.

---

### 📁 Milestone: Phase 1 — Dataset + CNN

> Create a GitHub Milestone called **"Phase 1: Dataset + CNN"** and attach issues #1–#6 to it.

---

#### Issue #1 — Complete Data Exploration Notebook

**Title**: `[Phase 1] Complete data exploration notebook`

**Labels**: `phase-1`, `notebook`, `in-progress`

**Branch**: `phase-1/dataset-exploration`

**Description**:
```
Finish the 1_data_exploration.ipynb notebook.

Tasks:
- [ ] Count and display total images per class (train + test)
- [ ] Display sample images from each coin class
- [ ] Check and print image dimensions
- [ ] Visualize class distribution using a bar chart
- [ ] Add markdown cells explaining what the data looks like

Closes when: notebook is complete, all cells run without error, and class distribution is clear.
```

---

#### Issue #2 — Implement Image Preprocessing Module

**Title**: `[Phase 1] Create src/preprocessing.py`

**Labels**: `phase-1`, `feature`

**Branch**: `phase-1/cnn-preprocessing`

**Description**:
```
Create the src/preprocessing.py module.

Tasks:
- [ ] Function to resize images to 128x128 (or 64x64)
- [ ] Function to normalize pixel values to [0, 1]
- [ ] Set up ImageDataGenerator with augmentation:
    - horizontal_flip
    - rotation_range
    - zoom_range
    - width_shift_range
    - height_shift_range
- [ ] Add docstrings explaining why each step is done
- [ ] Test the module by printing a sample augmented batch shape

Closes when: preprocessing.py is created and tested.
```

---

#### Issue #3 — Build CNN Architecture and Train Model

**Title**: `[Phase 1] Build and train CNN in 2_cnn_training.ipynb`

**Labels**: `phase-1`, `notebook`, `ml`

**Branch**: `phase-1/cnn-training`

**Description**:
```
Build and train the coin classifier CNN from scratch.

Tasks:
- [ ] Load data using ImageDataGenerator (train + validation split)
- [ ] Build CNN architecture:
    - Conv2D → MaxPooling2D (x2 or x3)
    - Flatten
    - Dense (with ReLU)
    - Dropout
    - Dense output (softmax, 4 classes)
- [ ] Compile with Adam optimizer + categorical_crossentropy
- [ ] Train for 20-30 epochs with validation_data
- [ ] Plot training vs validation accuracy and loss curves
- [ ] Add markdown explanations for each layer and concept

Closes when: model trains successfully and plots are generated.
```

---

#### Issue #4 — Save Trained Model

**Title**: `[Phase 1] Save model to models/coin_classifier.keras`

**Labels**: `phase-1`, `feature`

**Branch**: `phase-1/cnn-training` *(same branch as Issue #3)*

**Description**:
```
Save the trained model after training completes.

Tasks:
- [ ] Add model.save("models/coin_classifier.keras") to notebook
- [ ] Verify the file is created in the models/ directory
- [ ] Add a note explaining .keras vs .h5 format

Closes when: coin_classifier.keras is saved and loadable.
```

---

#### Issue #5 — Create src/train.py Script

**Title**: `[Phase 1] Create src/train.py — standalone training script`

**Labels**: `phase-1`, `feature`

**Branch**: `phase-1/cnn-training`

**Description**:
```
Convert the training notebook logic into a reusable script.

Tasks:
- [ ] Import preprocessing utilities from preprocessing.py
- [ ] Build the CNN model
- [ ] Train with callbacks (e.g., ModelCheckpoint, EarlyStopping)
- [ ] Save model to models/coin_classifier.keras
- [ ] Add argparse or constants for easy configuration (image size, epochs, batch size)

Closes when: python src/train.py runs successfully and saves the model.
```

---

#### Issue #6 — Evaluate Model and Create Evaluation Notebook

**Title**: `[Phase 1] Build model evaluation in 3_model_evaluation.ipynb`

**Labels**: `phase-1`, `notebook`, `ml`

**Branch**: `phase-1/model-evaluation`

**Description**:
```
Evaluate the trained model on the test set.

Tasks:
- [ ] Load coin_classifier.keras
- [ ] Generate predictions on the test set
- [ ] Plot confusion matrix (using seaborn or matplotlib)
- [ ] Print classification report (precision, recall, F1-score per class)
- [ ] Display sample images with their predicted vs actual labels
- [ ] Add Phase 1 Revision Section as markdown cells

Closes when: evaluation notebook is complete with all metrics visible.
```

---

### 📁 Milestone: Phase 2 — CNN Prediction Pipeline

> Create a GitHub Milestone called **"Phase 2: CNN Prediction Pipeline"** and attach issues #7–#8.

---

#### Issue #7 — Create Prediction Module

**Title**: `[Phase 2] Create src/predict.py — prediction pipeline`

**Labels**: `phase-2`, `feature`

**Branch**: `phase-2/prediction-pipeline`

**Description**:
```
Build the prediction pipeline for new coin images.

Tasks:
- [ ] Load coin_classifier.keras
- [ ] Function: preprocess_image(image_path) → returns normalized array
- [ ] Function: predict_coin(image_path) → returns (class_label, confidence_score)
- [ ] Explain what model.predict() returns (softmax probabilities)
- [ ] Explain np.argmax() to get the predicted class
- [ ] Add docstrings for every function

Closes when: predict.py returns correct denomination and confidence for a test image.
```

---

#### Issue #8 — Test Prediction on Sample Images

**Title**: `[Phase 2] Test prediction pipeline with sample coin images`

**Labels**: `phase-2`, `testing`

**Branch**: `phase-2/prediction-pipeline`

**Description**:
```
Manually test the prediction module on sample images.

Tasks:
- [ ] Pick 4 test images (one per class) from dataset/test/
- [ ] Run predict_coin() on each
- [ ] Print: image path, predicted denomination, confidence %
- [ ] Compare predicted vs actual labels
- [ ] Add Phase 2 Revision Section as comments or a markdown file

Closes when: all 4 test images return correct predictions (or failures are explained).
```

---

### 📁 Milestone: Phase 3 — OCR + NLP

> Create a GitHub Milestone called **"Phase 3: OCR + NLP"** and attach issues #9–#12.

---

#### Issue #9 — Install and Configure Tesseract OCR

**Title**: `[Phase 3] Install Tesseract OCR and configure pytesseract`

**Labels**: `phase-3`, `setup`

**Branch**: `phase-3/ocr-setup`

**Description**:
```
Set up Tesseract OCR for text extraction from coin images.

Tasks:
- [ ] Install Tesseract binary (Windows: Tesseract installer from UB Mannheim)
- [ ] Install pytesseract via pip
- [ ] Set tesseract_cmd path in code (Windows specific)
- [ ] Test with a sample coin image: print extracted text
- [ ] Update requirements.txt with pytesseract

Closes when: pytesseract successfully extracts text from a coin image.
```

---

#### Issue #10 — Create OCR Module

**Title**: `[Phase 3] Create src/ocr.py — text extraction from coin images`

**Labels**: `phase-3`, `feature`

**Branch**: `phase-3/ocr-setup`

**Description**:
```
Build the OCR module to extract text from coin images.

Tasks:
- [ ] Function: preprocess_for_ocr(image_path) → grayscale + thresholded image
    - Use OpenCV: cv2.cvtColor, cv2.threshold
- [ ] Function: extract_text(image_path) → raw text string
    - Use pytesseract.image_to_string()
- [ ] Handle edge cases (empty string, no text detected)
- [ ] Add docstrings explaining why each preprocessing step helps OCR

Closes when: ocr.py extracts visible text from at least 2 coin images.
```

---

#### Issue #11 — Create NLP Processing Module

**Title**: `[Phase 3] Create src/nlp.py — NLP processing of OCR text`

**Labels**: `phase-3`, `feature`, `nlp`

**Branch**: `phase-3/nlp-processing`

**Description**:
```
Build the NLP module to clean and process OCR-extracted text.

Tasks:
- [ ] Download NLTK resources: punkt, stopwords, wordnet
- [ ] Function: clean_text(raw_text) → lowercase, remove special chars, strip whitespace
- [ ] Function: tokenize(text) → list of tokens
- [ ] Function: remove_stopwords(tokens) → filtered tokens
- [ ] Function: lemmatize(tokens) → root form tokens
- [ ] Function: process_coin_text(raw_text) → returns dict with all steps
- [ ] Add clear explanation of which NLP steps are useful for coin text vs just educational

Closes when: nlp.py processes OCR output and returns cleaned keywords.
```

---

#### Issue #12 — Connect CNN + OCR + NLP Pipeline

**Title**: `[Phase 3] Integrate CNN prediction + OCR + NLP into unified pipeline`

**Labels**: `phase-3`, `integration`

**Branch**: `phase-3/nlp-processing`

**Description**:
```
Connect all three components for a single image input.

Tasks:
- [ ] Function: analyze_coin(image_path) → returns:
    {
      "denomination": "₹5",
      "confidence": 94.2,
      "raw_text": "INDIA 5 RUPEES",
      "cleaned_text": "india 5 rupees",
      "keywords": ["india", "rupee"]
    }
- [ ] Test analyze_coin() on 2-3 coin images
- [ ] Print full output clearly
- [ ] Add Phase 3 Revision Section as comments

Closes when: analyze_coin() returns all three outputs for a given image.
```

---

### 📁 Milestone: Phase 4 — Streamlit Application

> Create a GitHub Milestone called **"Phase 4: Streamlit Application"** and attach issues #13–#15.

---

#### Issue #13 — Build Coin Knowledge Base

**Title**: `[Phase 4] Create src/coin_info.py — coin knowledge base`

**Labels**: `phase-4`, `feature`

**Branch**: `phase-4/coin-knowledge-base`

**Description**:
```
Create a local knowledge base for coin information.

Tasks:
- [ ] Create a Python dictionary (or JSON file) with info for:
    - ₹1: composition, year, notable features, obverse/reverse description
    - ₹2: same
    - ₹5: same
    - ₹10: same
- [ ] Function: get_coin_info(denomination) → returns dict
- [ ] Handle unknown denomination gracefully

Closes when: get_coin_info("5") returns correct coin data.
```

---

#### Issue #14 — Build Streamlit UI

**Title**: `[Phase 4] Build Streamlit application in app.py`

**Labels**: `phase-4`, `feature`, `ui`

**Branch**: `phase-4/streamlit-app`

**Description**:
```
Build the main Streamlit web app.

Tasks:
- [ ] Image upload widget (st.file_uploader)
- [ ] Display uploaded coin image
- [ ] Button: "Analyze Coin"
- [ ] Show CNN result: denomination + confidence (use st.progress or st.metric)
- [ ] Show OCR result: extracted text
- [ ] Show NLP result: cleaned keywords
- [ ] Show Coin Info: from knowledge base (as a nice card/table)
- [ ] Add spinner (st.spinner) for each processing step
- [ ] Handle errors gracefully (e.g., no text found by OCR)

Closes when: app.py runs with streamlit run app.py and correctly processes a coin image.
```

---

#### Issue #15 — Final Testing and README Update

**Title**: `[Phase 4] Final end-to-end testing + update README.md`

**Labels**: `phase-4`, `documentation`, `testing`

**Branch**: `phase-4/streamlit-app`

**Description**:
```
Final testing and documentation.

Tasks:
- [ ] Test the full app with all 4 coin denominations
- [ ] Verify CNN prediction, OCR, NLP, and coin info all display correctly
- [ ] Update README.md with:
    - Project description
    - Tech stack
    - Setup instructions (install requirements, run app)
    - Screenshots of the running app
    - Architecture diagram (data flow)
    - Future upgrades section
- [ ] Add Phase 4 Revision Section
- [ ] Tag v1.0 release on GitHub

Closes when: app works end-to-end and README is complete.
```

---

## 🏷️ Labels to Create on GitHub

Go to your GitHub repo → Issues → Labels → and create these:

| Label | Color | Purpose |
|---|---|---|
| `phase-1` | `#0075ca` | Phase 1 tasks |
| `phase-2` | `#e4e669` | Phase 2 tasks |
| `phase-3` | `#d93f0b` | Phase 3 tasks |
| `phase-4` | `#0e8a16` | Phase 4 tasks |
| `notebook` | `#cccccc` | Jupyter notebook work |
| `feature` | `#a2eeef` | New feature/module |
| `ml` | `#5319e7` | Machine learning specific |
| `nlp` | `#f9d0c4` | NLP specific |
| `setup` | `#bfd4f2` | Environment/setup tasks |
| `testing` | `#fef2c0` | Testing tasks |
| `integration` | `#c5def5` | Connecting components |
| `ui` | `#d4c5f9` | UI/frontend work |
| `documentation` | `#0075ca` | Docs and README |
| `in-progress` | `#fbca04` | Currently being worked on |

---

## 🔁 Workflow Summary

```
1. Pick an issue from the list above
2. Create the branch from dev (naming matches branch column)
3. Do the work
4. Commit with message referencing the issue:
   e.g., "feat: add preprocessing module (closes #2)"
5. Push branch and open PR: feature-branch → dev
6. Merge PR when issue tasks are all checked off
7. After all issues in a Phase Milestone are closed → merge dev → main
8. Create a GitHub Release/Tag at the end of each phase
```

---

## 🚀 Release Tags

| Tag | When to create |
|---|---|
| `v0.1-data-exploration` | After Issue #1 is closed |
| `v0.2-cnn-trained` | After Issues #2–#6 are closed (Phase 1 complete) |
| `v0.3-prediction` | After Issues #7–#8 are closed (Phase 2 complete) |
| `v0.4-ocr-nlp` | After Issues #9–#12 are closed (Phase 3 complete) |
| `v1.0-mvp` | After Issues #13–#15 are closed (Phase 4 complete) |
