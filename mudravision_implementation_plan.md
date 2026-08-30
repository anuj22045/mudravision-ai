# 🇮🇳 MudraVision AI — Implementation Plan

> **Project Goal**: Build an AI-powered Indian Coin Recognition system that uses CNN to identify denominations, OCR to extract text, and NLP to process it — all wrapped in a Streamlit UI.  
> **Learning Focus**: Practically understand CNN + NLP by building from scratch.

---

## 📌 Current Status

| Item | Status |
|---|---|
| Dataset organized (train/test split) | ✅ Done |
| `prepare_dataset.py` | ✅ Done |
| `notebooks/1_data_exploration.ipynb` | 🔄 In Progress |
| `notebooks/2_cnn_training.ipynb` | ⬜ Pending |
| `notebooks/3_model_evaluation.ipynb` | ⬜ Pending |
| `src/` modules | ⬜ Pending |
| `app.py` (Streamlit) | ⬜ Pending |

---

## 🗂️ Project Structure (Final Target)

```
mudravision-ai/
│
├── dataset/
│   ├── train/
│   │   ├── 1_rupee/
│   │   ├── 2_rupee/
│   │   ├── 5_rupee/
│   │   └── 10_rupee/
│   └── test/
│       ├── 1_rupee/
│       ├── 2_rupee/
│       ├── 5_rupee/
│       └── 10_rupee/
│
├── notebooks/
│   ├── 1_data_exploration.ipynb   ← In Progress
│   ├── 2_cnn_training.ipynb       ← Phase 1
│   └── 3_model_evaluation.ipynb   ← Phase 1
│
├── src/
│   ├── preprocessing.py           ← Phase 1
│   ├── train.py                   ← Phase 1
│   ├── predict.py                 ← Phase 2
│   ├── ocr.py                     ← Phase 3
│   ├── nlp.py                     ← Phase 3
│   └── coin_info.py               ← Phase 4
│
├── models/
│   └── coin_classifier.keras      ← Phase 1 output
│
├── app.py                         ← Phase 4
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔵 Phase 1: Dataset + CNN

### Goal
Understand image data, build a CNN from scratch, train it, and evaluate it.

### Step-by-Step Plan

#### Step 1.1 — Complete `1_data_exploration.ipynb` *(In Progress)*
- [ ] Count images per class (train + test)
- [ ] Display sample images from each class
- [ ] Check image sizes/dimensions
- [ ] Visualize class distribution (bar chart)
- [ ] Understand what the data looks like before training
- **Files**: `notebooks/1_data_exploration.ipynb`

#### Step 1.2 — Image Preprocessing (`preprocessing.py`)
- [ ] Resize all images to a fixed size (128×128 or 64×64)
- [ ] Normalize pixel values from [0–255] → [0–1]
- [ ] Explain **why** normalization helps gradient descent
- [ ] Apply Data Augmentation (rotation, flip, zoom, shift)
- [ ] Explain **why** augmentation prevents overfitting
- **Files**: `src/preprocessing.py`

#### Step 1.3 — Build CNN in `2_cnn_training.ipynb`
- [ ] Create ImageDataGenerators for train + validation
- [ ] Explain each layer:
  - `Conv2D` → feature extraction
  - `MaxPooling2D` → downsampling
  - `Flatten` → connect CNN to Dense
  - `Dense` → classification
  - `Dropout` → regularization
- [ ] Compile with `Adam` optimizer + `categorical_crossentropy` loss
- [ ] Train for 20–30 epochs
- [ ] Plot training vs validation accuracy and loss curves
- **Files**: `notebooks/2_cnn_training.ipynb`

#### Step 1.4 — Save the Model
- [ ] Save model as `models/coin_classifier.keras`
- [ ] Explain the `.keras` vs `.h5` format

#### Step 1.5 — Evaluate in `3_model_evaluation.ipynb`
- [ ] Load saved model
- [ ] Predict on test set
- [ ] Plot confusion matrix
- [ ] Print classification report (precision, recall, F1)
- [ ] Show sample predictions with images
- **Files**: `notebooks/3_model_evaluation.ipynb`, `src/train.py`

#### ✅ Phase 1 Revision Section (to be written after completion)
- What is a CNN and how does it work?
- Why do we normalize images?
- What is overfitting and how does augmentation help?
- What does each layer (Conv, Pool, Dense, Dropout) do?
- How do accuracy and loss curves tell us about training?

---

## 🟠 Phase 2: CNN Prediction Pipeline

### Goal
Load the saved model and make predictions on new coin images.

### Step-by-Step Plan

#### Step 2.1 — Create `predict.py`
- [ ] Load `coin_classifier.keras`
- [ ] Preprocess a single input image (resize + normalize)
- [ ] Run `model.predict()`
- [ ] Return: predicted class label + confidence score (%)
- [ ] Explain what `softmax` output means (probability distribution)
- [ ] Explain `np.argmax()` to pick the winning class
- **Files**: `src/predict.py`

#### Step 2.2 — Test Prediction Manually
- [ ] Pick a test image from `dataset/test/`
- [ ] Run `predict.py` on it
- [ ] Print denomination and confidence
- [ ] Show predicted vs actual label

#### ✅ Phase 2 Revision Section (to be written after completion)
- What is a prediction pipeline?
- What does `softmax` produce?
- Why do we preprocess the image the same way during inference?
- What is confidence score?

---

## 🟡 Phase 3: OCR + NLP

### Goal
Extract visible text from coin images using Tesseract OCR, clean it with NLP, and combine with CNN output.

### Step-by-Step Plan

#### Step 3.1 — Setup Tesseract OCR
- [ ] Install Tesseract binary
- [ ] Install `pytesseract` Python wrapper
- [ ] Test with a sample image
- [ ] Explain what OCR is and how Tesseract works

#### Step 3.2 — Create `ocr.py`
- [ ] Use `pytesseract.image_to_string()` on coin image
- [ ] Use OpenCV for image preprocessing before OCR:
  - Convert to grayscale
  - Apply thresholding (improves OCR accuracy)
- [ ] Return raw extracted text
- **Files**: `src/ocr.py`

#### Step 3.3 — Create `nlp.py`
- [ ] Clean the raw OCR text (lowercase, remove special chars, strip whitespace)
- [ ] Tokenization → split text into individual words
- [ ] Stopword removal → remove "the", "is", "of" (using NLTK)
- [ ] Lemmatization → reduce words to root form (using NLTK WordNetLemmatizer)
- [ ] Explain clearly: which NLP techniques are **actually useful** for coin text vs which are just educational here
- [ ] Return: cleaned text, tokens, final processed keywords
- **Files**: `src/nlp.py`

#### Step 3.4 — Connect OCR+NLP with CNN Prediction
- [ ] For a given image:
  1. CNN → denomination + confidence
  2. OCR → raw text
  3. NLP → cleaned keywords
- [ ] Return all three results together

#### ✅ Phase 3 Revision Section (to be written after completion)
- What is OCR and how does Tesseract extract text?
- What is tokenization, stopword removal, and lemmatization?
- Which NLP steps are genuinely useful for short coin text?
- How do CNN + OCR + NLP work together?

---

## 🟢 Phase 4: Streamlit Application

### Goal
Wrap everything into a clean, usable web app.

### Step-by-Step Plan

#### Step 4.1 — Build Coin Knowledge Base (`coin_info.py`)
- [ ] Create a Python dictionary with info for each denomination:
  - ₹1: year introduced, metal, notable features
  - ₹2: year introduced, metal, notable features
  - ₹5: year introduced, metal, notable features
  - ₹10: year introduced, metal, notable features
- [ ] Function: `get_coin_info(denomination)` → returns dict
- **Files**: `src/coin_info.py`

#### Step 4.2 — Build `app.py` (Streamlit UI)
- [ ] Image upload widget
- [ ] On upload:
  1. Display uploaded image
  2. Run CNN → show denomination + confidence bar
  3. Run OCR → show extracted text
  4. Run NLP → show processed keywords
  5. Look up coin info from knowledge base → display
- [ ] Add loading spinners for each step
- [ ] Keep UI simple and clean
- **Files**: `app.py`

#### Step 4.3 — Final Architecture Explanation
- [ ] Draw the complete data flow:
  ```
  User uploads image
        ↓
  Image Preprocessing (resize, normalize)
        ↓                    ↓
   CNN Model           OpenCV preprocessing
        ↓                    ↓
  Denomination +        Tesseract OCR
  Confidence                 ↓
                       Raw Text → NLP cleaning
                             ↓
                       Cleaned Keywords
                             ↓
                   Coin Knowledge Base lookup
                             ↓
                   Streamlit UI Display
  ```

#### ✅ Phase 4 Revision Section (to be written after completion)
- What is Streamlit and why use it for ML apps?
- How does the full pipeline work end to end?
- Where does CNN fit vs OCR vs NLP?

---

## 🔮 Future Upgrades (Post Basic Project)

These are suggested only after the above 4 phases are complete and understood:

| Upgrade | Why |
|---|---|
| Transfer Learning (MobileNet/ResNet) | Better accuracy with pretrained weights |
| YOLOv8 for coin detection | Handle multiple coins in one image |
| EasyOCR instead of Tesseract | Better OCR for curved/faded text |
| Word Embeddings (Word2Vec) | Semantic similarity for text |
| RAG with LangChain | Retrieve coin history from a knowledge base |
| Database (SQLite) | Store prediction history |

---

## 📋 Key Rules (Reference)

1. Build CNN from scratch first — no transfer learning initially
2. Explain every concept before coding
3. Complete each phase before moving to the next
4. After each major step, test/run it first, then show results
5. Debug errors without replacing everything
6. Keep it simple — no LLMs, no transformers in v1
