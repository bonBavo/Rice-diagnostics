# Rice Disease Diagnostic System

**Case-study location:** Mwea, Kenya  
**Current implementation:** local Flask website and six-class TensorFlow/Keras image classifier  
**Project status:** working prototype; dataset evaluation is available, but Mwea field validation and user evaluation are not complete.

This README documents what the software currently does, how the data and model are prepared, how to run the application, what evidence is available, and what remains incomplete. It is written both as an operator's guide and as a sequence for explaining the project to a lecturer. It does not claim field performance, expert approval, or user-study results that have not been collected.

For a separate new-computer installation guide covering Docker Hub and dataset acquisition/preparation, see [SETUP.md](SETUP.md).
For a Gemini-ready prompt to generate presentation slides, speaker notes, and an explainer-video script, see [GEMINI_PRESENTATION_BRIEF.md](GEMINI_PRESENTATION_BRIEF.md).

## Share the source on GitHub

The repository should contain application/training source, documentation, tests, configuration examples, and (only if its redistribution terms permit) the active model artifacts. `.gitignore` excludes local environments, uploads/history, raw and organized datasets, temporary candidate models, and the Docker image archive. The exact Kaggle licenses have not yet been recorded, so do not commit or redistribute dataset images until their terms are verified. Never commit `.env`; use `.env.example` as a safe template. The runtime image itself is published separately at `bravenamuli/rice-diagnostics:1.1`.

## 1. Project purpose and scope

The application accepts a photo, preprocesses it, and uses a trained convolutional neural network (CNN) to return the most likely supported rice-leaf class. It reports a confidence value and labels low-confidence results as uncertain. A browser history stores prior diagnoses locally.

The currently supported classes are:

| Model label | Meaning |
|---|---|
| `healthy` | Healthy rice leaf |
| `blast` | Rice blast (including source folders named `leaf_blast`) |
| `brown_spot` | Brown spot |
| `bacterial_leaf_blight` | Bacterial leaf blight |
| `leaf_scald` | Leaf scald |
| `narrow_brown_spot` | Narrow brown spot |

`class_labels.py` is the shared source of truth for the model class order and supported source-folder aliases. Unsupported Paddy Doctor categories are ignored by the organizer; unsupported class folders in the training input are rejected rather than silently mapped to a disease.

**Not in the current model:** rice yellow mottle virus (RYMV), pests or other diseases, and a separately trained detector for “not a rice leaf.” Do not interpret an uncertain result as proof that an upload is not a rice leaf.

## 2. Repository map

| Path | Responsibility |
|---|---|
| `app/__init__.py` | Flask application factory, configuration, one-time model initialization, SQLite setup, and blueprint registration |
| `app/routes.py` | Web and JSON endpoints, upload validation, prediction request flow, history and result pages |
| `app/services/predictor.py` | Loads model metadata and Keras model once; resizes images and returns top predictions and confidence |
| `class_labels.py` | Canonical six-class list and aliases shared by organizing and training |
| `organize_dataset.py` | Previews or copies supported classes from the local Paddy Doctor source; writes provenance to the manifest |
| `scripts/import_kaggle_dataset.py` | Imports only the Kaggle `train` split, filters duplicates/overlap, and updates the manifest |
| `rice_cnn.py` | Dataset loading, duplicate-aware splits, CNN training, evaluation, reports, and model export |
| `run.py` | Loads `.env` values and starts the local Flask development server |
| `Dockerfile`, `compose.yaml`, `.dockerignore` | Build and run the inference website in a local Docker container while keeping training data out of the image |
| `dataset/` | Organized class folders used by training, plus `manifest.csv` |
| `model/` | Model currently loaded by the application and its class metadata and evaluation evidence |
| `model_sixclass_candidate/` | Six-class candidate training output retained separately from the active model |
| `model/previous_4class/`, `model/previous_3class/` | Backups of prior model versions |
| `templates/`, `static/` | HTML pages, styles, and browser-side image preview |
| `uploads/` | Locally saved JPEG copies of uploaded images |
| `history.db` | SQLite diagnosis history database |
| `tests/` | Tests for the web app, dataset organizing/import, and split safeguards |
| `rice_project_plan (1).xlsx` | Project progress tracker, decisions, photo log, and pending report tasks |

The `Rice Leaf Disease Images/` and `external_datasets/` folders are local source material, not Flask application code. The adjacent Kaggle data directory is expected at `..\RiceLeafsDisease` by the documented importer/training commands; change `--source` or the validation path if your copy is elsewhere.

## 3. Data sources and preparation

### 3.1 Local Paddy Doctor source

`organize_dataset.py` scans the configured source (by default, the local `paddy_doctor/` folder if present, otherwise `Rice Leaf Disease Images/`). It maps only supported source labels into the canonical classes. Other source categories are counted as ignored, not relabelled as one of the supported diseases.

Preview first; preview mode does not copy files:

```powershell
.\.venv\Scripts\python.exe organize_dataset.py
```

To copy the selected classes and update `dataset/manifest.csv`:

```powershell
.\.venv\Scripts\python.exe organize_dataset.py --copy
```

If healthy images are in a separate folder, include that folder explicitly:

```powershell
.\.venv\Scripts\python.exe organize_dataset.py --healthy-source "Rice Leaf Disease Images\healthy" --copy
```

For a different source path, add `--source "C:\path\to\source"`. Inspect the preview and output before running training. The manifest records the output-relative file, canonical class, source dataset, original source folder, and original image path. Keep this manifest: it is the project's data-provenance record.

### 3.2 Kaggle source

The second dataset has `train/` and `validation/` folders with six classes. Import its training images only:

```powershell
.\.venv\Scripts\python.exe -m scripts.import_kaggle_dataset --source "..\RiceLeafsDisease"
```

The importer maps `leaf_blast` to `blast`, keeps leaf scald and narrow brown spot as distinct labels, skips byte-identical images already in the organized data, excludes training images also present in Kaggle validation, and records imported files in the manifest. It does not copy Kaggle validation images into `dataset/`; that split remains available for separate evaluation.

The local Kaggle folder currently does not record the dataset page URL or license. Before distributing images or asserting licensing in the report, locate the exact source and verify its reuse terms. Keep attribution and source details in the report.

### 3.3 Training-input checks and data leakage controls

The training loader:

1. Requires the six supported class directories and uses the fixed order from `class_labels.py`.
2. Considers JPEG and PNG images, skips unreadable images, and de-duplicates byte-identical files.
3. Rejects identical image bytes assigned conflicting class labels.
4. Calculates a perceptual hash and groups same-class images with Hamming distance at most four so a group is assigned to one internal split, not split across train/validation/test.
5. Checks the external validation set for exact byte-level overlap with the organized training source and repeated validation files.

These controls lower some leakage risks; they do not prove that every near-duplicate, same-plant photo, or source-specific visual cue has been removed. The manifest does not currently provide verified plant IDs for the public datasets. An external validation set from the same Kaggle source is not equivalent to independent Mwea field data.

## 4. Training and model design

The current training command used to produce the six-class candidate is:

```powershell
.\.venv\Scripts\python.exe rice_cnn.py train --data_dir dataset --external_validation_dir "..\RiceLeafsDisease\validation" --out_dir model_sixclass_candidate --img_size 160 --batch_size 32 --epochs 5 --fine_tune_epochs 0 --weights imagenet
```

Do not direct a new training run into `model/` until you have reviewed its reports and decided to promote it. Training an existing output directory can replace artifacts there; use a new candidate directory for each experiment.

### 4.1 What training does

1. **Load and label:** read images from the six class folders and encode labels using the shared class order.
2. **Split:** create deterministic, stratified training, validation, and internal test subsets. The default validation and test fractions are each 15%; random seed is 42. Near-duplicate groups are kept together.
3. **Decode and resize:** decode images as RGB and resize to 160 × 160 for this run.
4. **Augment training images:** randomly flip, rotate, zoom, adjust contrast, and adjust brightness. Augmentation is used while fitting, not as a replacement for new real-world examples.
5. **Normalize:** scale pixel values from 0–255 to the range expected by MobileNetV2. The rescaling layer is part of the saved model, so the web predictor supplies resized 0–255 pixels.
6. **Build the CNN:** use an ImageNet-pretrained MobileNetV2 feature-extraction backbone, initially frozen, followed by global average pooling, dropout, and a six-output softmax classification layer. The softmax outputs are class scores that sum to one; they are not guaranteed to be calibrated probabilities.
7. **Train:** optimize the classification head with Adam and sparse categorical cross-entropy. Class weights compensate for unequal training counts. Early stopping and learning-rate reduction monitor validation loss. Optional fine-tuning can unfreeze some backbone layers, but the recorded run used `--fine_tune_epochs 0`.
8. **Evaluate:** use the held-back internal test set once for final internal metrics and optionally score the separate Kaggle validation set. The test split is not used to fit model weights.
9. **Save evidence and artifacts:** write the Keras model, class metadata, training curves, text classification report, and confusion matrix. A TensorFlow Lite export is also attempted; it is not used by the current Flask app.

### 4.2 Current artifacts

The active application reads `model/rice_disease_model.keras` and `model/class_names.json`. The metadata supplies class names and image size, preventing the web service from maintaining an independent class order. Current report files in `model/` include:

| Artifact | Use |
|---|---|
| `classification_report.txt` | Per-class precision, recall, F1, and support on the grouped internal test split |
| `confusion_matrix.png` | Internal test errors by actual and predicted class |
| `kaggle_validation_report.txt` | Per-class results on the separate Kaggle validation set |
| `kaggle_validation_confusion_matrix.png` | Confusion matrix on that Kaggle validation set |
| `training_curves.png` | Training and validation accuracy/loss by epoch |
| `rice_disease_model.keras` | Model used by Flask |
| `rice_disease_model.tflite` | Optional exported model; not currently served |
| `class_names.json` | Model class order, input size, seed, and external-validation flag |

Backups of previous three- and four-class models are retained under `model/previous_3class/` and `model/previous_4class/`. They are for provenance/rollback, not the models currently served.

## 5. Results currently available

The recorded six-class model was trained for five epochs. These figures are copied from the saved reports and describe these dataset splits only:

| Evaluation set | Images | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Grouped internal test | 799 | 85.73% | 0.8427 | 0.8886 | 0.8556 |
| Separate Kaggle validation | 518 | 72.39% | 0.7658 | 0.7222 | 0.7158 |

Kaggle validation recall is especially low for blast (51.76%), brown spot (52.87%), and bacterial leaf blight (50.59%). This means accuracy alone hides important class-specific errors. Consult the reports and confusion matrices when explaining performance.

**Interpretation:** these are preliminary results on Paddy Doctor/Kaggle-derived data. The Kaggle set is from the same dataset source/domain as the imported training portion. Neither result demonstrates performance on Mwea farms, other cameras, or images taken in the field. No field accuracy, expert-confirmation rate, KALRO approval, or user feedback is claimed.

## 6. Flask application and request flow

The app uses the `create_app()` factory in `app/__init__.py`. At startup it reads the model metadata and loads the Keras model once through `app/services/predictor.py`. Restart the app after replacing model files.

### 6.1 Browser flow

1. Open `/` and choose a photo.
2. The page previews the selected image in the browser.
3. The browser submits the image as multipart form data to `/diagnose`.
4. The server validates and normalizes the upload to a JPEG copy, then calls the predictor.
5. A confident result is shown on the result page; a low-confidence response is presented as uncertain instead of making a disease claim.
6. The diagnosis and uploaded image reference are available in that browser's local history. A user can delete a history item.

### 6.2 Routes

| Method and path | Purpose |
|---|---|
| `GET /` | Upload page and image preview |
| `GET /health` | Application/model status and loaded class names |
| `POST /diagnose` | Browser upload and result-page flow |
| `POST /api/v1/predict` | JSON prediction API; accepts multipart field `image` (or `photo`) |
| `GET /result/<id>` | Show a saved diagnosis belonging to the current browser session |
| `GET /history` | Show up to 200 recent diagnoses for that browser |
| `GET /photo/<filename>` | Serve an uploaded photo only when it belongs to that browser's record |
| `POST /delete/<id>` | Delete a diagnosis and its saved photo |

Example successful JSON API response:

```json
{
  "success": true,
  "prediction": "blast",
  "confidence": 0.9421,
  "status": "confident",
  "message": null
}
```

Low-confidence responses use `"prediction": "unsure"` and include a caution message. This message says the image may not be a rice plant/leaf or may be unclear; it is not an output from a trained non-rice detector. Missing files and unreadable/unsupported images return a client error; an unavailable model returns HTTP 503.

### 6.3 Upload handling and local storage

Uploads are limited to 8 MB. The server checks the decoded image format (JPEG/PNG; MPO is also accepted by Pillow), applies EXIF orientation, converts to RGB, resizes the saved copy to a maximum 1600 × 1600, and assigns a random server-generated filename. The predictor then resizes the image to the model's 160 × 160 input size.

Diagnosis records are stored in the local SQLite database `history.db`; images are stored in `uploads/`. The record includes an anonymous browser-session ID, timestamp, generated image filename, prediction/uncertain status, confidence, and top results. The application does not request a person's name or contact details. The `history.db` and `uploads/` contents are local files and should be treated as potentially sensitive; remove them deliberately when no longer needed. The current history is local to this installation, not a cloud backup or multi-device account.

### 6.4 Settings

`run.py` loads `.env` with `python-dotenv`. Configuration can be set with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | project `model/` | Directory containing the Keras model and metadata |
| `UPLOAD_DIR` | project `uploads/` | Saved uploaded image copies |
| `DB_PATH` | project `history.db` | SQLite history file |
| `CONFIDENCE_THRESHOLD` | `0.70` | Top-score cutoff for showing a confident result |
| `HOST` | `127.0.0.1` | Development server bind address |
| `PORT` | `5000` | Development server port |
| `FLASK_DEBUG` | `0` | Set to `1` only for local development debugging |

Do not expose Flask's development server publicly or enable debug mode in a deployed service. No production deployment is currently documented or claimed.

## 7. Set up and run locally on Windows

The project targets Python 3.12 and uses the project-local `.venv`. From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists and its dependencies are installed, skip environment creation and installation. Start the web application:

```powershell
.\.venv\Scripts\python.exe run.py
```

Open `http://127.0.0.1:5000/`. In another PowerShell window, check health:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

The health response should report `"status": "ok"` and the six class names. Stop the development server with Ctrl+C in its terminal.

To call the API from PowerShell:

```powershell
curl.exe -X POST -F "image=@C:\path\to\rice_leaf.jpg" http://127.0.0.1:5000/api/v1/predict
```

If the application says the model is unavailable, check that `MODEL_DIR` points to a folder containing both `rice_disease_model.keras` and `class_names.json`, then restart the server.

## 8. Run and transfer the app with Docker

Docker is the simplest way to run the website on another laptop without copying this laptop's `.venv`. The container includes the Flask application and active six-class model. It intentionally excludes training datasets, previous model backups, local uploads/history, the workbook, and the development virtual environment. You do not need the datasets to run predictions; you do need them if you plan to retrain.

### Requirements

- Docker Desktop installed and running on the laptop (Windows containers are not needed; use Docker Desktop's Linux-container mode).
- Enough free disk space for the Linux base image, TensorFlow dependencies, and the image. The resulting image can be large.
- For a local source build, the project folder should include `Dockerfile`, `compose.yaml`, `.dockerignore`, `requirements-docker.txt`, `app/`, `templates/`, `static/`, `class_labels.py`, and `model/`.

### Pull and start

The published release uses the Docker Hub image `bravenamuli/rice-diagnostics:1.1`. From the folder containing `compose.yaml`, run:

```powershell
docker compose pull
docker compose up -d
docker compose ps
```

If the Docker Hub repository is private, first run `docker login --username bravenamuli` with your own authorized account.

The container is published on host port **5001** to avoid clashing with the native Flask server, which commonly uses 5000. Wait for the first startup/model load, then open `http://127.0.0.1:5001/`. Check the application and loaded classes with:

```powershell
Invoke-RestMethod http://127.0.0.1:5001/health
```

It should report `"status": "ok"` and the six supported class names. Compose binds the website to this laptop's loopback address only. The container uses one Gunicorn worker and several request threads to avoid loading multiple copies of the TensorFlow model into memory.

Useful commands:

```powershell
docker compose logs -f
docker compose restart
docker compose down
```

`docker compose down` stops/removes the container but **keeps** the named `rice-diagnostics-data` volume containing diagnosis history and uploaded photos. To erase that stored data as well, use `docker compose down --volumes` only when you intentionally want to delete it. Do not use that command for routine shutdown.

### Prepare another laptop

1. Install and start Docker Desktop; ensure it is using Linux containers.
2. Copy `compose.yaml` to a folder on the new laptop.
3. In PowerShell in that folder, run `docker compose pull` and `docker compose up -d`.
4. Verify `http://127.0.0.1:5001/health`, then open the local website and try a sample image.

The new laptop creates its own empty Docker volume for history and uploads. The existing laptop's volume is not part of the project folder; diagnosis history is not transferred by copying the source. If history or uploads must be migrated, back them up explicitly from the volume rather than copying the excluded `history.db` from an old local run.

If the recipient has no internet access or Docker Hub pull is impractical, export the image to a drive:

```powershell
docker save --output rice-diagnostics-image.tar rice-diagnostics:local
```

Copy `rice-diagnostics-image.tar` and `compose.yaml` to the new laptop. Load the image, then start it (the imported tag matches the image named in Compose):

```powershell
docker load --input rice-diagnostics-image.tar
docker compose up -d
```

The archive is large; make sure the transfer drive has enough free space. Keep the source project/model files too if you want to retrain or rebuild the image later.

Compose accepts `SECRET_KEY` and `CONFIDENCE_THRESHOLD` from the shell or a local `.env` file. The bundled fallback secret is intended only for a local demonstration. For a stable local session key in PowerShell, set a secret before starting Compose:

```powershell
$env:SECRET_KEY = [guid]::NewGuid().ToString() + [guid]::NewGuid().ToString()
docker compose up -d
```

Use the same secret after restarts if you want existing browser sessions to remain valid. Do not publish the local-only fallback or expose this development project directly to the public internet; a public deployment needs an appropriately managed secret, HTTPS, and production security review.

### Publish a new model version to Docker Hub

The current Docker Hub image packages the active model for CPU inference using pinned TensorFlow CPU 2.21.0; Docker does not train the model or require a GPU. To release a model update, first train and evaluate locally, review the candidate reports, and only then replace the active `model/` files. Sign in, build a new version tag, and push it:

```powershell
docker login --username bravenamuli
docker build -t bravenamuli/rice-diagnostics:1.2 .
docker push bravenamuli/rice-diagnostics:1.2
```

Update the `image:` tag in `compose.yaml` to the new version for recipients. The Dockerfile uses `requirements-docker.txt`, a runtime-only dependency list that omits training/reporting tools. The build context excludes `.venv/`, `dataset/`, `Rice Leaf Disease Images/`, candidate and prior models, `.env`, uploads, and local database files. This keeps private/large training data and per-computer state out of the Docker image.

## 9. Tests and reproducibility

Run the existing test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover app creation, health, missing and unsupported uploads, prediction response structure, uncertain-result messaging, label aliases, dataset import/deduplication, and training split safeguards. Model-backed upload tests run inference when the configured model is available; otherwise they check the documented model-unavailable response. A passing unit test suite is not a substitute for field evaluation or a human usability study.

Reproducibility measures include a fixed random seed (42), deterministic file ordering and split logic, recorded model class order/input size, preserved source paths in the manifest, and saved evaluation artifacts. Full bit-for-bit repeatability is not guaranteed across different TensorFlow versions, hardware, or numerical backends. Record Python/package versions, training arguments, data source/version, and date for any new experiment.

## 10. Limitations and risks to explain honestly

1. **No rice/non-rice gate:** the six-output softmax always compares the image against the six trained classes. A non-rice image can receive a confident disease label. The 0.70 threshold catches some low scores but is not a non-rice detector.
2. **Dataset domain mismatch:** current evaluation is on public dataset images, not Mwea field photos. Background, camera, lighting, rice variety, growth stage, and symptom severity may differ.
3. **Uneven per-class performance:** external Kaggle recall is weak for three classes. A false negative can matter in a diagnostic setting; the app should be treated as a prototype/decision aid, not a replacement for an agricultural expert.
4. **Limited validation independence:** exact byte-level overlap is rejected, but visual similarity or related images may remain. The current internal perceptual grouping is within each class. There are no expert-verified plant IDs to group multiple photos from the same plant across the entire corpus.
5. **Source/license documentation incomplete:** the Kaggle dataset URL and license have not been recorded in the project. Verify attribution and permitted use before sharing data or distributing the application.
6. **Class scope is fixed to six:** RYMV and other categories are not currently trained. Adding a class requires labelled examples, retraining, evaluation, and checking that advice/UI/docs remain correct; do not just rename an existing class.
7. **Prediction confidence is not calibrated certainty:** softmax confidence should not be interpreted as the probability that a diagnosis is correct. The threshold is configurable and has not been validated as a safety boundary.
8. **No clinical/agronomic ground-truth claim:** advice is informational and may be missing for newer classes. Confirm disease diagnosis and treatment with a qualified local extension officer.
9. **Local prototype only:** SQLite and uploaded photos are stored on the machine. There is no authentication, remote hosting, backup, multi-user account system, or production deployment.
10. **No completed human evaluation:** permission, consent, participant testing, questionnaires/interviews, and usability findings have not been completed.
11. **Academic/report items remain:** supervisor agreement on final scope, Mwea data collection, Kaggle citation/license, figures/diagrams, outstanding report sections, results chapter, and conclusion remain to be addressed as applicable to university requirements. Check the tracker workbook and confirm its decisions with the supervisor.

## 11. Recommended next work

Prioritize work that improves evidence and addresses the biggest risks:

1. Confirm the website and six-class scope with the supervisor; reconcile older report statements that describe Android or binary classification.
2. Find the exact Kaggle dataset page, record its citation/license, and document every data source and mapping.
3. Obtain permission and consent before collecting Mwea photos. Have an extension officer or qualified expert verify labels; record plant IDs so photos from one plant stay in one split.
4. Build a separate rice-leaf-vs-not-rice gate only after collecting representative labelled positive and negative images. Evaluate false acceptance of non-rice images and false rejection of rice leaves on a held-out test set. Do not promise perfect rejection.
5. Improve disease-class recall using reviewed labels, representative field images, and careful experiments. Compare candidate models on fixed untouched test/validation sets before replacing `model/`.
6. Run documented user testing only after approvals; report method and actual findings, including failures.
7. Finish report diagrams and chapters using the actual architecture, actual saved metrics, and clearly stated limitations.

## 12. Lecturer presentation sequence

Use this order to explain the project from motivation to evidence. Keep the distinction between implemented work and future work explicit.

### Short opening (about 30 seconds)

> “This project is a local web prototype for supporting rice-leaf disease screening. A user uploads a leaf photograph; a TensorFlow/Keras CNN classifies it into one of six supported classes, reports its score, and saves a local diagnosis history. The current evaluation uses public datasets. It has not yet been validated on Mwea field images or with farmers.”

### Technical explanation (about 5–8 minutes)

1. **Problem and scope:** Explain why image-based screening is being explored, identify Mwea as the intended case-study context, and list the six implemented labels. State that RYMV is excluded because no labelled RYMV dataset is currently used.
2. **Data provenance:** Show the source folders and `dataset/manifest.csv`. Explain that the organizer maps only supported classes, ignores unrelated categories, and keeps each copied file traceable to its source.
3. **Leakage and partitions:** Explain the separate training, validation, and test roles. The model is fit on training data, validation supports training choices/early stopping, and the internal test split is held back for final scoring. Near-duplicate grouping and exact-hash checks reduce overlap risk. Kaggle validation is separately reported and was not copied into training.
4. **Image processing:** Describe RGB decoding, resize to 160 × 160, and scaling from 0–255 into MobileNetV2's expected range. Random augmentation is used for training only.
5. **CNN architecture:** Explain transfer learning: a frozen ImageNet-pretrained MobileNetV2 extracts visual features, and a pooling/dropout/dense softmax head selects one of six classes. The trained head learns the project-specific labels.
6. **Training and saved outputs:** Describe the five-epoch head-training run, class weighting, validation monitoring, reproducible seed, and the saved `.keras` model, metadata, curves, reports, and confusion matrices. Clarify that TensorFlow Lite export is optional and not the web inference path.
7. **Web request flow:** Show `templates/index.html` → Flask `/diagnose` or `/api/v1/predict` → `save_upload` validation → `Predictor.predict` → JSON/result page → SQLite history. Note the model is loaded once at app startup.
8. **Results:** Present both evaluation sets and their sizes. Report the 85.73% internal and 72.39% Kaggle accuracy, then immediately discuss the low Kaggle recall for blast, brown spot, and bacterial leaf blight. Use the confusion matrices to explain which classes are confused.
9. **Testing and demonstration:** Start the server, show `/health`, upload a clear supported sample, show an uncertain result if one occurs, and open history. Explain that automated tests cover request handling and pipeline safeguards; they do not prove farm accuracy.
10. **Limitations and next steps:** Emphasize no non-rice detector, no Mwea/expert/user evaluation, uncertain licensing citation, and local-only storage. Finish with the planned collection of expert-verified Mwea data and proper held-out evaluation.

### Suggested live demo order

1. In PowerShell at the project root, run `.\.venv\Scripts\python.exe run.py`.
2. Open `http://127.0.0.1:5000/` and show the six supported classes and upload guidance.
3. Open `http://127.0.0.1:5000/health` and show that the model loaded with six labels.
4. Upload a clear example image from an existing class folder under `dataset/`.
5. Explain the result, confidence, and uncertainty behavior without presenting a single image as an accuracy test.
6. Show `/history` and identify the local SQLite/image storage.
7. Point to `model/classification_report.txt`, `model/kaggle_validation_report.txt`, and both confusion matrices as the actual evaluation evidence.
8. Close by stating the non-rice and Mwea-field-validation limitations.

If live internet, TensorFlow startup, or the laptop causes a demo problem, use saved reports and screenshots as evidence, but label them as saved artifacts and do not pretend the live inference succeeded.

## 13. Suggested architecture explanation

```text
Local source folders
       |
       v
organize_dataset.py / import_kaggle_dataset.py
       |                         \
       v                          +--> separate Kaggle validation
dataset/<class>/ + manifest.csv                 |
       |                                         |
       v                                         v
rice_cnn.py: deduplicate -> grouped split -> train -> evaluate
       |                                         |
       +--> model/*.keras + metadata + reports <--+
                         |
                         v
Browser -> Flask routes -> upload validation -> Predictor -> CNN
   ^                                                   |
   |                                                   v
   +------ result page / JSON <--- confidence/status
                         |
                         v
               SQLite history + uploads/
```

## 14. Project completion status

| Area | Current evidence/status |
|---|---|
| Dataset pipeline | Supported source mapping, Kaggle train importer, manifest provenance, and duplicate/overlap safeguards are implemented |
| Six-class model | Trained model is in `model/`; metrics and plots are saved; external performance is uneven |
| Flask application | Upload, health check, prediction, result, history, and delete flows are implemented |
| Automated tests | Test suite exists; run the command in Section 8 to verify the current checkout |
| Mwea field dataset | Not collected/verified in the current evidence |
| Non-rice detector | Not implemented; current uncertainty threshold is not a substitute |
| Human/user evaluation | Not completed |
| Report | Use `rice_project_plan (1).xlsx` to track remaining diagrams, report sections, decisions, and evidence; verify requirements with the supervisor |
#   R i c e - d i a g n o s t i c s  
 