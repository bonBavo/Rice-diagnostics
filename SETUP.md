# Rice Disease Diagnostic System: Setup on Another Computer

This guide explains how to run the published application with Docker and, separately, how to obtain and prepare the datasets if you need to retrain the CNN. Docker Hub contains the **application and the active trained model only**. It does not include the datasets, training reports, or this project's source code.

## 1. What you need

For prediction-only use:

- A Windows, macOS, or Linux computer with Docker Desktop/Engine installed and running.
- An internet connection for the first Docker Hub pull.
- The `compose.yaml` file from this project.

For training or evaluation as well:

- The project source files, including `rice_cnn.py`, `organize_dataset.py`, `class_labels.py`, and `scripts/import_kaggle_dataset.py`.
- The supported datasets in the folder structure described in Section 5.
- Python 3.12 and the project's Python dependencies (or a configured development environment).
- Sufficient disk space for the image datasets and training outputs. TensorFlow training can take considerable time and memory; GPU use is not required by the application container.

## 2. Run the published Docker image

Install Docker Desktop on Windows or macOS, start it, and make sure its Linux container engine is running. On Linux, install and start Docker Engine and the Docker Compose plugin. In a terminal, confirm Docker works:

```powershell
docker --version
docker compose version
```

Create a new folder for the deployment and save the project's `compose.yaml` into it. The file should refer to the published image:

```yaml
services:
  rice-diagnostics:
    image: bravenamuli/rice-diagnostics:1.1
    ports:
      - "127.0.0.1:5001:5000"
    environment:
      SECRET_KEY: "${SECRET_KEY:-local-only-change-this-secret}"
      CONFIDENCE_THRESHOLD: "${CONFIDENCE_THRESHOLD:-0.70}"
    volumes:
      - rice-diagnostics-data:/app/data
    init: true
    restart: unless-stopped

volumes:
  rice-diagnostics-data:
    name: docker_rice-diagnostics-data
```

In a terminal opened in that folder, pull the image and start the application:

```powershell
docker compose pull
docker compose up -d
docker compose ps
```

The first pull downloads the image (about 2.22 GB in the tested local build; actual transfer size can vary). The container runs the CPU version of TensorFlow and may take a little time on first startup while loading the model.

Open the website at **http://127.0.0.1:5001/**. Check the model health and supported labels:

```powershell
Invoke-RestMethod http://127.0.0.1:5001/health
```

Expected response: status `ok` and six classes: `healthy`, `blast`, `brown_spot`, `bacterial_leaf_blight`, `leaf_scald`, and `narrow_brown_spot`.

The Docker Hub repository must be public for an anonymous pull. If it is private, sign in with an account that has access:

```powershell
docker login --username bravenamuli
```

Do not send account passwords or access tokens to another person. Use an authorized account and enter credentials only into Docker's login prompt.

### Stop, restart, and view logs

```powershell
docker compose logs -f
docker compose restart
docker compose down
```

The named Docker volume `rice-diagnostics-data` preserves diagnosis history and uploaded photos when the container is stopped or upgraded. Each computer has its own volume; it is not downloaded from Docker Hub or shared with other users. To intentionally erase that local history and uploads, use:

```powershell
docker compose down --volumes
```

This permanently removes the volume's data. Do not use it as a routine stop command.

### Secure the local session key

The fallback `SECRET_KEY` in the example is for a quick local demonstration, not a shared or internet-facing deployment. For a persistent private key, create a `.env` file beside `compose.yaml` containing a long random value:

```text
SECRET_KEY=replace-with-a-long-random-secret
CONFIDENCE_THRESHOLD=0.70
```

Keep this `.env` private and do not include it in a public source repository or share it with the image. Reuse the same key after restarts if you want existing browser sessions to remain valid. The Compose port is bound to loopback (`127.0.0.1`), so this setup is intended for access on that computer, not as a public web deployment.

## 3. Which image classes are supported

The published model recognizes these six trained labels:

| Canonical folder/model label | Accepted source-folder spelling |
|---|---|
| `healthy` | `healthy`, `normal` |
| `blast` | `blast`, `leaf_blast`, `leafblast`, `riceblast` |
| `brown_spot` | `brown_spot`, `brownspot` |
| `bacterial_leaf_blight` | `bacterial_leaf_blight`, `bacterialblight`, `blb` |
| `leaf_scald` | `leaf_scald`, `leafscald` |
| `narrow_brown_spot` | `narrow_brown_spot`, `narrowbrownspot` |

The project does not currently include RYMV or a separately trained rice-vs-not-rice detector. A low-confidence `"unsure"` answer is not proof that an upload is not a rice leaf.

## 4. Where to obtain the datasets

The original dataset files are kept locally by the project owner; they are **not** bundled in the Docker Hub image. The project manifest identifies the sources as:

1. `paddy_doctor` — the local source folder is named `Rice Leaf Disease Images/` in the current project copy. It contains source class folders including `Bacterialblight`, `Blast`, `Brownspot`, `healthy`, and `Tungro`. The manifest does not contain the original download URL or a dataset license.
2. `kaggle_riceleafsdisease` — a separately obtained Kaggle rice leaf dataset with `train/` and `validation/` folders and six class folders. The manifest records the source's local folder name, but the exact Kaggle page URL and license were not recorded in the project.

Because the exact download pages and reuse terms are not recorded, this guide does **not** guess or claim a specific Kaggle page/license. To reproduce the data exactly, ask the person who downloaded the existing folders for the original Kaggle page URLs, dataset versions, download dates, and license terms, or use their Kaggle download history. Record those details in your project report before redistributing dataset images.

If obtaining a fresh copy, start with Kaggle's dataset search pages for [Paddy Doctor](https://www.kaggle.com/datasets?search=Paddy%20Doctor) and [RiceLeafsDisease](https://www.kaggle.com/datasets?search=RiceLeafsDisease). Search results can change; compare the exact dataset title, owner, class-folder names, split structure, and license with Section 5 and the existing manifest before downloading or using a copy. Kaggle may require an account and acceptance of a dataset's terms. Do not assume similarly named datasets have identical images, labels, licenses, or splits. Preserve the original downloaded archive/folder and its source-page/license information.

No RYMV dataset is currently part of this pipeline. Do not create or relabel samples as RYMV; only add this class after obtaining real, labelled, appropriately licensed RYMV data and making a new model version.

## 5. Dataset folders for retraining

### 5.1 Get the full project source

Obtain the project source folder from its owner (for example, via the project's approved Git repository or a private file transfer). Docker Hub distributes the executable inference image, not scripts, reports, datasets, or the workbook. Confirm the source includes:

```text
rice_diagnostics/
├── app/
├── class_labels.py
├── organize_dataset.py
├── rice_cnn.py
├── scripts/
│   └── import_kaggle_dataset.py
├── model/
├── requirements.txt
└── rice_project_plan (1).xlsx
```

If there is no source repository available, request a source-code archive from the project owner. Do not try to recreate the training project from the Docker image; the image contains only runtime files and the trained model.

### 5.2 Arrange the local Paddy Doctor source

Place the extracted Paddy Doctor source under the project root as `Rice Leaf Disease Images/`, or pass its actual path to `organize_dataset.py --source`. The organizer maps supported class folders and ignores unrelated categories; do not rename unrelated folders to make them look like a supported class.

If the healthy samples are separate, pass their folder explicitly:

```powershell
.\.venv\Scripts\python.exe organize_dataset.py --source "Rice Leaf Disease Images" --healthy-source "Rice Leaf Disease Images\healthy"
```

This previews the selection without copying. Review the printed mapping first. Copy only after confirming it is correct:

```powershell
.\.venv\Scripts\python.exe organize_dataset.py --source "Rice Leaf Disease Images" --healthy-source "Rice Leaf Disease Images\healthy" --copy
```

The organizer writes selected images under `dataset/` and preserves source traceability in `dataset/manifest.csv`. Keep the manifest and check the class mapping before training. Do not delete or replace existing dataset files unless you have a verified backup.

### 5.3 Arrange and import the Kaggle training split

Extract the second Kaggle dataset beside the project, for example:

```text
PycharmProjects/
├── rice_diagnostics/
└── RiceLeafsDisease/
    ├── train/
    │   ├── bacterial_leaf_blight/
    │   ├── brown_spot/
    │   ├── healthy/
    │   ├── leaf_blast/
    │   ├── leaf_scald/
    │   └── narrow_brown_spot/
    └── validation/
        └── [same six class folders]
```

From the `rice_diagnostics` project root, previewing is not supported by the importer; it performs validated imports and updates the manifest:

```powershell
.\.venv\Scripts\python.exe -m scripts.import_kaggle_dataset --source "..\RiceLeafsDisease"
```

If the dataset is elsewhere, change `--source` to its actual directory. This importer uses the Kaggle `train/` split only. It leaves `validation/` separate and excludes byte-identical training files that overlap with validation. Do not manually copy validation images into `dataset/`.

### 5.4 Expected organized training folders

The training directory should contain one folder per canonical class:

```text
dataset/
├── manifest.csv
├── healthy/
├── blast/
├── brown_spot/
├── bacterial_leaf_blight/
├── leaf_scald/
└── narrow_brown_spot/
```

`dataset/manifest.csv` records each organized image's relative output file, class, source dataset, source folder, and original path. Those original paths are absolute paths from the computer where the images were organized and will not automatically resolve on a different laptop; the copied files and relative manifest file entries are what matter for training. Ensure the source and usage license allow the intended academic/research use.

## 6. Set up Python if training is required

Docker is enough to run predictions; Python is needed to organize data, train, or run the training-pipeline tests. Install Python 3.12, open PowerShell in the project root, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 7. Train and evaluate a candidate model

From the project root, with both organized training data and a separate Kaggle validation folder present:

```powershell
.\.venv\Scripts\python.exe rice_cnn.py train --data_dir dataset --external_validation_dir "..\RiceLeafsDisease\validation" --out_dir model_candidate_new --img_size 160 --batch_size 32 --epochs 5 --fine_tune_epochs 0 --weights imagenet
```

Use a **new output directory** for each experiment. The command creates a deterministic grouped internal train/validation/test split, trains from the training partition, monitors validation loss, evaluates on the held-back internal test set, evaluates the separate Kaggle validation set, and writes reports, confusion matrices, training curves, metadata, and the model candidate. Review all per-class metrics; do not choose a model based only on accuracy and do not claim Mwea field performance from public-dataset scores.

The external set needs all six classes, and the pipeline rejects exact image overlaps with the organized data. These safeguards do not guarantee complete near-duplicate or plant-level separation. Use expert-verified Mwea data and plant-aware grouping for a genuine local-field evaluation.

Do not overwrite `model/` or publish a new image until the candidate has been reviewed and deliberately approved. The current published image is version `1.1`; rebuilding/publishing a UI update does not change model weights or evaluation results.

## 8. Troubleshooting

| Symptom | What to check |
|---|---|
| `docker` command is not found | Install Docker Desktop/Engine, reopen the terminal, and check `docker --version`. |
| Docker cannot connect to the engine | Start Docker Desktop and wait until its Linux engine is ready. |
| `docker compose pull` says `unauthorized` | The Hub repository may be private; sign in with an account authorized to read it. |
| Port 5001 is already in use | Stop the other service using that port or change the left-hand host port in `compose.yaml`, for example `"127.0.0.1:5002:5000"`, then use port 5002 in the browser. |
| Health says `model_not_loaded` | Check container logs with `docker compose logs`; the published image should contain both the model and `class_names.json`. |
| Dataset organizer says source missing | Pass the actual source directory with `--source`. |
| Importer says a class folder is missing | Check both Kaggle splits include all six class folders, with supported names as in `class_labels.py`. |
| Importer reports duplicates/overlap | Review the source data and manifest; do not place the validation split in training. |
| Training says external validation overlaps | Obtain a clean validation split; do not suppress the overlap check to force a result. |
| Kaggle download URL/license is needed | Obtain it from the original downloader or Kaggle download history; the exact URL/license is not stored in the current project. |

## 9. Known limitations

- The current six-class classifier is not a dedicated detector for non-rice plants or objects. An uncertain prediction is only based on a score threshold.
- The published model was evaluated on grouped public-dataset and separate Kaggle validation images, not Mwea field images or a completed farmer/user trial.
- Kaggle validation recall was uneven for blast, brown spot, and bacterial leaf blight. Consult the saved reports and confusion matrices; do not present the model as a verified diagnostic replacement for an agricultural expert.
- RYMV and unsupported diseases/pests are not included.
- A Docker Hub image contains no source datasets, history from another machine, or training environment. Each installation stores history/photos in its own local Docker volume.
- The exact original Kaggle dataset URLs and license statements have not yet been documented. Verify before redistribution.
