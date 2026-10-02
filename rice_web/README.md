# Rice Leaf Doctor (web app)

Upload a photo of a rice leaf, get a diagnosis with a confidence score, advice, and a history of past checks.

## Run it
1. Train a model first with `rice_cnn.py` (it creates a `model_out/` folder). Copy `model_out/` into this folder.
2. `pip install -r requirements.txt`
3. `python app.py` then open http://127.0.0.1:5000

To try it from a phone on the same Wi-Fi: `HOST=0.0.0.0 python app.py` (Windows PowerShell: `$env:HOST="0.0.0.0"; python app.py`), then open `http://<your-computer-ip>:5000` on the phone.

Optional settings (environment variables): `MODEL_DIR`, `CONFIDENCE_THRESHOLD` (default 0.70), `SECRET_KEY`, `HOST`, `PORT`.

## How it behaves
- Photos are checked as real JPEG/PNG images (max 8 MB), re-saved under a random name, then classified.
- Below the confidence threshold the result says "Not sure" and shows no treatment advice.
- History is stored per browser (anonymous cookie, SQLite file `history.db`). Each record can be deleted with its photo.

## Before real users see it
- `advice.json` is DRAFT guidance. Have KALRO or Mwea extension officers review and correct it.
- Tune `CONFIDENCE_THRESHOLD` using your validation results, not the 0.70 default.
- Flask's built-in server is for development. For deployment use a production server (for example gunicorn on Linux, waitress on Windows) behind HTTPS.
- A model can be confidently wrong, especially on photos that are not rice leaves or are unlike the training images.
- Tested with Python 3, Flask 3.1, TensorFlow 2.21 and Pillow 12.
