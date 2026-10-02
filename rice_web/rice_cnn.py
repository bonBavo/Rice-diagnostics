#!/usr/bin/env python3
"""
Multi-class rice leaf disease classifier (CNN with transfer learning).

Replaces the binary healthy/diseased sketch in section 3.5.6.

Expected dataset layout: one folder per class, folder name = class label.

    dataset/
        healthy/                  *.jpg
        <disease_1>/              *.jpg
        <disease_2>/              *.jpg
        ...

Use the disease names confirmed for Mwea (e.g. with KALRO / extension
officers) as folder names.

Usage:
    python rice_cnn.py train   --data_dir dataset --out_dir model_out
    python rice_cnn.py predict --model_dir model_out --image leaf.jpg

Requirements:
    pip install tensorflow scikit-learn matplotlib pillow
"""
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png"}
MODEL_FILE = "rice_disease_model.keras"
TFLITE_FILE = "rice_disease_model.tflite"
META_FILE = "class_names.json"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def is_readable(path):
    """True if Pillow can open the file (filters out corrupt uploads)."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def list_images(data_dir):
    """Return (paths, integer labels, class names) from a folder-per-class tree."""
    root = pathlib.Path(data_dir)
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    if len(class_names) < 2:
        raise SystemExit(f"Need at least 2 class folders in {root}, found {len(class_names)}.")

    paths, labels, skipped = [], [], 0
    for idx, name in enumerate(class_names):
        for p in sorted((root / name).iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if is_readable(p):
                paths.append(str(p))
                labels.append(idx)
            else:
                skipped += 1
    if skipped:
        print(f"Skipped {skipped} unreadable image(s).")
    return np.array(paths), np.array(labels), class_names


def load_image(path, img_size):
    """Decode a JPEG/PNG as RGB and resize. Returns float32 pixels in 0-255.

    The same function is used for training and prediction, so both see
    identical colour order (the original cv2 code read BGR).
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return tf.image.resize(img, (img_size, img_size))


def make_dataset(paths, labels, img_size, batch_size, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(lambda p, y: (load_image(p, img_size), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_data(paths, labels, val_frac, test_frac):
    """Stratified train/validation/test split.

    Caution: if the same leaf or plant was photographed several times, keep
    all of its photos in ONE split, otherwise the test score will be inflated.
    """
    from sklearn.model_selection import train_test_split

    p_train, p_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=val_frac + test_frac,
        stratify=labels, random_state=SEED)
    p_val, p_test, y_val, y_test = train_test_split(
        p_tmp, y_tmp, test_size=test_frac / (val_frac + test_frac),
        stratify=y_tmp, random_state=SEED)
    return (p_train, y_train), (p_val, y_val), (p_test, y_test)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(num_classes, img_size, weights):
    """MobileNetV2 backbone + softmax head.

    Input is raw 0-255 pixels; scaling to [-1, 1] happens inside the model,
    so deployment code never has to repeat the preprocessing.
    Augmentation layers are active only during training.
    """
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights)
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)   # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)                          # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="rice_disease_cnn"), base


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


def callbacks(patience):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=max(1, patience // 2)),
    ]


# --------------------------------------------------------------------------
# Evaluation and reporting
# --------------------------------------------------------------------------
def evaluate(model, test_ds, class_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    n = len(class_names)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)

    accuracy = float((y_true == y_pred).mean())
    report = classification_report(y_true, y_pred, labels=list(range(n)),
                                   target_names=class_names, digits=4, zero_division=0)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%  ({len(y_true)} unseen images)\n")
    print(report)
    (out_dir / "classification_report.txt").write_text(
        f"Test accuracy: {accuracy:.4f}\n\n{report}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.2 * n + 2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_history(hist, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist["accuracy"], label="train"); a1.plot(hist["val_accuracy"], label="validation")
    a1.set_title("Accuracy"); a1.set_xlabel("Epoch"); a1.legend()
    a2.plot(hist["loss"], label="train"); a2.plot(hist["val_loss"], label="validation")
    a2.set_title("Loss"); a2.set_xlabel("Epoch"); a2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def export_tflite(model, path):
    """Smaller model for offline use on a phone."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        path.write_bytes(converter.convert())
        print(f"Saved {path}")
    except Exception as exc:  # export is optional
        print(f"TFLite export skipped: {exc}")


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def train(args):
    from sklearn.utils.class_weight import compute_class_weight

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    paths, labels, class_names = list_images(args.data_dir)
    print(f"{len(paths)} images, {len(class_names)} classes:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {int((labels == i).sum())}")

    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = split_data(
        paths, labels, args.val_frac, args.test_frac)
    print(f"Split: {len(p_tr)} train / {len(p_va)} validation / {len(p_te)} test")

    train_ds = make_dataset(p_tr, y_tr, args.img_size, args.batch_size, training=True)
    val_ds = make_dataset(p_va, y_va, args.img_size, args.batch_size, training=False)
    test_ds = make_dataset(p_te, y_te, args.img_size, args.batch_size, training=False)

    # Rare diseases get more weight so the model does not just favour big classes.
    w = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y_tr)
    class_weight = {i: float(v) for i, v in enumerate(w)}

    weights = None if args.weights == "none" else "imagenet"
    model, base = build_model(len(class_names), args.img_size, weights)

    # Phase 1: train only the new classification head
    print("\nPhase 1: training classification head")
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                   class_weight=class_weight, callbacks=callbacks(patience=4))
    history = {k: list(v) for k, v in h1.history.items()}

    # Phase 2: fine-tune the top of the backbone at a low learning rate
    if args.fine_tune_epochs > 0:
        print("\nPhase 2: fine-tuning top backbone layers")
        base.trainable = True
        for layer in base.layers[:-args.unfreeze]:
            layer.trainable = False
        for layer in base.layers:                      # keep BatchNorm statistics fixed
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, 1e-5)                     # must recompile after changing trainable
        start = len(h1.epoch)
        h2 = model.fit(train_ds, validation_data=val_ds,
                       initial_epoch=start, epochs=start + args.fine_tune_epochs,
                       class_weight=class_weight, callbacks=callbacks(patience=3))
        for k, v in h2.history.items():
            history[k] += list(v)

    plot_history(history, out_dir)
    evaluate(model, test_ds, class_names, out_dir)

    model.save(out_dir / MODEL_FILE)
    (out_dir / META_FILE).write_text(json.dumps(
        {"class_names": class_names, "img_size": args.img_size}, indent=2))
    print(f"Saved {out_dir / MODEL_FILE}")
    export_tflite(model, out_dir / TFLITE_FILE)


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
def diagnose(image_path, model, meta, top_k=3, threshold=0.70):
    """Classify one leaf image. Importable from a Flask route.

    `threshold` is a starting value: choose it from validation results.
    Below it, the answer is flagged as uncertain instead of trusted.
    """
    img = load_image(str(image_path), meta["img_size"])
    probs = model.predict(tf.expand_dims(img, 0), verbose=0)[0]
    top = np.argsort(probs)[::-1][:top_k]
    results = [{"label": meta["class_names"][i], "confidence": float(probs[i])} for i in top]
    return {
        "prediction": results[0]["label"],
        "confidence": results[0]["confidence"],
        "status": "confident" if results[0]["confidence"] >= threshold else "uncertain",
        "top_k": results,
    }


def predict(args):
    model_dir = pathlib.Path(args.model_dir)
    model = tf.keras.models.load_model(model_dir / MODEL_FILE)
    meta = json.loads((model_dir / META_FILE).read_text())
    result = diagnose(args.image, model, meta, args.top_k, args.threshold)

    print(f"\nPrediction: {result['prediction']} ({result['confidence'] * 100:.1f}% confidence)")
    if result["status"] == "uncertain":
        print("Low confidence - retake the photo (good light, single leaf, in focus) "
              "or consult an agricultural officer.")
    print("\nTop matches:")
    for r in result["top_k"]:
        print(f"  {r['label']:<28} {r['confidence'] * 100:5.1f}%")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train and evaluate the model")
    t.add_argument("--data_dir", required=True)
    t.add_argument("--out_dir", default="model_out")
    t.add_argument("--img_size", type=int, default=224)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--epochs", type=int, default=15, help="head-training epochs")
    t.add_argument("--fine_tune_epochs", type=int, default=10)
    t.add_argument("--unfreeze", type=int, default=30, help="backbone layers to fine-tune")
    t.add_argument("--val_frac", type=float, default=0.15)
    t.add_argument("--test_frac", type=float, default=0.15)
    t.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    t.set_defaults(func=train)

    p = sub.add_parser("predict", help="diagnose one leaf image")
    p.add_argument("--model_dir", default="model_out")
    p.add_argument("--image", required=True)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.70)
    p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
