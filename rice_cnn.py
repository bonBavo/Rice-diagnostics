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
    python rice_cnn.py train   --data_dir dataset --external_validation_dir validation
    python rice_cnn.py predict --model_dir model_out --image leaf.jpg

Requirements:
    pip install tensorflow scikit-learn matplotlib pillow
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from class_labels import CLASS_NAMES, canonical_class_name

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
    """Return paths and labels in the project's fixed class order."""
    root = pathlib.Path(data_dir)
    directories = {}
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        class_name = canonical_class_name(folder.name)
        if class_name is None:
            raise ValueError(f"Unsupported class folder in {root}: {folder.name}")
        if class_name in directories:
            raise ValueError(
                f"Multiple folders map to {class_name!r} in {root}"
            )
        directories[class_name] = folder
    missing = [name for name in CLASS_NAMES if name not in directories]
    if missing:
        raise SystemExit(
            f"Missing required class folders in {root}: {', '.join(missing)}"
        )

    paths, labels, skipped = [], [], 0
    seen_hashes = {}
    for idx, name in enumerate(CLASS_NAMES):
        for p in sorted(directories[name].rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if is_readable(p):
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                previous_label = seen_hashes.get(digest)
                if previous_label is not None:
                    if previous_label != idx:
                        raise ValueError(
                            "Byte-identical images have conflicting class labels."
                        )
                    skipped += 1
                    continue
                seen_hashes[digest] = idx
                paths.append(str(p))
                labels.append(idx)
            else:
                skipped += 1
    if skipped:
        print(f"Skipped {skipped} unreadable or exact-duplicate image(s).")
    return np.array(paths), np.array(labels), list(CLASS_NAMES)


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


def _perceptual_hash(path):
    """Return a 64-bit DCT perceptual hash for duplicate-image grouping."""
    from PIL import Image, ImageOps
    from scipy.fft import dctn

    with Image.open(path) as image:
        pixels = np.asarray(
            ImageOps.exif_transpose(image).convert("L").resize((32, 32)),
            dtype=np.float32,
        )
    low_frequencies = dctn(pixels, norm="ortho")[:8, :8].reshape(-1)
    bits = low_frequencies > np.median(low_frequencies[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def split_data(paths, labels, val_frac, test_frac):
    """Make a deterministic stratified split without separating near-duplicates.

    Images with a perceptual-hash Hamming distance of at most four are treated
    as one group and always assigned to the same split.
    """
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")

    rng = np.random.RandomState(SEED)
    classes = np.unique(labels)
    train_idx, val_idx, test_idx = [], [], []
    hashes = [_perceptual_hash(path) for path in paths]

    exact_hashes = {}
    for path, label in zip(paths, labels):
        digest = hashlib.sha256(pathlib.Path(path).read_bytes()).digest()
        previous_label = exact_hashes.get(digest)
        if previous_label is not None and previous_label != label:
            raise ValueError(
                "Byte-identical images have conflicting class labels; "
                "review the dataset before training."
            )
        exact_hashes[digest] = label

    for c in classes:
        idx = np.where(labels == c)[0]
        groups = []
        for image_index in idx:
            image_hash = hashes[image_index]
            for group in groups:
                if any((image_hash ^ other_hash).bit_count() <= 4
                       for other_hash in group[1]):
                    group[0].append(image_index)
                    group[1].append(image_hash)
                    break
            else:
                groups.append(([image_index], [image_hash]))

        if len(groups) < 3:
            raise ValueError(
                f"Class {CLASS_NAMES[int(c)]} has fewer than 3 distinct image groups; "
                "cannot create train/validation/test splits."
            )
        order = rng.permutation(len(groups))
        n_test = max(1, int(round(len(groups) * test_frac)))
        n_val = max(1, int(round(len(groups) * val_frac)))
        if n_test + n_val >= len(groups):
            raise ValueError(
                f"Class {CLASS_NAMES[int(c)]} has too few distinct images for these split fractions."
            )
        test_groups = order[:n_test]
        val_groups = order[n_test:n_test + n_val]
        train_groups = order[n_test + n_val:]
        for group_index in test_groups:
            test_idx.extend(groups[group_index][0])
        for group_index in val_groups:
            val_idx.extend(groups[group_index][0])
        for group_index in train_groups:
            train_idx.extend(groups[group_index][0])

    split_sets = (set(train_idx), set(val_idx), set(test_idx))
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Image groups crossed dataset splits.")
    return (
        (paths[train_idx], labels[train_idx]),
        (paths[val_idx], labels[val_idx]),
        (paths[test_idx], labels[test_idx]),
    )


def validate_external_split(data_paths, data_labels, validation_paths, validation_labels):
    """Reject byte-identical overlaps between the project data and external validation."""
    if set(np.unique(validation_labels)) != set(range(len(CLASS_NAMES))):
        raise ValueError("External validation must contain images from every supported class.")

    training_hashes = {
        hashlib.sha256(pathlib.Path(path).read_bytes()).digest(): int(label)
        for path, label in zip(data_paths, data_labels)
    }
    validation_hashes = {}
    for path, label in zip(validation_paths, validation_labels):
        digest = hashlib.sha256(pathlib.Path(path).read_bytes()).digest()
        previous_label = validation_hashes.get(digest)
        if previous_label is not None:
            if previous_label != int(label):
                raise ValueError("External validation has a duplicate with conflicting labels.")
            raise ValueError("External validation contains a repeated image.")
        validation_hashes[digest] = int(label)
        if digest in training_hashes:
            raise ValueError(
                "External validation contains a byte-identical image also present "
                "in the organized dataset."
            )


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
def _confusion_matrix(y_true, y_pred, n):
    """Compute confusion matrix without sklearn."""
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _classification_report(y_true, y_pred, class_names):
    """Generate a text classification report without sklearn."""
    n = len(class_names)
    cm = _confusion_matrix(y_true, y_pred, n)
    lines = [f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"]
    lines.append("")
    precisions, recalls, f1s, supports = [], [], [], []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        lines.append(f"{name:>20} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")
        precisions.append(prec); recalls.append(rec); f1s.append(f1); supports.append(support)
    lines.append("")
    total = sum(supports)
    macro_p = np.mean(precisions); macro_r = np.mean(recalls); macro_f1 = np.mean(f1s)
    lines.append(f"{'macro avg':>20} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total:>10}")
    w = np.array(supports, dtype=float)
    if w.sum() > 0:
        wp = np.average(precisions, weights=w); wr = np.average(recalls, weights=w); wf = np.average(f1s, weights=w)
    else:
        wp = wr = wf = 0.0
    lines.append(f"{'weighted avg':>20} {wp:>10.4f} {wr:>10.4f} {wf:>10.4f} {total:>10}")
    return "\n".join(lines)


def evaluate(model, test_ds, class_names, out_dir, report_prefix="test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(class_names)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)

    accuracy = float((y_true == y_pred).mean())
    report = _classification_report(y_true, y_pred, class_names)
    print(f"\n{report_prefix} accuracy: {accuracy * 100:.2f}%  ({len(y_true)} images)\n")
    print(report)
    report_file = (
        "classification_report.txt"
        if report_prefix == "test"
        else f"{report_prefix}_report.txt"
    )
    (out_dir / report_file).write_text(
        f"{report_prefix} accuracy: {accuracy:.4f}\n\n{report}")

    cm = _confusion_matrix(y_true, y_pred, n)
    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.2 * n + 2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix ({report_prefix.replace('_', ' ')})")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    matrix_file = (
        "confusion_matrix.png"
        if report_prefix == "test"
        else f"{report_prefix}_confusion_matrix.png"
    )
    fig.savefig(out_dir / matrix_file, dpi=150)
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
def _compute_class_weight(labels, num_classes):
    """Compute balanced class weights without sklearn."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    n_samples = len(labels)
    weights = n_samples / (num_classes * np.maximum(counts, 1))
    return {i: float(weights[i]) for i in range(num_classes)}


def train(args):
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

    external_validation = None
    if args.external_validation_dir:
        ext_paths, ext_labels, ext_class_names = list_images(
            args.external_validation_dir
        )
        if ext_class_names != class_names:
            raise ValueError("External validation class order differs from training.")
        validate_external_split(paths, labels, ext_paths, ext_labels)
        external_validation = make_dataset(
            ext_paths, ext_labels, args.img_size, args.batch_size, training=False
        )
        print(f"Separate external validation: {len(ext_paths)} images")

    # Rare diseases get more weight so the model does not just favour big classes.
    class_weight = _compute_class_weight(y_tr, len(class_names))

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
    if external_validation is not None:
        evaluate(
            model,
            external_validation,
            class_names,
            out_dir,
            report_prefix="kaggle_validation",
        )

    model.save(out_dir / MODEL_FILE)
    (out_dir / META_FILE).write_text(json.dumps(
        {
            "class_names": class_names,
            "img_size": args.img_size,
            "seed": SEED,
            "external_validation": bool(args.external_validation_dir),
        },
        indent=2))
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
    t.add_argument(
        "--external_validation_dir",
        help="Optional separately held-out folder-per-class evaluation dataset.",
    )
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
