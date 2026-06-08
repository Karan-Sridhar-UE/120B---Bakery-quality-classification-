import os, shutil, glob, random, hashlib, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_fscore_support)

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ----------------------- CONFIGURATION -----------------------
# Point at the whole /kaggle/input tree; os.walk finds the labelled good/bad
# folders, so the exact nesting of your uploaded dataset does not matter.
INPUT_ROOTS = ["/kaggle/input"]

COMBINED_DIR = "/kaggle/working/dataset_combined"
SPLIT_DIR    = "/kaggle/working/dataset_split"
FIG_DIR      = "/kaggle/working/figures"
MODEL_DIR    = "/kaggle/working/models"
for d in (FIG_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

IMG_SIZE   = (160, 160)
BATCH_SIZE = 32
EPOCHS_CNN       = 25
EPOCHS_TL_HEAD   = 12
EPOCHS_TL_FT     = 6
SPLIT_RATIOS = (0.70, 0.15, 0.15)
AUTOTUNE = tf.data.AUTOTUNE

# Data-quality controls
DEDUP = True                  # remove exact-duplicate images (prevents train/test leakage)
MIN_UNIQUE_PER_CLASS = 60     # stop early if a dataset has too few real images per class
MAX_PER_CLASS = 1500          # cap images per class to keep training time reasonable (None = use all)

# Optional manual mapping: folder-name substring (lowercase) -> 'good' or 'defective'.
# Checked BEFORE the automatic keyword rules. Use this when a dataset's "good"
# folder isn't an obvious word (e.g. a bread dataset with folders 'bread' and
# 'moldy' -> set {"moldy": "defective", "bread": "good"}). Leave {} to rely on
# automatic detection (good/fresh/healthy vs bad/moldy/rotten/stale/burnt).
CLASS_OVERRIDES = {}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
for root in INPUT_ROOTS:
    for dp, dirs, files in os.walk(root):
        n = len([f for f in files if f.lower().endswith(IMG_EXTS)])
        depth = dp.replace(root, "").count("/")
        if n or depth <= 4:
            print("  " * depth, os.path.basename(dp) or dp, f"({n} images)")

def classify_folder(name):
    n = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    for sub, label in CLASS_OVERRIDES.items():          # manual overrides first
        if sub.lower().replace(" ", "").replace("-", "").replace("_", "") in n:
            return label
    if any(k in n for k in ["good", "fresh", "nonmold", "notmold", "normal", "healthy"]):
        return "good"
    if any(k in n for k in ["bad", "mold", "stale", "rotten", "burnt", "defect", "spoil"]):
        return "defective"
    return None

def build_combined(input_roots, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    counts = {"good": 0, "defective": 0}
    for c in counts:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)
    skipped = []
    for root in input_roots:
        if not os.path.exists(root):
            continue
        for dp, _, files in os.walk(root):
            imgs = [f for f in files if f.lower().endswith(IMG_EXTS)]
            if not imgs:
                continue
            label = None
            for p in reversed(dp.replace("\\", "/").split("/")):
                label = classify_folder(p)
                if label:
                    break
            if label is None:
                skipped.append((dp, len(imgs))); continue
            for f in imgs:
                ext = os.path.splitext(f)[1].lower()
                shutil.copy(os.path.join(dp, f),
                            os.path.join(out_dir, label, f"{label}_{counts[label]:06d}{ext}"))
                counts[label] += 1
    return counts, skipped

counts, skipped = build_combined(INPUT_ROOTS, COMBINED_DIR)
print("After merging:", counts, "| total =", sum(counts.values()))
if skipped:
    print("Skipped (no good/defective keyword):")
    for d, n in skipped:
        print(f"  {n:>5}  {d}")

def audit_and_clean(root, dedup=True):
    removed_corrupt, removed_dup = 0, 0
    unique_per_class, seen = {}, set()
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        uniq = 0
        for f in list(os.listdir(cls_dir)):
            path = os.path.join(cls_dir, f)
            try:                                   # 1) corrupted / unreadable
                with Image.open(path) as im:
                    im.verify()
            except Exception:
                os.remove(path); removed_corrupt += 1; continue
            with open(path, "rb") as fh:           # 2) exact duplicate (md5)
                h = hashlib.md5(fh.read()).hexdigest()
            if h in seen:
                if dedup:
                    os.remove(path); removed_dup += 1
            else:
                seen.add(h); uniq += 1
        unique_per_class[cls] = uniq
    return removed_corrupt, removed_dup, unique_per_class

rc, rd, unique_per_class = audit_and_clean(COMBINED_DIR, dedup=DEDUP)
print(f"Removed corrupted/unreadable : {rc}")
print(f"Removed exact duplicates     : {rd}  (DEDUP={DEDUP})")
print("Unique images per class      :", unique_per_class)

counts = {c: len(os.listdir(os.path.join(COMBINED_DIR, c)))
          for c in sorted(os.listdir(COMBINED_DIR))}
print("Images available per class   :", counts, "| total =", sum(counts.values()))

# ---- guard: refuse to continue on a degenerate dataset ----
fewest = min(unique_per_class.values()) if unique_per_class else 0
if fewest < MIN_UNIQUE_PER_CLASS:
    raise ValueError(
        f"\n\nSTOP: the smallest class has only {fewest} UNIQUE images "
        f"(threshold = {MIN_UNIQUE_PER_CLASS}).\n"
        "This dataset is mostly duplicates and is NOT suitable for a credible "
        "CNN project — training/test would share identical images and accuracy "
        "would be meaningless.\n"
        "Pick a different dataset, attach it via 'Add Input', and re-run.\n")
print("\nDataset passed the uniqueness check — safe to continue.")

# Optional: cap images per class to keep training time reasonable
if MAX_PER_CLASS:
    for c in list(counts.keys()):
        cls_dir = os.path.join(COMBINED_DIR, c)
        files = os.listdir(cls_dir)
        if len(files) > MAX_PER_CLASS:
            random.shuffle(files)
            for f in files[MAX_PER_CLASS:]:
                os.remove(os.path.join(cls_dir, f))
    counts = {c: len(os.listdir(os.path.join(COMBINED_DIR, c)))
              for c in sorted(os.listdir(COMBINED_DIR))}
    print("After capping per class      :", counts, "| total =", sum(counts.values()))

summary = pd.DataFrame({"Class": list(counts.keys()),
                        "Number of images": list(counts.values())})
total = summary["Number of images"].sum()
summary["Percentage"] = (summary["Number of images"] / total * 100).round(1)
print(summary.to_string(index=False))
print(f"\nTotal images : {total}")
print(f"Classes      : {len(counts)} -> {list(counts.keys())}")

plt.figure(figsize=(6, 4))
sns.barplot(data=summary, x="Class", y="Number of images",
            palette=["#d9534f", "#5cb85c"])
plt.title("Class Distribution – Bakery Product Quality")
for i, v in enumerate(summary["Number of images"]):
    plt.text(i, v + total * 0.005, str(v), ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/01_class_distribution.png", dpi=150); plt.show()

def split_combined(combined_dir, out_dir, ratios=SPLIT_RATIOS, seed=SEED):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    classes = sorted(os.listdir(combined_dir))
    for split in ("train", "val", "test"):
        for c in classes:
            os.makedirs(os.path.join(out_dir, split, c), exist_ok=True)
    for c in classes:
        files = os.listdir(os.path.join(combined_dir, c))
        train_f, temp_f = train_test_split(files, train_size=ratios[0], random_state=seed)
        val_f, test_f = train_test_split(
            temp_f, train_size=ratios[1] / (ratios[1] + ratios[2]), random_state=seed)
        for split, fs in (("train", train_f), ("val", val_f), ("test", test_f)):
            for f in fs:
                shutil.copy(os.path.join(combined_dir, c, f),
                            os.path.join(out_dir, split, c, f))

split_combined(COMBINED_DIR, SPLIT_DIR)
for split in ("train", "val", "test"):
    line = [f"{split:<6}"]
    for c in sorted(counts):
        line.append(f"{c}={len(os.listdir(os.path.join(SPLIT_DIR, split, c)))}")
    print("  ".join(line))

train_ds = keras.utils.image_dataset_from_directory(
    f"{SPLIT_DIR}/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=True, seed=SEED)
val_ds = keras.utils.image_dataset_from_directory(
    f"{SPLIT_DIR}/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False)
test_ds = keras.utils.image_dataset_from_directory(
    f"{SPLIT_DIR}/test", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False)

class_names = train_ds.class_names      # ['defective', 'good']
print("Class order:", class_names)

# Augmentation pipeline (applied to TRAIN only, in tf.data)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.10),
], name="data_augmentation")

def _cast(x, y):
    return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

def _augment(x, y):
    return data_augmentation(x, training=True), y

train_ds = train_ds.map(_cast, num_parallel_calls=AUTOTUNE).cache()
train_ds = train_ds.map(_augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds   = val_ds.map(_cast, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_ds  = test_ds.map(_cast, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# Sample images (from the un-augmented validation set)
plt.figure(figsize=(10, 6))
for images, lbls in val_ds.take(1):
    for i in range(min(9, images.shape[0])):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(lbls[i].numpy()[0])]); plt.axis("off")
plt.suptitle("Sample Images"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_sample_images.png", dpi=150); plt.show()

plt.figure(figsize=(10, 6))
for images, _ in val_ds.take(1):
    first = images[0:1]
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(data_augmentation(first, training=True)[0].numpy().astype("uint8"))
        plt.axis("off")
plt.suptitle("Data Augmentation Examples"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_augmentation.png", dpi=150); plt.show()

def build_cnn(input_shape=(*IMG_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)        # normalize to [0,1]
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", name="last_conv")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="Custom_CNN")

def make_callbacks():
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=3, min_lr=1e-6)]

cnn = build_cnn()
cnn.compile(optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
cnn.summary()

history_cnn = cnn.fit(train_ds, validation_data=val_ds,
                      epochs=EPOCHS_CNN, callbacks=make_callbacks())

def plot_curves(hist, title, fname):
    h = hist if isinstance(hist, dict) else hist.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["accuracy"], label="train"); ax[0].plot(h["val_accuracy"], label="val")
    ax[0].set_title(f"{title} – Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["loss"], label="train"); ax[1].plot(h["val_loss"], label="val")
    ax[1].set_title(f"{title} – Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/{fname}", dpi=150); plt.show()

plot_curves(history_cnn, "Custom CNN", "04_cnn_curves.png")

def build_transfer(app_fn, preprocess_fn, name):
    base = app_fn(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = preprocess_fn(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=name), base

def _compile(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])

def train_transfer(model, base):
    _compile(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_TL_HEAD, callbacks=make_callbacks())
    base.trainable = True                       # fine-tune top layers
    for layer in base.layers[:-30]:
        layer.trainable = False
    _compile(model, 1e-5)
    n1 = len(h1.history["loss"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=n1 + EPOCHS_TL_FT,
                   initial_epoch=n1, callbacks=make_callbacks())
    return {k: h1.history[k] + h2.history.get(k, []) for k in h1.history}

# MobileNetV2
mnet, mnet_base = build_transfer(
    keras.applications.MobileNetV2,
    keras.applications.mobilenet_v2.preprocess_input, "MobileNetV2")
hist_mnet = train_transfer(mnet, mnet_base)
plot_curves(hist_mnet, "MobileNetV2", "05_mobilenet_curves.png")

# ResNet50
resnet, resnet_base = build_transfer(
    keras.applications.ResNet50,
    keras.applications.resnet50.preprocess_input, "ResNet50")
hist_resnet = train_transfer(resnet, resnet_base)
plot_curves(hist_resnet, "ResNet50", "06_resnet_curves.png")

def get_true_probs(model, dataset):
    y_true = np.concatenate([y.numpy().ravel() for _, y in dataset]).astype(int)
    probs = model.predict(dataset, verbose=0).ravel()
    return y_true, probs

def evaluate_model(model, dataset, name, idx):
    loss, acc = model.evaluate(dataset, verbose=0)
    y_true, probs = get_true_probs(model, dataset)
    y_pred = (probs > 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    # inference time per image (timed over the whole test set)
    _ = model.predict(dataset.take(1), verbose=0)         # warm-up
    t0 = time.time(); _ = model.predict(dataset, verbose=0)
    infer_ms = (time.time() - t0) / len(y_true) * 1000

    print(f"\n===== {name} =====")
    print(f"Accuracy {acc:.4f} | Precision {prec:.4f} | Recall {rec:.4f} | "
          f"F1 {f1:.4f} | Inference {infer_ms:.2f} ms/img")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix – {name}")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/{idx}_cm_{name}.png", dpi=150); plt.show()

    fpr, tpr, _ = roc_curve(y_true, probs); roc_auc = auc(fpr, tpr)
    return {"name": name, "accuracy": acc, "loss": loss, "precision": prec,
            "recall": rec, "f1": f1, "auc": roc_auc, "infer_ms": infer_ms,
            "y_true": y_true, "y_pred": y_pred, "probs": probs,
            "fpr": fpr, "tpr": tpr}

cnn_res    = evaluate_model(cnn,    test_ds, "Custom CNN",  "07")
mnet_res   = evaluate_model(mnet,   test_ds, "MobileNetV2", "08")
resnet_res = evaluate_model(resnet, test_ds, "ResNet50",    "09")
all_res = [cnn_res, mnet_res, resnet_res]

comparison = pd.DataFrame([{
    "Model": r["name"], "Accuracy": r["accuracy"], "Precision": r["precision"],
    "Recall": r["recall"], "F1": r["f1"], "ROC AUC": r["auc"],
    "Inference (ms/img)": r["infer_ms"]} for r in all_res]).round(4)
print(comparison.to_string(index=False))
comparison.to_csv(f"{FIG_DIR}/model_comparison.csv", index=False)

# Accuracy / F1 / AUC grouped bars
metrics = ["Accuracy", "F1", "ROC AUC"]
x = np.arange(len(comparison)); w = 0.25
plt.figure(figsize=(8, 4.5))
for i, m in enumerate(metrics):
    plt.bar(x + (i - 1) * w, comparison[m], w, label=m)
plt.xticks(x, comparison["Model"]); plt.ylim(0, 1.08); plt.ylabel("Score")
plt.title("Model Performance Comparison"); plt.legend()
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/10_model_comparison.png", dpi=150); plt.show()

# Inference-time comparison (efficiency)
plt.figure(figsize=(6, 4))
sns.barplot(data=comparison, x="Model", y="Inference (ms/img)", palette="viridis")
plt.title("Inference Time per Image (lower = faster)")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/11_inference_time.png", dpi=150); plt.show()

# Combined ROC curves
plt.figure(figsize=(5.5, 5))
for r in all_res:
    plt.plot(r["fpr"], r["tpr"], label=f"{r['name']} (AUC={r['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves – All Models"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/12_roc_all.png", dpi=150); plt.show()

def make_gradcam(img_array, model, last_conv="last_conv"):
    grad_model = keras.models.Model(model.inputs,
                    [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()

def overlay(img, heat):
    heat = np.uint8(255 * heat)
    jet = cm.get_cmap("jet")(np.arange(256))[:, :3][heat]
    jet = keras.utils.array_to_img(jet).resize((IMG_SIZE[1], IMG_SIZE[0]))
    jet = keras.utils.img_to_array(jet)
    return np.clip(jet * 0.4 + img, 0, 255).astype("uint8")

# rebuild ordered test images + use the CNN predictions
test_images = np.concatenate([x.numpy() for x, _ in test_ds], axis=0)
yt, yp = cnn_res["y_true"], cnn_res["y_pred"]
# positive class = 'good' (index 1)
cases = {
    "Correct":        np.where(yt == yp)[0],
    "False Positive": np.where((yp == 1) & (yt == 0))[0],  # predicted good, was defective
    "False Negative": np.where((yp == 0) & (yt == 1))[0],  # predicted defective, was good
}
plt.figure(figsize=(11, 4))
for i, (label, idxs) in enumerate(cases.items()):
    plt.subplot(1, 3, i + 1)
    if len(idxs) == 0:
        plt.text(0.5, 0.5, f"No {label}\ncases", ha="center"); plt.axis("off"); continue
    j = int(idxs[0]); img = test_images[j]
    heat = make_gradcam(img[np.newaxis, ...], cnn)
    plt.imshow(overlay(img, heat))
    plt.title(f"{label}\ntrue={class_names[yt[j]]}, pred={class_names[yp[j]]}", fontsize=9)
    plt.axis("off")
plt.suptitle("Grad-CAM – Custom CNN"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/13_gradcam_cases.png", dpi=150); plt.show()

best = max(all_res, key=lambda r: r["accuracy"])
print("Best model:", best["name"])
yt, yp = best["y_true"], best["y_pred"]
wrong = np.where(yt != yp)[0]
print(f"Misclassified: {len(wrong)} / {len(yt)}")
plt.figure(figsize=(10, 6))
for i, j in enumerate(wrong[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[j].astype("uint8"))
    plt.title(f"true={class_names[yt[j]]}\npred={class_names[yp[j]]}",
              fontsize=9, color="red"); plt.axis("off")
plt.suptitle(f"Misclassified Examples – {best['name']}"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/14_error_analysis.png", dpi=150); plt.show()

cnn.save(f"{MODEL_DIR}/custom_cnn.keras")
mnet.save(f"{MODEL_DIR}/mobilenetv2.keras")
resnet.save(f"{MODEL_DIR}/resnet50.keras")
print("Saved models and figures to /kaggle/working/")
for f in sorted(os.listdir(FIG_DIR)):
    print("  figures/", f)
