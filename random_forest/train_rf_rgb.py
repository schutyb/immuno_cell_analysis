from pathlib import Path

import joblib
import numpy as np
import tifffile as tiff
from PIL import Image

from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from skimage.filters import gaussian, sobel, laplace

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# =========================
# EDITAR
# =========================

VISIT_FOLDER = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01"

MODEL_OUTPUT = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/models/rf_rgb_visit01.joblib"

PATCH_DIR_NAME = "patch"
PATCH_MASK_DIR_NAME = "patch_mask"

RANDOM_SEED = 0
NEGATIVE_TO_POSITIVE_RATIO = 4

N_ESTIMATORS = 300
MAX_DEPTH = 25


# =========================
# LOADERS
# =========================

def load_rgb(path):
    path = Path(path)

    if path.suffix.lower() in [".tif", ".tiff"]:
        img = tiff.imread(path)
    else:
        img = np.array(Image.open(path).convert("RGB"))

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"RGB inválido: {path}, shape={img.shape}")

    return img.astype(np.uint8)


def load_mask(path):
    path = Path(path)

    if path.suffix.lower() in [".tif", ".tiff"]:
        mask = tiff.imread(path)
    else:
        mask = np.array(Image.open(path).convert("L"))

    mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8)


# =========================
# FEATURES RGB ONLY
# =========================

def extract_rgb_features(img):
    img_f = img.astype(np.float32) / 255.0

    R = img_f[..., 0]
    G = img_f[..., 1]
    B = img_f[..., 2]

    total = R + G + B + 1e-8

    r_norm = R / total
    g_norm = G / total
    b_norm = B / total

    hsv = rgb2hsv(img_f)
    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    lab = rgb2lab(img_f)
    L = lab[..., 0] / 100.0
    A = lab[..., 1] / 128.0
    BB = lab[..., 2] / 128.0

    gray = rgb2gray(img_f).astype(np.float32)

    grad = sobel(gray).astype(np.float32)

    gauss1 = gaussian(gray, sigma=1, preserve_range=True).astype(np.float32)
    gauss2 = gaussian(gray, sigma=2, preserve_range=True).astype(np.float32)
    gauss4 = gaussian(gray, sigma=4, preserve_range=True).astype(np.float32)

    log1 = laplace(gauss1).astype(np.float32)
    log2 = laplace(gauss2).astype(np.float32)

    features = np.stack(
        [
            R, G, B,
            r_norm, g_norm, b_norm,
            R - G, R - B, G - B,
            R / (G + 1e-8),
            R / (B + 1e-8),
            G / (B + 1e-8),
            H, S, V,
            L, A, BB,
            gray,
            grad,
            gauss1, gauss2, gauss4,
            log1, log2,
        ],
        axis=-1,
    )

    return features.reshape(-1, features.shape[-1]).astype(np.float32)


# =========================
# DATASET
# =========================

def find_patch_pairs(visit_folder):
    visit_folder = Path(visit_folder)

    patch_files = sorted(
        visit_folder.glob(f"Mosaic*/random_forest/{PATCH_DIR_NAME}/*.png")
    )

    pairs = []

    for img_path in patch_files:
        random_forest_dir = img_path.parent.parent
        mask_dir = random_forest_dir / PATCH_MASK_DIR_NAME

        expected_mask = mask_dir / f"{img_path.stem}_mask.png"

        if expected_mask.exists():
            pairs.append((img_path, expected_mask))
        else:
            print(f"WARNING: no mask for {img_path.name}")
            print(f"Expected: {expected_mask}")

    return pairs


def sample_pixels_from_patch(img, mask, rng):
    X = extract_rgb_features(img)
    y = mask.reshape(-1).astype(np.uint8)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0:
        n_neg = min(800, len(neg_idx))
        selected_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        return X[selected_neg], y[selected_neg]

    n_neg = min(len(neg_idx), len(pos_idx) * NEGATIVE_TO_POSITIVE_RATIO)

    selected_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    selected_idx = np.concatenate([pos_idx, selected_neg])

    rng.shuffle(selected_idx)

    return X[selected_idx], y[selected_idx]


def build_training_dataset(pairs):
    rng = np.random.default_rng(RANDOM_SEED)

    X_list = []
    y_list = []

    for img_path, mask_path in pairs:
        img = load_rgb(img_path)
        mask = load_mask(mask_path)

        if img.shape[:2] != mask.shape:
            print(f"SKIP shape mismatch: {img_path.name}")
            continue

        Xp, yp = sample_pixels_from_patch(img, mask, rng)

        X_list.append(Xp)
        y_list.append(yp)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


# =========================
# TRAIN
# =========================

def train_random_forest(X, y):
    print("\nDataset:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"positive pixels: {np.sum(y == 1)}")
    print(f"negative pixels: {np.sum(y == 0)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    print("\nValidation:")
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred, digits=4))

    return clf


def main():
    pairs = find_patch_pairs(VISIT_FOLDER)

    print(f"Patches encontrados: {len(pairs)}")

    if len(pairs) == 0:
        raise RuntimeError("No se encontraron patches/masks.")

    X, y = build_training_dataset(pairs)

    clf = train_random_forest(X, y)

    model_path = Path(MODEL_OUTPUT)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, model_path)

    print(f"\nModelo guardado en:")
    print(model_path)


if __name__ == "__main__":
    main()