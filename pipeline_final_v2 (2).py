"""
Détection de crises d'épilepsie (CHB-MIT) - version 2
MNE -> segmentation -> features -> X / y / patient_id -> Leave-One-Patient-Out
-> XGBoost -> Accuracy / Precision / Recall / F1 -> matrice de confusion
-> sauvegarde du modèle

Nouveautés : normalisation par patient + lissage temporel, avec comparaison
des 4 configurations (base, +lissage, +normalisation, +les deux).

Installation :
    pip install mne numpy scipy pandas scikit-learn xgboost matplotlib joblib
"""
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier

# ----------------------------- PARAMÈTRES ---------------------------------
DATA_DIR = Path(r"C:\Users\Pc Lenovo\Downloads\dataset")
OUT_DIR = Path(r"C:\Users\Pc Lenovo\Desktop\prototype\resultats")
CACHE_DIR = OUT_DIR / "cache_v2"          # même cache que la version précédente

EPOCH_SEC = 5.0
OVERLAP_SEC = 0.0
L_FREQ, H_FREQ = 0.5, 45.0
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
EXCLUDE = {"-", "VNS", "ECG", "LOC", "ROC"}
MAX_SPW = 50
SMOOTH_SIZE = 3                           # taille du filtre médian (3 ou 5)
RANDOM_STATE = 42
# --------------------------------------------------------------------------

mne.set_log_level("ERROR")


# ------------------------------ LABELS ------------------------------------
def parse_summary(path):
    text = Path(path).read_text(errors="ignore")
    out = {}
    for block in re.split(r"File Name:\s*", text)[1:]:
        name = block.split()[0].strip()
        starts = re.findall(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", block)
        ends = re.findall(r"Seizure\s*\d*\s*End Time:\s*(\d+)", block)
        out[name] = [(int(s), int(e)) for s, e in zip(starts, ends)]
    return out


def load_all_summaries():
    files = sorted(DATA_DIR.rglob("*summary*.txt"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier *summary*.txt dans {DATA_DIR}")
    seizures = {}
    for f in files:
        seizures.update(parse_summary(f))
    return seizures


# ------------------------------ SIGNAL ------------------------------------
def clean_names(names):
    return [n.strip().upper() for n in names]


def find_common_channels(edf_files):
    common = None
    for f in edf_files:
        raw = mne.io.read_raw_edf(f, preload=False)
        names = {n for n in clean_names(raw.ch_names)
                 if n not in EXCLUDE and not re.search(r"-\d$", n)}
        common = names if common is None else common & names
    if not common:
        raise RuntimeError("Aucun canal commun entre les fichiers EDF.")
    return sorted(common)


def hjorth(x):
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    v0, v1, v2 = x.var(-1), dx.var(-1), ddx.var(-1)
    eps = 1e-12
    mob = np.sqrt(v1 / (v0 + eps))
    cpx = np.sqrt(v2 / (v1 + eps)) / (mob + eps)
    return v0, mob, cpx


def extract_features(data, sfreq):
    n_ep, n_ch, _ = data.shape
    nperseg = min(data.shape[-1], int(2 * sfreq))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    total = psd[..., (freqs >= L_FREQ) & (freqs <= H_FREQ)].sum(-1) + 1e-12

    feats = []
    for lo, hi in BANDS.values():
        bp = psd[..., (freqs >= lo) & (freqs < hi)].sum(-1)
        feats += [np.log10(bp + 1e-12), bp / total]

    act, mob, cpx = hjorth(data.reshape(-1, data.shape[-1]))
    feats += [np.log10(act.reshape(n_ep, n_ch) + 1e-12),
              mob.reshape(n_ep, n_ch), cpx.reshape(n_ep, n_ch)]
    feats += [data.mean(-1), data.std(-1), skew(data, axis=-1),
              kurtosis(data, axis=-1), np.ptp(data, axis=-1)]
    return np.concatenate(feats, axis=1)


def process_file(edf_path, channels, seizures):
    raw = mne.io.read_raw_edf(edf_path, preload=False)
    raw.rename_channels(dict(zip(raw.ch_names, clean_names(raw.ch_names))))
    raw.pick(channels)
    raw.reorder_channels(channels)
    raw.load_data()
    raw.filter(L_FREQ, H_FREQ)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_SEC, overlap=OVERLAP_SEC, preload=True)
    X = extract_features(epochs.get_data(), raw.info["sfreq"])

    t0 = np.arange(len(X)) * (EPOCH_SEC - OVERLAP_SEC)
    t1 = t0 + EPOCH_SEC
    y = np.zeros(len(X), dtype=int)
    for s, e in seizures:
        y[(t0 < e) & (t1 > s)] = 1
    return X, y


# ------------------------------ DATASET -----------------------------------
def build_dataset():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable : {DATA_DIR}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(DATA_DIR.rglob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"Aucun .edf trouvé dans {DATA_DIR}")
    seizure_map = load_all_summaries()

    usable = []
    for f in edf_files:
        if f.name in seizure_map:
            usable.append(f)
        else:
            print(f"[AVERTISSEMENT] {f.name} absent des summary -> ignoré")

    # ordre chronologique : patient puis nom de fichier
    usable.sort(key=lambda p: p.name)

    channels = find_common_channels(usable)
    print(f"{len(usable)} fichiers EDF, {len(channels)} canaux communs\n")

    X_list, y_list, g_list, f_list = [], [], [], []
    for f in usable:
        pid = f.stem.split("_")[0]
        cache = CACHE_DIR / f"{f.stem}.npz"
        if cache.exists():
            d = np.load(cache)
            X, y = d["X"], d["y"]
        else:
            X, y = process_file(f, channels, seizure_map[f.name])
            np.savez(cache, X=X, y=y)
        X_list.append(X)
        y_list.append(y)
        g_list += [pid] * len(X)
        f_list += [f.stem] * len(X)
        print(f"  {pid} | {f.name:18s} | {len(X):4d} segments | "
              f"{int(y.sum()):3d} segments de crise")

    X = np.nan_to_num(np.vstack(X_list), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.concatenate(y_list)
    groups, files = np.array(g_list), np.array(f_list)

    for p in np.unique(groups):
        if int(y[groups == p].sum()) == 0:
            print(f"[AVERTISSEMENT] {p} n'a aucun segment de crise")
    if y.sum() == 0:
        raise RuntimeError("Aucune crise dans les données.")
    return X, y, groups, files


# ------------------------- NORMALISATION / LISSAGE ------------------------
def normalize_per_patient(X, groups):
    """z-score des features à l'intérieur de chaque patient."""
    Xn = X.copy()
    for p in np.unique(groups):
        m = groups == p
        Xn[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Xn


def smooth_predictions(pred, files):
    """Filtre médian appliqué séparément dans chaque fichier EDF."""
    out = pred.copy()
    for f in np.unique(files):
        m = files == f
        out[m] = median_filter(pred[m], size=SMOOTH_SIZE, mode="nearest")
    return out


# ------------------------------ MODELE ------------------------------------
def spw_of(y):
    pos = y.sum()
    return min((len(y) - pos) / pos, MAX_SPW) if pos else 1.0


def make_model(spw):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        random_state=RANDOM_STATE, n_jobs=-1)


def run_lopo(X, y, groups, label):
    """Retourne les prédictions brutes LOPO (alignées sur y)."""
    pred = np.zeros_like(y)
    print(f"\n--- LOPO : {label} ---")
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model = make_model(spw_of(y[tr]))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        print(f"  Test={groups[te][0]} | segments crise={int(y[te].sum())} | "
              f"recall={recall_score(y[te], pred[te], zero_division=0):.3f} | "
              f"precision={precision_score(y[te], pred[te], zero_division=0):.3f}")
    return pred


def all_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups, files = build_dataset()
    print(f"\nX={X.shape} | crises={int(y.sum())} ({100 * y.mean():.2f}%) | "
          f"patients={list(np.unique(groups))}")
    np.savez(OUT_DIR / "dataset.npz", X=X, y=y, patient_id=groups, source_file=files)

    Xn = normalize_per_patient(X, groups)

    # 2 entraînements LOPO ; le lissage s'applique ensuite aux prédictions
    pred_raw = run_lopo(X, y, groups, "features brutes")
    pred_norm = run_lopo(Xn, y, groups, "features normalisées par patient")

    configs = {
        "1. Base": (pred_raw, False),
        "2. + lissage": (smooth_predictions(pred_raw, files), False),
        "3. + normalisation": (pred_norm, True),
        "4. + normalisation + lissage": (smooth_predictions(pred_norm, files), True),
    }

    # ---------------- Comparaison ----------------
    table = pd.DataFrame({k: all_metrics(y, p) for k, (p, _) in configs.items()}).T
    print("\n=== Comparaison des configurations (LOPO, tous patients) ===")
    print(table.round(4).to_string())
    table.to_csv(OUT_DIR / "comparaison_configs.csv")

    best_name = table["f1"].idxmax()
    best_pred, best_norm = configs[best_name]
    print(f"\nMeilleure configuration (F1) : {best_name}")

    # ---------------- Détail de la meilleure config ----------------
    table.loc[[best_name]].T.to_csv(OUT_DIR / "metrics.csv", header=["valeur"])
    rows = []
    for p in np.unique(groups):
        m = groups == p
        rows.append({"patient": p, "segments_crise": int(y[m].sum()),
                     **all_metrics(y[m], best_pred[m])})
    per_patient = pd.DataFrame(rows)
    per_patient.to_csv(OUT_DIR / "metrics_par_patient.csv", index=False)
    print("\nPar patient :\n", per_patient.round(3).to_string(index=False))

    cm = confusion_matrix(y, best_pred, labels=[0, 1])
    print("\nMatrice de confusion [[VN FP] [FN VP]] :\n", cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Crise"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matrice de confusion (LOPO)\n{best_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "matrice_confusion.png", dpi=150)

    # ---------------- Modèle final + sauvegarde ----------------
    X_final = Xn if best_norm else X
    final = make_model(spw_of(y))
    final.fit(X_final, y)
    final.save_model(OUT_DIR / "xgboost_eeg.json")
    joblib.dump({"model": final,
                 "normalize_per_patient": best_norm,
                 "smooth_size": SMOOTH_SIZE if "lissage" in best_name else None,
                 "config": best_name},
                OUT_DIR / "xgboost_eeg.joblib")
    print(f"\nTerminé. Résultats et modèle dans : {OUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
