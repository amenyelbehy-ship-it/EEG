"""
Prototype d'aide à la lecture EEG (détection de périodes suspectes de crise)

Le médecin charge un fichier EDF ; l'application applique le modèle XGBoost
entraîné, puis affiche :
  - la courbe du score du modèle dans le temps,
  - la liste des périodes suspectes (début, fin, durée, score),
  - le tracé EEG autour de chaque période, pour vérification visuelle,
  - un résumé texte et un export CSV.

Installation :  pip install streamlit
Lancement   :  streamlit run app_medecin.py
(à lancer depuis le dossier qui contient aussi pipeline_final_v2.py)

AVERTISSEMENT : prototype de recherche, non validé cliniquement.
"""
import json
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import median_filter

# Réutilise EXACTEMENT les mêmes fonctions qu'à l'entraînement (filtre, segments, features)
import pipeline_final_v2 as pf

# ----------------------------- PARAMÈTRES ---------------------------------
BASE_DIR = Path(r"C:\Users\Pc Lenovo\Desktop\prototype")
MODEL_PATH = BASE_DIR / "resultats" / "xgboost_eeg.joblib"
CHANNELS_JSON = BASE_DIR / "resultats" / "canaux_entrainement.json"
# --------------------------------------------------------------------------

mne.set_log_level("ERROR")
st.set_page_config(page_title="Aide à la lecture EEG", layout="wide")


# ------------------------------ CHARGEMENTS -------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict):
        return obj["model"], obj
    return obj, {}


@st.cache_resource
def get_training_channels():
    """Canaux utilisés à l'entraînement (mêmes règles que pipeline_final_v2)."""
    if CHANNELS_JSON.exists():
        return json.loads(CHANNELS_JSON.read_text())
    edfs = sorted(pf.DATA_DIR.rglob("*.edf"))
    if not edfs:
        raise FileNotFoundError(
            f"Impossible de retrouver les canaux d'entraînement : aucun .edf dans {pf.DATA_DIR}")
    channels = pf.find_common_channels(edfs)
    CHANNELS_JSON.write_text(json.dumps(channels))
    return channels


# ------------------------------ ANALYSE -----------------------------------
def analyse_recording(file_bytes, channels, model, meta):
    """EDF -> signal filtré (pour affichage) + score du modèle par segment."""
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        raw = mne.io.read_raw_edf(path, preload=False)
        raw.rename_channels(dict(zip(raw.ch_names, pf.clean_names(raw.ch_names))))
        missing = [c for c in channels if c not in raw.ch_names]
        if missing:
            raise ValueError("Canaux manquants dans ce fichier : " + ", ".join(missing))
        raw.pick(channels)
        raw.reorder_channels(channels)
        raw.load_data()
        raw.filter(pf.L_FREQ, pf.H_FREQ)
        sfreq = raw.info["sfreq"]

        epochs = mne.make_fixed_length_epochs(
            raw, duration=pf.EPOCH_SEC, overlap=pf.OVERLAP_SEC, preload=True)
        X = pf.extract_features(epochs.get_data(), sfreq)
        signal = (raw.get_data() * 1e6).astype(np.float32)       # microvolts, pour l'affichage
    finally:
        os.unlink(path)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Le modèle attend {model.n_features_in_} features, mais ce fichier en donne "
            f"{X.shape[1]} (canaux différents de l'entraînement ?).")

    if meta.get("normalize_per_patient", False):
        X = pf.normalize_per_patient(X, np.zeros(len(X), dtype=int))

    probs = model.predict_proba(X)[:, 1]
    return {"signal": signal, "sfreq": sfreq, "probs": probs, "channels": channels}


# ------------------------------ ÉVÉNEMENTS --------------------------------
def make_events(pred, probs):
    """Fusionne les segments positifs consécutifs en périodes suspectes."""
    step = pf.EPOCH_SEC - pf.OVERLAP_SEC
    events, i, n = [], 0, len(pred)
    while i < n:
        if pred[i] == 1:
            j = i
            while j + 1 < n and pred[j + 1] == 1:
                j += 1
            events.append({
                "debut": i * step,
                "fin": j * step + pf.EPOCH_SEC,
                "score_moy": float(probs[i:j + 1].mean()),
                "score_max": float(probs[i:j + 1].max()),
            })
            i = j + 1
        else:
            i += 1
    for e in events:
        e["duree"] = e["fin"] - e["debut"]
    return events


def fmt_time(t):
    t = int(round(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def events_table(events):
    return pd.DataFrame({
        "N°": range(1, len(events) + 1),
        "Début": [fmt_time(e["debut"]) for e in events],
        "Fin": [fmt_time(e["fin"]) for e in events],
        "Durée (s)": [round(e["duree"]) for e in events],
        "Score moyen": [round(e["score_moy"], 2) for e in events],
        "Score max": [round(e["score_max"], 2) for e in events],
    })


def build_report(filename, duration, events):
    lines = [f"Fichier analysé : {filename}",
             f"Durée de l'enregistrement : {fmt_time(duration)}"]
    if not events:
        lines.append("Aucune période suspecte de crise n'a été signalée par le modèle.")
    else:
        total = sum(e["duree"] for e in events)
        lines.append(f"Le modèle signale {len(events)} période(s) suspecte(s), "
                     f"pour une durée cumulée d'environ {round(total)} s.")
        for k, e in enumerate(events, 1):
            lines.append(f"  {k}. de {fmt_time(e['debut'])} à {fmt_time(e['fin'])} "
                         f"({round(e['duree'])} s), score max {e['score_max']:.2f}")
    lines += ["",
              "Ce résultat est une aide à la lecture produite par un prototype de recherche.",
              "Il ne constitue pas un diagnostic et doit être vérifié sur le tracé EEG."]
    return "\n".join(lines)


# ------------------------------ GRAPHIQUES --------------------------------
def plot_scores(probs, thr, events):
    step = pf.EPOCH_SEC - pf.OVERLAP_SEC
    t = np.arange(len(probs)) * step / 60
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(t, probs, lw=1, color="#1F3864")
    ax.axhline(thr, color="gray", ls="--", lw=1, label=f"seuil = {thr:.2f}")
    for k, e in enumerate(events):
        ax.axvspan(e["debut"] / 60, e["fin"] / 60, color="red", alpha=0.25,
                   label="période suspecte" if k == 0 else None)
    ax.set_xlabel("Temps (minutes)")
    ax.set_ylabel("Score du modèle")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, t[-1] + step / 60 if len(t) else 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_eeg(signal, ch_names, sfreq, t_start, t_end, events):
    i0, i1 = int(t_start * sfreq), int(t_end * sfreq)
    seg = signal[:, i0:i1]
    seg = (seg - seg.mean(1, keepdims=True)) / (seg.std(1, keepdims=True) + 1e-9)
    t = np.arange(seg.shape[1]) / sfreq + t_start
    n = len(ch_names)
    fig, ax = plt.subplots(figsize=(11, 0.3 * n + 1.5))
    offsets = (n - 1 - np.arange(n)) * 4
    for k in range(n):
        ax.plot(t, seg[k] + offsets[k], lw=0.6, color="black")
    for e in events:
        if e["fin"] > t_start and e["debut"] < t_end:
            ax.axvspan(max(e["debut"], t_start), min(e["fin"], t_end),
                       color="red", alpha=0.15)
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names, fontsize=7)
    ax.set_xlabel("Temps (s)")
    ax.set_xlim(t_start, t_end)
    fig.tight_layout()
    return fig


# ------------------------------ INTERFACE ---------------------------------
st.title("Aide à la lecture EEG : périodes suspectes de crise")
st.warning(
    "Prototype de recherche, non validé cliniquement. Les résultats sont une aide à la "
    "lecture, pas un diagnostic : toute période signalée doit être vérifiée sur le tracé.")

try:
    model, meta = load_model()
    channels = get_training_channels()
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.header("Paramètres")
uploaded = st.sidebar.file_uploader("Fichier EEG (.edf)", type=["edf"])
thr = st.sidebar.slider("Seuil de détection", 0.05, 0.95, 0.50, 0.05,
                        help="Plus il est bas, plus le modèle est sensible (plus d'alarmes, "
                             "dont de fausses). Plus il est haut, plus il est strict.")
default_smooth = meta.get("smooth_size") or 1
smooth = st.sidebar.select_slider(
    "Lissage (segments)", options=[1, 3, 5], value=default_smooth if default_smooth in (1, 3, 5) else 1,
    help="Supprime les alarmes isolées. 1 = pas de lissage.")
st.sidebar.caption(f"Modèle : XGBoost | configuration : {meta.get('config', 'inconnue')}")

if uploaded is None:
    st.info("Charge un fichier EDF dans le menu de gauche pour lancer l'analyse.")
    st.stop()

key = (uploaded.name, uploaded.size)
if st.session_state.get("rec_key") != key:
    try:
        with st.spinner("Analyse en cours (filtrage, segmentation, calcul des features)..."):
            st.session_state["rec"] = analyse_recording(
                uploaded.getvalue(), channels, model, meta)
            st.session_state["rec_key"] = key
    except Exception as e:
        st.error(f"Analyse impossible : {e}")
        st.stop()

rec = st.session_state["rec"]
signal, sfreq, probs = rec["signal"], rec["sfreq"], rec["probs"]
duration = signal.shape[1] / sfreq

if meta.get("normalize_per_patient") and duration < 600:
    st.warning("Enregistrement court (< 10 min) : la normalisation par patient est moins fiable.")

pred = (probs >= thr).astype(int)
if smooth > 1:
    pred = median_filter(pred, size=smooth, mode="nearest")
events = make_events(pred, probs)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Durée", fmt_time(duration))
c2.metric("Segments analysés", len(probs))
c3.metric("Périodes suspectes", len(events))
c4.metric("Temps suspect cumulé", f"{round(sum(e['duree'] for e in events))} s")

st.subheader("Score du modèle dans le temps")
st.pyplot(plot_scores(probs, thr, events))
st.caption("Le score va de 0 à 1. Ce n'est pas une probabilité calibrée : "
           "il sert à comparer les segments entre eux.")

st.subheader("Périodes suspectes")
if events:
    table = events_table(events)
    st.dataframe(table, hide_index=True)
    st.download_button("Télécharger la liste (CSV)",
                       table.to_csv(index=False).encode("utf-8-sig"),
                       file_name="periodes_suspectes.csv", mime="text/csv")
else:
    st.success("Aucune période suspecte avec ces réglages.")

st.subheader("Tracé EEG")
win = st.slider("Durée affichée (s)", 10, 60, 30, 5)
if events:
    mode = st.radio("Affichage", ["Autour d'une période suspecte", "Fenêtre manuelle"],
                    horizontal=True)
else:
    mode = "Fenêtre manuelle"

if mode == "Autour d'une période suspecte":
    idx = st.selectbox(
        "Période", range(len(events)),
        format_func=lambda k: f"Période {k + 1} ({fmt_time(events[k]['debut'])} - "
                              f"{fmt_time(events[k]['fin'])})")
    t_start = max(0.0, events[idx]["debut"] - 10)
else:
    t_start = float(st.slider("Début de la fenêtre (s)", 0, int(max(1, duration - win)), 0))
t_end = min(duration, t_start + win)
st.pyplot(plot_eeg(signal, rec["channels"], sfreq, t_start, t_end, events))
st.caption("Chaque canal est mis à l'échelle séparément pour la lisibilité : "
           "les amplitudes ne sont pas comparables entre canaux. Zone rouge = période suspecte.")

st.subheader("Résumé")
report = build_report(uploaded.name, duration, events)
st.text(report)
st.download_button("Télécharger le résumé (.txt)", report.encode("utf-8"),
                   file_name="resume_analyse.txt", mime="text/plain")
