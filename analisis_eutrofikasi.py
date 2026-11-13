"""
Jalankan dari folder proyek:
    pip install jupyter
jupyter notebook Notebook_Analisis_Eutrofikasi.ipynb

File yang dihasilkan: CSV simulasi, grafik, metrik evaluasi, dan model joblib.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SEED = 61
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
CSV_PATH = DATA_DIR / "dataset_simulasi_eutrofikasi.csv"

FEATURES = [
    "total_nitrogen_mg_l",
    "water_temperature_c",
    "dissolved_oxygen_mg_l",
    "annual_rainfall_mm",
    "agricultural_land_use_pct",
    "urban_land_use_pct",
    "mean_lake_depth_m",
    "ph",
]
CLASS_ORDER = ["Oligotrofik", "Mesotrofik", "Eutrofik"]


def calculate_tsi(tp_ug_l: float, chlorophyll_ug_l: float, secchi_m: float) -> tuple[float, float, float, float]:
    """Carlson TSI components; TP in µg/L, Chl-a in µg/L, Secchi in m."""
    tsi_tp = 14.42 * np.log(tp_ug_l) + 4.15
    tsi_chla = 9.81 * np.log(chlorophyll_ug_l) + 30.6
    tsi_secchi = 60.0 - 14.41 * np.log(secchi_m)
    ctsi = float(np.mean([tsi_tp, tsi_chla, tsi_secchi]))
    return float(tsi_tp), float(tsi_chla), float(tsi_secchi), ctsi


def class_from_ctsi(ctsi: float) -> str:
    if ctsi < 40:
        return "Oligotrofik"
    if ctsi <= 50:
        return "Mesotrofik"
    return "Eutrofik"


def generate_dataset(n_lakes: int = 20, samples_per_lake: int = 60) -> pd.DataFrame:
    """Create an explicitly synthetic tropical-lake-like educational dataset."""
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for lake_idx in range(n_lakes):
        lake_id = f"SIM-LAKE-{lake_idx + 1:03d}"
        agriculture = float(rng.uniform(8, 82))
        urban = float(rng.uniform(2, 28))
        depth = float(np.exp(rng.uniform(np.log(1.2), np.log(18))))
        rainfall = float(np.clip(rng.normal(1850, 420), 800, 3200))
        base_temp = float(rng.normal(27.2, 1.4))
        lake_effect = float(rng.normal(0, 0.35))

        for sample_idx in range(samples_per_lake):
            month = int(rng.integers(1, 13))
            year = int(rng.integers(2018, 2026))
            day = int(rng.integers(1, 28))
            sample_date = pd.Timestamp(year=year, month=month, day=day)
            temp = float(np.clip(
                base_temp + 0.9 * np.sin((month - 1) / 12 * 2 * np.pi) + rng.normal(0, 0.8),
                19, 34,
            ))

            # Latent nutrient-pressure factor used only to construct the simulation.
            pressure = (
                0.80 * ((agriculture - 45) / 22)
                + 0.23 * ((rainfall - 1850) / 420)
                + 0.14 * ((temp - 27) / 2)
                - 0.30 * ((depth - 6) / 4)
                + 0.10 * ((urban - 12) / 8)
                + lake_effect
                + rng.normal(0, 0.6)
            )

            tn = float(np.clip(0.45 * np.exp(0.28 * pressure + rng.normal(0, 0.15)), 0.05, 3.0))
            tp = float(np.clip(16 * np.exp(0.52 * pressure + rng.normal(0, 0.24)), 1.0, 150.0))
            chla = float(np.clip(4.0 * np.exp(0.62 * pressure + rng.normal(0, 0.26)), 0.1, 120.0))
            secchi = float(np.clip(3.4 * np.exp(-0.28 * pressure + rng.normal(0, 0.22)), 0.25, 15.0))
            do = float(np.clip(8.7 - 0.55 * pressure - 0.13 * (temp - 27) + rng.normal(0, 0.55), 2.0, 13.0))
            ph = float(np.clip(7.4 + 0.18 * pressure + rng.normal(0, 0.22), 6.0, 9.2))

            tsi_tp, tsi_chla, tsi_secchi, ctsi = calculate_tsi(tp, chla, secchi)
            label = class_from_ctsi(ctsi)

            rows.append({
                "sample_id": f"SIM-{lake_idx + 1:03d}-{sample_idx + 1:03d}",
                "lake_id": lake_id,
                "sample_date": sample_date.date().isoformat(),
                "total_nitrogen_mg_l": round(tn, 4),
                "water_temperature_c": round(temp, 3),
                "dissolved_oxygen_mg_l": round(do, 3),
                "annual_rainfall_mm": round(rainfall, 2),
                "agricultural_land_use_pct": round(agriculture, 2),
                "urban_land_use_pct": round(urban, 2),
                "mean_lake_depth_m": round(depth, 3),
                "ph": round(ph, 3),
                "total_phosphorus_ug_l": round(tp, 4),
                "chlorophyll_a_ug_l": round(chla, 4),
                "secchi_depth_m": round(secchi, 4),
                "tsi_total_phosphorus": round(tsi_tp, 3),
                "tsi_chlorophyll_a": round(tsi_chla, 3),
                "tsi_secchi_depth": round(tsi_secchi, 3),
                "carlson_tsi_composite": round(ctsi, 3),
                "eutrophication_class": label,
                "data_status": "SIMULASI - BUKAN PENGUKURAN LAPANGAN",
            })

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    return df


def make_model(kind: str):
    if kind == "logistic_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=SEED,
            )),
        ])
    if kind == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=400,
                max_depth=9,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            )),
        ])
    raise ValueError(f"Model tidak dikenal: {kind}")


def save_figures(df: pd.DataFrame, y_test: pd.Series, pred: np.ndarray,
                 model, X_test: pd.DataFrame, test_metrics: dict,
                 model_comparison: pd.DataFrame) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    files: dict[str, str] = {}

    # 1. Distribusi label
    fig, ax = plt.subplots(figsize=(8, 4.7))
    counts = df["eutrophication_class"].value_counts().reindex(CLASS_ORDER)
    colors = ["#2b8cbe", "#f0ad4e", "#d9534f"]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.62)
    ax.set_title("Distribusi Kelas Status Trofik (Data Simulasi)", weight="bold")
    ax.set_xlabel("Kelas trofik")
    ax.set_ylabel("Jumlah sampel")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 7,
                f"{value} ({value/len(df):.1%})", ha="center", fontsize=9)
    ax.set_ylim(0, max(counts.values) * 1.18)
    fig.tight_layout()
    p = FIG_DIR / "01_distribusi_kelas.png"; fig.savefig(p, dpi=190); plt.close(fig); files["class_distribution"] = str(p)

    # 2. Distribusi CTSI
    fig, ax = plt.subplots(figsize=(8, 4.7))
    palette = {"Oligotrofik": "#2b8cbe", "Mesotrofik": "#f0ad4e", "Eutrofik": "#d9534f"}
    sns.histplot(data=df, x="carlson_tsi_composite", hue="eutrophication_class",
                 bins=30, element="step", stat="count", common_norm=False,
                 palette=palette, ax=ax)
    ax.axvline(40, color="#555555", linestyle="--", linewidth=1.2, label="Batas 40")
    ax.axvline(50, color="#333333", linestyle=":", linewidth=1.4, label="Batas 50")
    ax.set_title("Sebaran Carlson Trophic State Index (CTSI)", weight="bold")
    ax.set_xlabel("Nilai CTSI komposit")
    ax.set_ylabel("Jumlah sampel")
    ax.legend(title="Kelas / batas", fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / "02_sebaran_ctsi.png"; fig.savefig(p, dpi=190); plt.close(fig); files["ctsi_distribution"] = str(p)

    # 3. TP vs chlorophyll-a (indikator pembentuk target, bukan fitur model)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="total_phosphorus_ug_l", y="chlorophyll_a_ug_l",
                    hue="eutrophication_class", hue_order=CLASS_ORDER,
                    palette=palette, alpha=0.62, s=28, ax=ax)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Total Fosfor dan Klorofil-a Menurut Kelas Trofik", weight="bold")
    ax.set_xlabel("Total fosfor (µg/L, skala log)")
    ax.set_ylabel("Klorofil-a (µg/L, skala log)")
    ax.legend(title="Kelas trofik")
    fig.tight_layout()
    p = FIG_DIR / "03_tp_chla_kelas.png"; fig.savefig(p, dpi=190); plt.close(fig); files["tp_chla"] = str(p)

    # 4. Korelasi fitur prediktor
    fig, ax = plt.subplots(figsize=(9, 6.8))
    corr = df[FEATURES].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, linewidths=0.4, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Korelasi Antarfitur Prediktor", weight="bold")
    fig.tight_layout()
    p = FIG_DIR / "04_korelasi_fitur.png"; fig.savefig(p, dpi=190); plt.close(fig); files["feature_correlation"] = str(p)

    # 5. Confusion matrix
    cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Confusion Matrix - Logistic Regression", weight="bold")
    ax.set_xlabel("Prediksi model")
    ax.set_ylabel("Kelas sebenarnya")
    fig.tight_layout()
    p = FIG_DIR / "05_confusion_matrix.png"; fig.savefig(p, dpi=190); plt.close(fig); files["confusion_matrix"] = str(p)

    # 6. Comparison metrics
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    comp = model_comparison.set_index("model")[["accuracy", "balanced_accuracy", "macro_f1"]]
    comp.plot(kind="bar", ax=ax, color=["#4c78a8", "#72b7b2", "#f2cf5b"], width=0.7)
    ax.set_title("Perbandingan Kinerja Model pada Data Uji", weight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(["Regresi Logistik", "Random Forest"], rotation=0)
    ax.legend(["Accuracy", "Balanced accuracy", "Macro F1"], loc="lower right", fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / "06_perbandingan_model.png"; fig.savefig(p, dpi=190); plt.close(fig); files["model_comparison"] = str(p)

    # 7. Permutation importance
    pi = permutation_importance(model, X_test, y_test, n_repeats=12,
                                random_state=SEED, scoring="f1_macro", n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=True)
    imp.to_csv(OUT_DIR / "feature_importance_permutation.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#4c78a8", alpha=0.9)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_title("Permutation Feature Importance (Macro F1)", weight="bold")
    ax.set_xlabel("Penurunan skor Macro F1 saat fitur diacak")
    ax.set_ylabel("Fitur")
    fig.tight_layout()
    p = FIG_DIR / "07_feature_importance.png"; fig.savefig(p, dpi=190); plt.close(fig); files["feature_importance"] = str(p)

    return files


def run_analysis(df: pd.DataFrame | None = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.read_csv(CSV_PATH)

    # Pemeriksaan kualitas data.
    duplicate_rows = int(df.duplicated().sum())
    missing_values = int(df[FEATURES].isna().sum().sum())
    if duplicate_rows:
        df = df.drop_duplicates().copy()
    if df["eutrophication_class"].isna().any():
        raise ValueError("Label target ditemukan kosong.")

    X = df[FEATURES].copy()
    y = df["eutrophication_class"].copy()
    groups = df["lake_id"].copy()

    # Pisahkan grup danau agar sampel dari danau yang sama tidak bocor ke data uji.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_lakes = sorted(groups.iloc[test_idx].unique().tolist())

    # Validasi silang hanya dilakukan pada data latih; grup pada data uji
    # tidak boleh ikut menentukan pemilihan model.
    groups_train = groups.iloc[train_idx]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    model_specs = {
        "Regresi Logistik": "logistic_regression",
        "Random Forest": "random_forest",
    }
    rows = []
    fitted_models = {}
    for display_name, kind in model_specs.items():
        m = make_model(kind)
        cv_scores = cross_val_score(m, X_train, y_train, groups=groups_train, cv=cv,
                                    scoring="f1_macro", n_jobs=-1)
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        rows.append({
            "model": display_name,
            "accuracy": float(accuracy_score(y_test, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, p)),
            "macro_f1": float(f1_score(y_test, p, average="macro")),
            "cv_macro_f1_mean": float(cv_scores.mean()),
            "cv_macro_f1_std": float(cv_scores.std()),
        })
        fitted_models[kind] = m

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT_DIR / "perbandingan_model.csv", index=False)
    # Pilih model berdasarkan rata-rata macro-F1 pada grouped CV.
    selected_row = comparison.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
    selected_display = str(selected_row["model"])
    selected_kind = model_specs[selected_display]
    selected_model = fitted_models[selected_kind]
    y_pred = selected_model.predict(X_test)
    probabilities = selected_model.predict_proba(X_test)
    auc = float(roc_auc_score(y_test, probabilities,
                              labels=selected_model.classes_,
                              multi_class="ovr", average="macro"))

    metrics = {
        "dataset_type": "SIMULASI - bukan pengukuran lapangan",
        "seed": SEED,
        "n_rows": int(len(df)),
        "n_lakes": int(df["lake_id"].nunique()),
        "n_features": int(len(FEATURES)),
        "features": FEATURES,
        "excluded_target_components": ["total_phosphorus_ug_l", "chlorophyll_a_ug_l", "secchi_depth_m", "carlson_tsi_composite", "tsi_total_phosphorus", "tsi_chlorophyll_a", "tsi_secchi_depth"],
        "duplicate_rows_before_cleaning": duplicate_rows,
        "missing_predictor_values": missing_values,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "test_lake_ids": test_lakes,
        "class_counts": {k: int(v) for k, v in df["eutrophication_class"].value_counts().reindex(CLASS_ORDER).fillna(0).items()},
        "class_percentages": {k: float(v) for k, v in (df["eutrophication_class"].value_counts(normalize=True).reindex(CLASS_ORDER).fillna(0) * 100).items()},
        "selected_model": selected_display,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "test_roc_auc_ovr_macro": auc,
        "classification_report": classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0),
        "cv_macro_f1_mean": float(selected_row["cv_macro_f1_mean"]),
        "cv_macro_f1_std": float(selected_row["cv_macro_f1_std"]),
        "model_comparison": rows,
    }

    # Persist final model, metrics, and predictions.
    joblib.dump({"model": selected_model, "features": FEATURES, "class_order": CLASS_ORDER,
                 "data_status": "SIMULASI - bukan pengukuran lapangan"}, OUT_DIR / "model_klasifikasi_eutrofikasi.joblib")
    result_df = df.iloc[test_idx][["sample_id", "lake_id", "sample_date", "eutrophication_class"]].copy()
    result_df["predicted_class"] = y_pred
    result_df["prediction_correct"] = result_df["eutrophication_class"] == result_df["predicted_class"]
    for i, cls in enumerate(selected_model.classes_):
        result_df[f"probability_{cls.lower()}"] = probabilities[:, i]
    result_df.to_csv(OUT_DIR / "hasil_prediksi_data_uji.csv", index=False)

    with open(OUT_DIR / "metrik_evaluasi.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(classification_report(y_test, y_pred, labels=CLASS_ORDER,
                                       output_dict=True, zero_division=0)).T.to_csv(
        OUT_DIR / "classification_report.csv", encoding="utf-8-sig")

    fig_files = save_figures(df, y_test, y_pred, selected_model, X_test, metrics, comparison)
    metrics["figures"] = fig_files
    with open(OUT_DIR / "metrik_evaluasi.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    df = generate_dataset()
    metrics = run_analysis(df)
    print("Dataset:", CSV_PATH)
    print("Jumlah sampel:", metrics["n_rows"])
    print("Jumlah danau simulasi:", metrics["n_lakes"])
    print("Model terpilih:", metrics["selected_model"])
    print("Akurasi uji: {:.2%}".format(metrics["test_accuracy"]))
    print("Macro F1 uji: {:.2%}".format(metrics["test_macro_f1"]))
    print("Macro F1 grouped CV: {:.2%}".format(metrics["cv_macro_f1_mean"]))
    print("Seluruh keluaran tersimpan di folder data/, figures/, dan outputs/.")


if __name__ == "__main__":
    main()
