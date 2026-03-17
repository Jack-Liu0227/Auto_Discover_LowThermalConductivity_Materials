from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False

try:
    from matminer.featurizers.composition import ElementProperty

    MATMINER_AVAILABLE = True
except ImportError:
    MATMINER_AVAILABLE = False
    ElementProperty = None

try:
    from pymatgen.core import Composition, Element

    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    Composition = None
    Element = None


def _float_or_default(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        text = str(value).split()[0]
        try:
            return float(text)
        except Exception:
            return default


def _estimate_valence(element: "Element") -> float:
    group = int(element.group)
    if 1 <= group <= 2:
        return float(group)
    if 13 <= group <= 18:
        return float(group - 10)
    positive_states = [int(state) for state in element.common_oxidation_states if int(state) > 0]
    if positive_states:
        return float(min(positive_states))
    try:
        valence = getattr(element, "valence", None)
        if isinstance(valence, tuple) and len(valence) == 2:
            return float(valence[1])
    except Exception:
        pass
    return 0.0


@lru_cache(maxsize=1)
def _build_atomic_props() -> dict[str, dict[str, float]]:
    cache_path = Path(__file__).with_name("atomic_props_cache.json")
    if PYMATGEN_AVAILABLE and Element is not None:
        atomic_props: dict[str, dict[str, float]] = {}
        for atomic_number in range(1, 119):
            element = Element.from_Z(atomic_number)
            radius = element.atomic_radius
            if radius is None:
                radius = element.average_ionic_radius
            try:
                eneg = _float_or_default(element.X)
            except Exception:
                eneg = 0.0
            atomic_props[element.symbol] = {
                "mass": _float_or_default(element.atomic_mass),
                "radius": _float_or_default(radius),
                "eneg": eneg,
                "group": float(element.group),
                "period": float(element.row),
                "valence": _estimate_valence(element),
            }
        return atomic_props
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    raise RuntimeError("Neither pymatgen nor atomic_props_cache.json is available for atomic properties")


ATOMIC_PROPS = _build_atomic_props()

TARGET_COLUMN = "k(W/Km)"
FORMULA_COLUMN = "Formula"
CORRELATION_THRESHOLD = 0.85
DEFAULT_TOP_N = 15
DEFAULT_RANDOM_STATE = 42
DEFAULT_SELECTION_METRIC = "rmse"


@dataclass
class BootstrapArtifacts:
    input_path: str
    output_dir: str
    character_features_path: str
    feature_target_correlation_path: str
    feature_correlation_matrix_path: str
    correlation_pruned_features_path: str
    shap_importance_path: str
    selected_features_path: str
    manifest_path: str
    correlation_heatmap_path: str
    shap_summary_path: str
    feature_curve_csv_path: str
    feature_curve_plot_path: str


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [col for col in df.columns if not str(col).startswith("Unnamed:")]
    cleaned = df.loc[:, keep_cols].copy()
    blank_cols = [col for col in cleaned.columns if str(col).strip() == ""]
    if blank_cols:
        cleaned = cleaned.drop(columns=blank_cols)
    all_nan_cols = [col for col in cleaned.columns if cleaned[col].isna().all()]
    if all_nan_cols:
        cleaned = cleaned.drop(columns=all_nan_cols)
    return cleaned


def parse_formula(formula: str) -> dict[str, float]:
    composition: dict[str, float] = {}

    def expand_paren(match: re.Match[str]) -> str:
        content, factor = match.groups()
        multiplier = float(factor) if factor else 1.0
        sub_matches = re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", content)
        expanded_parts: list[str] = []
        for element, count in sub_matches:
            original_count = float(count) if count else 1.0
            expanded_parts.append(f"{element}{original_count * multiplier}")
        return "".join(expanded_parts)

    expanded_formula = re.sub(r"\((.*?)\)(\d*\.?\d*)", expand_paren, str(formula).replace(" ", ""))
    final_matches = re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", expanded_formula)
    for element, count in final_matches:
        amount = float(count) if count else 1.0
        composition[element] = composition.get(element, 0.0) + amount
    return composition


def wtpercentstr_to_formula(wt_str: Any) -> str:
    if pd.isnull(wt_str):
        return ""
    return str(wt_str).replace(";", "").replace(":", "").replace(",", "").strip()


def load_initial_dataset(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = _drop_unnamed_columns(df)
    required = [FORMULA_COLUMN, TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[FORMULA_COLUMN].isna().any():
        raise ValueError("Formula column contains missing values")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values")
    return df


def _build_working_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "wt%" not in df.columns:
        df["wt%"] = df[FORMULA_COLUMN]
    df["wt_formula_for_matminer"] = df["wt%"].apply(wtpercentstr_to_formula)
    df["Composition"] = df[FORMULA_COLUMN].apply(parse_formula)
    unsupported: set[str] = set()
    for composition in df["Composition"]:
        unsupported.update(set(composition) - set(ATOMIC_PROPS))
    if unsupported:
        raise ValueError(f"Unsupported elements: {sorted(unsupported)}")
    return df


def calculate_manual_features(
    df: pd.DataFrame,
    all_elements: set[str],
    ratio_series_dict: dict[str, pd.Series],
    total_atoms_series: pd.Series,
) -> dict[str, Any]:
    features: dict[str, Any] = {}

    atom_fraction_dict = {
        element: pd.Series(
            np.where(total_atoms_series > 0, ratio_series_dict[element] / total_atoms_series, 0.0),
            index=df.index,
        )
        for element in all_elements
    }

    def safe_mean(weighted_sum: Any, total_atoms: pd.Series | np.ndarray) -> np.ndarray:
        result = np.where(total_atoms > 0, weighted_sum / total_atoms, 0.0)
        return result.flatten() if getattr(result, "ndim", 1) > 1 else result

    prop_names = ["mass", "radius", "eneg", "valence", "group", "period"]
    weighted_sums = {prop: 0.0 for prop in prop_names}

    for element in all_elements:
        prop = ATOMIC_PROPS[element]
        atom_fraction = atom_fraction_dict[element]
        for prop_name in prop_names:
            weighted_sums[prop_name] += atom_fraction * prop[prop_name]

    avg_prop = {prop: weighted_sums[prop] for prop in prop_names}

    def calculate_variance(prop_name: str, avg_series: np.ndarray) -> np.ndarray:
        diff_sq_series = 0.0
        for element in all_elements:
            prop_val = ATOMIC_PROPS[element][prop_name]
            atom_fraction = atom_fraction_dict[element]
            diff_sq_series += atom_fraction * (prop_val - avg_series) ** 2
        return diff_sq_series.flatten() if getattr(diff_sq_series, "ndim", 1) > 1 else diff_sq_series

    def calculate_range(prop_name: str, composition: dict[str, float]) -> float:
        values = [ATOMIC_PROPS[element][prop_name] for element, ratio in composition.items() if ratio > 0 and element in ATOMIC_PROPS]
        return (max(values) - min(values)) if values else 0.0

    def calculate_min(prop_name: str, composition: dict[str, float]) -> float:
        values = [ATOMIC_PROPS[element][prop_name] for element, ratio in composition.items() if ratio > 0 and element in ATOMIC_PROPS]
        return min(values) if values else 0.0

    def calculate_max(prop_name: str, composition: dict[str, float]) -> float:
        values = [ATOMIC_PROPS[element][prop_name] for element, ratio in composition.items() if ratio > 0 and element in ATOMIC_PROPS]
        return max(values) if values else 0.0

    features["n_elements"] = df["Composition"].apply(lambda composition: sum(1 for ratio in composition.values() if ratio > 0))
    features["total_atoms"] = total_atoms_series

    nonzero_fraction_list = [
        np.array([float(atom_fraction_dict[element].iloc[idx]) for element in all_elements if float(atom_fraction_dict[element].iloc[idx]) > 0])
        for idx in range(len(df))
    ]
    features["max_atom_fraction"] = pd.Series(
        [float(fractions.max()) if fractions.size else 0.0 for fractions in nonzero_fraction_list],
        index=df.index,
    )
    features["min_atom_fraction"] = pd.Series(
        [float(fractions.min()) if fractions.size else 0.0 for fractions in nonzero_fraction_list],
        index=df.index,
    )
    features["atom_fraction_std"] = pd.Series(
        [float(fractions.std()) if fractions.size else 0.0 for fractions in nonzero_fraction_list],
        index=df.index,
    )
    features["config_entropy"] = pd.Series(
        [float(-np.sum(fractions * np.log(np.clip(fractions, 1e-12, None)))) if fractions.size else 0.0 for fractions in nonzero_fraction_list],
        index=df.index,
    )

    for prop_name in prop_names:
        features[f"{prop_name.capitalize()}_mean"] = avg_prop[prop_name]
        variance = calculate_variance(prop_name, avg_prop[prop_name])
        features[f"{prop_name.capitalize()}_variance"] = variance
        features[f"{prop_name.capitalize()}_std"] = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))
        features[f"{prop_name.capitalize()}_min"] = df["Composition"].apply(
            lambda composition, name=prop_name: calculate_min(name, composition)
        )
        features[f"{prop_name.capitalize()}_max"] = df["Composition"].apply(
            lambda composition, name=prop_name: calculate_max(name, composition)
        )
        features[f"{prop_name.capitalize()}_range"] = df["Composition"].apply(
            lambda composition, name=prop_name: calculate_range(name, composition)
        )

    def calculate_radius_delta(index: int) -> float:
        if total_atoms_series.iloc[index] == 0:
            return 0.0
        avg_radius = float(avg_prop["radius"][index])
        if avg_radius == 0.0:
            return 0.0
        squared_sum = 0.0
        for element in all_elements:
            ratio = float(ratio_series_dict[element].iloc[index])
            if ratio <= 0:
                continue
            x_i = float(atom_fraction_dict[element].iloc[index])
            r_i = ATOMIC_PROPS[element]["radius"]
            squared_sum += x_i * (1.0 - r_i / avg_radius) ** 2
        return float(np.sqrt(squared_sum))

    features["Radius_delta"] = pd.Series([calculate_radius_delta(i) for i in range(len(df))], index=df.index)

    def calculate_property_delta(index: int, prop_name: str) -> float:
        avg_value = float(avg_prop[prop_name][index])
        if avg_value == 0.0:
            return 0.0
        squared_sum = 0.0
        for element in all_elements:
            atom_fraction = float(atom_fraction_dict[element].iloc[index])
            if atom_fraction <= 0:
                continue
            prop_value = ATOMIC_PROPS[element][prop_name]
            squared_sum += atom_fraction * (1.0 - prop_value / avg_value) ** 2
        return float(np.sqrt(squared_sum))

    features["Mass_delta"] = pd.Series([calculate_property_delta(i, "mass") for i in range(len(df))], index=df.index)
    features["Eneg_delta"] = pd.Series([calculate_property_delta(i, "eneg") for i in range(len(df))], index=df.index)
    features["Valence_delta"] = pd.Series([calculate_property_delta(i, "valence") for i in range(len(df))], index=df.index)

    def calculate_mixing_enthalpy(index: int) -> float:
        if total_atoms_series.iloc[index] == 0:
            return 0.0
        total_sum = 0.0
        elements = [el for el in all_elements if float(ratio_series_dict[el].iloc[index]) > 0 and el in ATOMIC_PROPS]
        for left_idx, left_el in enumerate(elements):
            for right_idx, right_el in enumerate(elements):
                if left_idx >= right_idx:
                    continue
                x1 = float(atom_fraction_dict[left_el].iloc[index])
                x2 = float(atom_fraction_dict[right_el].iloc[index])
                eneg1 = ATOMIC_PROPS[left_el]["eneg"]
                eneg2 = ATOMIC_PROPS[right_el]["eneg"]
                total_sum += x1 * x2 * (eneg1 - eneg2) ** 2
        return float(4.0 * total_sum)

    features["Mixing_enthalpy"] = pd.Series([calculate_mixing_enthalpy(i) for i in range(len(df))], index=df.index)
    features["Mixing_entropy"] = features["config_entropy"] * 8.314
    return features


def build_manual_feature_frame(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build custom, non-matminer composition features."""
    df = _build_working_dataframe(raw_df)
    all_elements: set[str] = set()
    for composition in df["Composition"]:
        for element in composition:
            if element in ATOMIC_PROPS:
                all_elements.add(element)

    total_atoms_series = df["Composition"].apply(lambda composition: sum(composition.values()))
    ratio_series_dict = {
        element: df["Composition"].apply(lambda composition, el=element: composition.get(el, 0.0)).copy()
        for element in all_elements
    }

    manual_features = calculate_manual_features(df, all_elements, ratio_series_dict, total_atoms_series)
    feature_cols_manual = list(manual_features.keys())
    df_temp = df.drop(columns=["Composition", "wt_formula_for_matminer"], errors="ignore")
    df_temp = pd.concat([df_temp, pd.DataFrame(manual_features, index=df.index)], axis=1)
    return df_temp, feature_cols_manual


def build_matminer_feature_frame(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build matminer Magpie composition features."""
    df = _build_working_dataframe(raw_df)
    additional_wt_features = pd.DataFrame(index=df.index)
    matminer_cols: list[str] = []
    if MATMINER_AVAILABLE and Composition is not None:
        try:
            compositions: list[Any] = []
            for formula in df["wt_formula_for_matminer"]:
                comp_obj = None
                if formula:
                    try:
                        comp_obj = Composition(formula)
                    except Exception:
                        comp_obj = None
                compositions.append(comp_obj)

            valid_idx = [index for index, composition in enumerate(compositions) if composition is not None]
            valid_comps = [compositions[index] for index in valid_idx]
            if valid_comps:
                temp_df = pd.DataFrame({"composition": valid_comps})
                featurizer = ElementProperty.from_preset("magpie")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    feature_df = featurizer.featurize_dataframe(temp_df, "composition", ignore_errors=True)

                recovered_columns: dict[str, list[Any]] = {}
                for col in feature_df.columns:
                    if col == "composition":
                        continue
                    values = [np.nan] * len(df)
                    for row_index, original_index in enumerate(valid_idx):
                        value = feature_df.iloc[row_index][col]
                        if isinstance(value, np.ndarray) and value.ndim > 0:
                            values[original_index] = value.flatten()[0] if value.size > 0 else np.nan
                        elif not pd.isnull(value):
                            values[original_index] = value
                    recovered_columns[col] = values
                    matminer_cols.append(col)
                if recovered_columns:
                    additional_wt_features = pd.DataFrame(recovered_columns, index=df.index)
        except Exception:
            additional_wt_features = pd.DataFrame(index=df.index)
            matminer_cols = []
    return additional_wt_features, matminer_cols


def build_character_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    manual_df, feature_cols_manual = build_manual_feature_frame(raw_df)
    additional_wt_features, matminer_cols = build_matminer_feature_frame(raw_df)

    out_df = pd.concat([manual_df, additional_wt_features], axis=1)
    check_cols = feature_cols_manual + matminer_cols
    drop_cols: set[str] = set()
    for col in check_cols:
        if col not in out_df.columns:
            continue
        if out_df[col].isnull().all():
            drop_cols.add(col)
            continue
        unique_vals = out_df[col].dropna().unique()
        if len(unique_vals) <= 1:
            drop_cols.add(col)

    keep_cols = [FORMULA_COLUMN, TARGET_COLUMN] + [col for col in check_cols if col not in drop_cols]
    return out_df[keep_cols].copy()


def _sanitize_feature_frame(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_only = feature_df.drop(columns=[FORMULA_COLUMN, TARGET_COLUMN]).copy()
    removed: list[str] = []

    empty_cols = [col for col in feature_only.columns if feature_only[col].isna().all()]
    if empty_cols:
        feature_only = feature_only.drop(columns=empty_cols)
        removed.extend(empty_cols)

    constant_cols = [col for col in feature_only.columns if feature_only[col].nunique(dropna=True) <= 1]
    if constant_cols:
        feature_only = feature_only.drop(columns=constant_cols)
        removed.extend(constant_cols)

    duplicated_mask = feature_only.T.duplicated()
    duplicated_cols = feature_only.columns[duplicated_mask].tolist()
    if duplicated_cols:
        feature_only = feature_only.drop(columns=duplicated_cols)
        removed.extend(duplicated_cols)

    cleaned = pd.concat([feature_df[[FORMULA_COLUMN, TARGET_COLUMN]], feature_only], axis=1)
    return cleaned, removed


def _prune_correlated_features(
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float,
) -> tuple[list[str], list[dict[str, Any]], pd.DataFrame]:
    correlations = features.corr().abs()
    target_corr = features.apply(lambda col: col.corr(target)).fillna(0.0)
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    dropped: set[str] = set()
    drop_records: list[dict[str, Any]] = []

    for left in upper.columns:
        for right in upper.columns:
            corr_value = upper.loc[left, right]
            if pd.isna(corr_value) or corr_value <= threshold:
                continue
            if left in dropped or right in dropped:
                continue
            keep = left if abs(float(target_corr[left])) >= abs(float(target_corr[right])) else right
            drop = right if keep == left else left
            dropped.add(drop)
            drop_records.append(
                {
                    "kept_feature": keep,
                    "dropped_feature": drop,
                    "pair_correlation": float(corr_value),
                    "kept_target_correlation": float(target_corr[keep]),
                    "dropped_target_correlation": float(target_corr[drop]),
                }
            )

    selected = [col for col in features.columns if col not in dropped]
    corr_table = pd.DataFrame(
        {
            "feature": target_corr.index,
            "target_correlation": target_corr.values,
            "abs_target_correlation": target_corr.abs().values,
        }
    ).sort_values("abs_target_correlation", ascending=False)
    return selected, drop_records, corr_table


def _build_model(model_type: str, random_state: int):
    if model_type == "xgboost":
        if not XGBOOST_AVAILABLE or XGBRegressor is None:
            raise ImportError("xgboost is not installed. Use --model-type extra_trees or install xgboost.")
        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
    if model_type == "extra_trees":
        return ExtraTreesRegressor(n_estimators=300, random_state=random_state, n_jobs=1)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _compute_shap_importance(
    features: pd.DataFrame,
    target: pd.Series,
    output_dir: Path,
    model_type: str,
    random_state: int,
) -> pd.DataFrame:
    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )
    model = _build_model(model_type=model_type, random_state=random_state)
    model.fit(x_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]
    mean_abs = np.abs(shap_array).mean(axis=0)
    shap_df = pd.DataFrame({"feature": features.columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_array, features, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    return shap_df


def _evaluate_ranked_feature_curve(
    ranked_features: list[str],
    features: pd.DataFrame,
    target: pd.Series,
    output_dir: Path,
    model_type: str,
    random_state: int,
) -> pd.DataFrame:
    if not ranked_features:
        return pd.DataFrame(columns=["feature_count", "r2_mean", "r2_std", "mae_mean", "mae_std", "rmse_mean", "rmse_std"])

    def _rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scorers = {
        "r2": make_scorer(r2_score),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "rmse": make_scorer(_rmse_score, greater_is_better=False),
    }

    rows: list[dict[str, float]] = []
    for feature_count in range(1, len(ranked_features) + 1):
        current_features = ranked_features[:feature_count]
        model = _build_model(model_type=model_type, random_state=random_state)
        scores = cross_validate(
            model,
            features[current_features],
            target,
            cv=cv,
            scoring=scorers,
            n_jobs=1,
        )
        rows.append(
            {
                "feature_count": float(feature_count),
                "r2_mean": float(np.mean(scores["test_r2"])),
                "r2_std": float(np.std(scores["test_r2"])),
                "mae_mean": float(-np.mean(scores["test_mae"])),
                "mae_std": float(np.std(-scores["test_mae"])),
                "rmse_mean": float(-np.mean(scores["test_rmse"])),
                "rmse_std": float(np.std(-scores["test_rmse"])),
            }
        )

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(output_dir / "feature_count_metrics.csv", index=False, encoding="utf-8-sig")

    def _safe_best_index(series: pd.Series, mode: str) -> int | None:
        valid_series = series.dropna()
        if valid_series.empty:
            return None
        return int(valid_series.idxmax() if mode == "max" else valid_series.idxmin())

    best_r2_idx = _safe_best_index(curve_df["r2_mean"], mode="max")
    best_mae_idx = _safe_best_index(curve_df["mae_mean"], mode="min")
    best_rmse_idx = _safe_best_index(curve_df["rmse_mean"], mode="min")

    plt.figure(figsize=(12, 7))
    plt.plot(curve_df["feature_count"], curve_df["r2_mean"], marker="o", label="R2", color="#1f77b4")
    plt.plot(curve_df["feature_count"], curve_df["mae_mean"], marker="o", label="MAE", color="#d62728")
    plt.plot(curve_df["feature_count"], curve_df["rmse_mean"], marker="o", label="RMSE", color="#2ca02c")

    for idx, color, metric_col, label in [
        (best_r2_idx, "#1f77b4", "r2_mean", "Best R2"),
        (best_mae_idx, "#d62728", "mae_mean", "Best MAE"),
        (best_rmse_idx, "#2ca02c", "rmse_mean", "Best RMSE"),
    ]:
        if idx is None:
            continue
        x_val = float(curve_df.loc[idx, "feature_count"])
        y_val = float(curve_df.loc[idx, metric_col])
        plt.axvline(x=x_val, linestyle="--", color=color, alpha=0.5)
        plt.scatter([x_val], [y_val], color=color, s=120, marker="*", zorder=5)
        plt.text(x_val + 0.2, y_val, f"{label}: {int(x_val)}", color=color, fontsize=10, va="center")

    plt.xlabel("Number of Features Included")
    plt.ylabel("Cross-Validation Metric")
    plt.title(f"Model Performance vs. Number of Features ({model_type})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "feature_count_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    return curve_df


def _select_feature_count_from_curve(curve_df: pd.DataFrame, selection_metric: str) -> int:
    if curve_df.empty:
        return 0

    metric_name = selection_metric.lower()
    if metric_name == "r2":
        series = curve_df["r2_mean"].dropna()
        if series.empty:
            return 0
        return int(curve_df.loc[series.idxmax(), "feature_count"])
    if metric_name == "mae":
        series = curve_df["mae_mean"].dropna()
        if series.empty:
            return 0
        return int(curve_df.loc[series.idxmin(), "feature_count"])
    if metric_name == "rmse":
        series = curve_df["rmse_mean"].dropna()
        if series.empty:
            return 0
        return int(curve_df.loc[series.idxmin(), "feature_count"])
    raise ValueError(f"Unsupported selection_metric: {selection_metric}")


def _plot_correlation_heatmap(correlation_matrix: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 10))
    if SEABORN_AVAILABLE:
        sns.heatmap(correlation_matrix, cmap="coolwarm", center=0.0)
    else:
        matrix = correlation_matrix.to_numpy(dtype=float)
        image = plt.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        plt.colorbar(image, fraction=0.046, pad=0.04)
        ticks = range(len(correlation_matrix.columns))
        plt.xticks(ticks, correlation_matrix.columns, rotation=90, fontsize=6)
        plt.yticks(ticks, correlation_matrix.index, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_character_feature_bootstrap(
    input_path: str | Path,
    output_dir: str | Path,
    correlation_threshold: float = CORRELATION_THRESHOLD,
    top_n: int | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    model_type: str = "xgboost",
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> BootstrapArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_initial_dataset(input_path)
    feature_df = build_character_features(raw_df)
    feature_df, removed_cols = _sanitize_feature_frame(feature_df)
    feature_df.to_csv(output_dir / "character_features.csv", index=False, encoding="utf-8-sig")

    feature_only = feature_df.drop(columns=[FORMULA_COLUMN, TARGET_COLUMN])
    correlation_matrix = feature_only.corr()
    correlation_matrix.to_csv(output_dir / "feature_correlation_matrix.csv", encoding="utf-8-sig")
    _plot_correlation_heatmap(correlation_matrix, output_dir / "feature_correlation_heatmap.png")

    selected_after_corr, drop_records, corr_table = _prune_correlated_features(
        features=feature_only,
        target=feature_df[TARGET_COLUMN],
        threshold=correlation_threshold,
    )
    corr_table.to_csv(output_dir / "feature_target_correlation.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "correlation_pruned_features.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": correlation_threshold,
                "selected_features": selected_after_corr,
                "dropped_pairs": drop_records,
                "removed_during_sanitize": removed_cols,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    corr_pruned_features = feature_only[selected_after_corr].copy()
    shap_df = _compute_shap_importance(
        features=corr_pruned_features,
        target=feature_df[TARGET_COLUMN],
        output_dir=output_dir,
        model_type=model_type,
        random_state=random_state,
    )
    shap_df.to_csv(output_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")
    curve_df = _evaluate_ranked_feature_curve(
        ranked_features=shap_df["feature"].tolist(),
        features=corr_pruned_features,
        target=feature_df[TARGET_COLUMN],
        output_dir=output_dir,
        model_type=model_type,
        random_state=random_state,
    )

    auto_selected_count = _select_feature_count_from_curve(curve_df, selection_metric=selection_metric)
    final_selected_count = int(top_n) if top_n is not None else auto_selected_count
    final_selected_count = max(1, min(final_selected_count, len(shap_df))) if len(shap_df) > 0 else 0

    selected_features = shap_df.head(final_selected_count)["feature"].tolist()
    with (output_dir / "selected_features.json").open("w", encoding="utf-8") as handle:
        json.dump(selected_features, handle, indent=2, ensure_ascii=False)

    manifest = {
        "input_path": str(Path(input_path)),
        "output_dir": str(output_dir),
        "target_column": TARGET_COLUMN,
        "correlation_threshold": correlation_threshold,
        "top_n_override": top_n,
        "random_state": random_state,
        "model_type": model_type,
        "selection_metric": selection_metric,
        "auto_selected_feature_count": int(auto_selected_count),
        "final_selected_feature_count": int(final_selected_count),
        "matminer_available": MATMINER_AVAILABLE,
        "n_samples": int(len(feature_df)),
        "candidate_feature_count": int(feature_only.shape[1]),
        "post_correlation_feature_count": int(len(selected_after_corr)),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "removed_during_sanitize": removed_cols,
        "feature_curve_rows": int(len(curve_df)),
        "best_r2_feature_count": int(curve_df.loc[curve_df["r2_mean"].dropna().idxmax(), "feature_count"]) if curve_df["r2_mean"].notna().any() else 0,
        "best_mae_feature_count": int(curve_df.loc[curve_df["mae_mean"].dropna().idxmin(), "feature_count"]) if curve_df["mae_mean"].notna().any() else 0,
        "best_rmse_feature_count": int(curve_df.loc[curve_df["rmse_mean"].dropna().idxmin(), "feature_count"]) if curve_df["rmse_mean"].notna().any() else 0,
    }
    with (output_dir / "feature_selection_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return BootstrapArtifacts(
        input_path=str(Path(input_path)),
        output_dir=str(output_dir),
        character_features_path=str(output_dir / "character_features.csv"),
        feature_target_correlation_path=str(output_dir / "feature_target_correlation.csv"),
        feature_correlation_matrix_path=str(output_dir / "feature_correlation_matrix.csv"),
        correlation_pruned_features_path=str(output_dir / "correlation_pruned_features.json"),
        shap_importance_path=str(output_dir / "shap_importance.csv"),
        selected_features_path=str(output_dir / "selected_features.json"),
        manifest_path=str(output_dir / "feature_selection_manifest.json"),
        correlation_heatmap_path=str(output_dir / "feature_correlation_heatmap.png"),
        shap_summary_path=str(output_dir / "shap_summary.png"),
        feature_curve_csv_path=str(output_dir / "feature_count_metrics.csv"),
        feature_curve_plot_path=str(output_dir / "feature_count_metrics.png"),
    )


def resolve_default_input(project_root: str | Path) -> Path:
    project_root = Path(project_root)
    candidates = [
        project_root / "data" / "processed_data.csv",
        project_root / "llm" / "data" / "iteration_0" / "data.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find initial dataset in data/processed_data.csv or llm/data/iteration_0/data.csv")


def resolve_default_output(project_root: str | Path) -> Path:
    return Path(project_root) / "featureEngeering"


def artifacts_to_dict(artifacts: BootstrapArtifacts) -> dict[str, Any]:
    return asdict(artifacts)
