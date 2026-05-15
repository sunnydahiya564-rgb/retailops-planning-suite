from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful fallback if sklearn is absent
    SKLEARN_AVAILABLE = False


HISTORY_REQUIRED_COLUMNS = ["Month", "Variant ID", "Units Sold"]
HISTORY_COLUMN_ALIASES = {
    "date": "Month",
    "month": "Month",
    "sale date": "Month",
    "sales date": "Month",
    "billing date": "Month",
    "month start": "Month",
    "year-month": "Month",
    "year month": "Month",
    "period": "Month",
    "variant id": "Variant ID",
    "variant id (sku)": "Variant ID",
    "sku": "Variant ID",
    "item code": "Variant ID",
    "units sold": "Units Sold",
    "total units sold": "Units Sold",
    "monthly units sold": "Units Sold",
    "month units sold": "Units Sold",
    "sales": "Units Sold",
    "qty sold": "Units Sold",
    "quantity sold": "Units Sold",
    "quantity": "Units Sold",
    "sold qty": "Units Sold",
    "brand": "Brand",
    "category": "Category",
    "name": "Name",
}
SPECIAL_MONTH_OFFSET_ALIASES = {
    "same month last year": 12,
    "same month ly": 12,
    "last year same month": 12,
    "same month 2 years ago": 24,
    "same month 2y": 24,
    "same month 3 years ago": 36,
    "same month 3y": 36,
}
HISTORY_ID_COLUMNS = ["Variant ID", "Name", "Brand", "Category"]


@dataclass
class ForecastArtifacts:
    forecast: pd.DataFrame
    metrics: dict[str, float | str]
    monthly_history: pd.DataFrame


def normalize_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in df.columns:
        stripped = str(column).strip()
        alias = HISTORY_COLUMN_ALIASES.get(stripped.lower())
        if alias:
            rename_map[column] = alias
    return df.rename(columns=rename_map)


def _parse_month_header(header: str) -> pd.Timestamp | None:
    parsed = pd.to_datetime(str(header).strip(), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_period("M").to_timestamp()


def _convert_wide_history_to_long(history_df: pd.DataFrame) -> pd.DataFrame:
    clean_df = history_df.copy()
    id_columns = [column for column in HISTORY_ID_COLUMNS if column in clean_df.columns]
    if "Variant ID" not in id_columns:
        raise ValueError("Wide history upload must contain `Variant ID`.")

    month_columns: dict[str, pd.Timestamp] = {}
    seasonal_columns: dict[str, int] = {}
    for column in clean_df.columns:
        lowered = str(column).strip().lower()
        if column in id_columns:
            continue
        if lowered in SPECIAL_MONTH_OFFSET_ALIASES:
            seasonal_columns[column] = SPECIAL_MONTH_OFFSET_ALIASES[lowered]
            continue
        parsed_month = _parse_month_header(str(column))
        if parsed_month is not None:
            month_columns[column] = parsed_month

    if not month_columns:
        raise ValueError(
            "Could not detect month columns. Use either `Month, Variant ID, Units Sold` or a wide sheet with month names like `2026-04`."
        )

    latest_month = max(month_columns.values())
    forecast_month = latest_month + pd.offsets.MonthBegin(1)

    long_frames: list[pd.DataFrame] = []
    for column, month_value in month_columns.items():
        part = clean_df[id_columns + [column]].copy()
        part = part.rename(columns={column: "Units Sold"})
        part["Month"] = month_value
        long_frames.append(part)

    for column, offset_months in seasonal_columns.items():
        part = clean_df[id_columns + [column]].copy()
        part = part.rename(columns={column: "Units Sold"})
        part["Month"] = forecast_month - pd.DateOffset(months=offset_months)
        long_frames.append(part)

    return pd.concat(long_frames, ignore_index=True)


def _same_month_recent_value(series: pd.Series, target_month: int, rank_from_latest: int = 1) -> float:
    same_month_values = series[series.index.month == target_month]
    if same_month_values.empty or len(same_month_values) < rank_from_latest:
        return float("nan")
    return float(same_month_values.iloc[-rank_from_latest])


def prepare_history_frame(history_df: pd.DataFrame, sku_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    clean_df = normalize_history_columns(history_df.copy())
    if all(column in clean_df.columns for column in HISTORY_REQUIRED_COLUMNS):
        pass
    elif "Variant ID" in clean_df.columns:
        clean_df = _convert_wide_history_to_long(clean_df)
    else:
        missing_columns = [column for column in HISTORY_REQUIRED_COLUMNS if column not in clean_df.columns]
        raise ValueError(f"Missing required history columns: {', '.join(missing_columns)}")

    clean_df["Month"] = pd.to_datetime(clean_df["Month"], errors="coerce")
    clean_df = clean_df.dropna(subset=["Month"]).copy()
    clean_df["Month"] = clean_df["Month"].dt.to_period("M").dt.to_timestamp()
    clean_df["Variant ID"] = clean_df["Variant ID"].astype(str)
    clean_df["Units Sold"] = pd.to_numeric(clean_df["Units Sold"], errors="coerce").fillna(0).clip(lower=0)

    if sku_frame is not None:
        sku_lookup = sku_frame[["Variant ID", "Name", "Brand", "Category"]].drop_duplicates()
        clean_df = clean_df.merge(sku_lookup, on="Variant ID", how="left", suffixes=("", "_sku"))
        for column in ["Name", "Brand", "Category"]:
            sku_column = f"{column}_sku"
            if sku_column in clean_df.columns:
                clean_df[column] = clean_df[column].fillna(clean_df[sku_column])
                clean_df = clean_df.drop(columns=sku_column)

    for column in ["Name", "Brand", "Category"]:
        if column not in clean_df.columns:
            clean_df[column] = "Unknown"
        clean_df[column] = clean_df[column].fillna("Unknown").astype(str)

    monthly_history = (
        clean_df.groupby(["Month", "Variant ID", "Name", "Brand", "Category"], as_index=False)["Units Sold"]
        .sum()
        .sort_values(["Variant ID", "Month"])
        .reset_index(drop=True)
    )

    monthly_history["Year"] = monthly_history["Month"].dt.year
    monthly_history["Month Number"] = monthly_history["Month"].dt.month
    monthly_history["Month Label"] = monthly_history["Month"].dt.strftime("%b %Y")
    return monthly_history


def generate_demo_history(base_df: pd.DataFrame, periods: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    end_month = pd.Timestamp.today().normalize().replace(day=1)
    months = pd.date_range(end=end_month, periods=periods, freq="MS")

    category_seasonality = {
        "Personal Care": {11: 1.08, 12: 1.12, 1: 1.05},
        "Skin Care": {10: 1.07, 11: 1.09, 12: 1.11},
        "Baby Care": {7: 1.05, 8: 1.07},
        "Home Care": {3: 1.04, 4: 1.06, 10: 1.05},
        "Oral Care": {1: 1.06, 5: 1.04},
        "Snacks": {10: 1.08, 11: 1.1, 12: 1.13},
        "Beverages": {4: 1.08, 5: 1.12, 6: 1.15},
    }

    history_rows: list[dict[str, object]] = []
    for _, row in base_df.iterrows():
        base_per_day = max(float(row["Per Day"]), 0.5)
        brand = str(row["Brand"])
        category = str(row["Category"])
        name = str(row["Name"])
        sku = str(row["Variant ID"])
        brand_factor = 1 + ((abs(hash(brand)) % 9) / 100)
        sku_trend = 0.92 + ((abs(hash(sku)) % 15) / 100)

        for idx, month in enumerate(months):
            days = calendar.monthrange(month.year, month.month)[1]
            yearly_growth = 1 + ((idx / max(len(months) - 1, 1)) * 0.12)
            seasonal = category_seasonality.get(category, {}).get(month.month, 1.0)
            same_month_cycle = 1 + (0.06 * np.sin((month.month / 12) * 2 * np.pi))
            noise = rng.normal(0, base_per_day * days * 0.08)
            units_sold = max(base_per_day * days * brand_factor * sku_trend * yearly_growth * seasonal * same_month_cycle + noise, 0)
            history_rows.append(
                {
                    "Month": month,
                    "Variant ID": sku,
                    "Name": name,
                    "Brand": brand,
                    "Category": category,
                    "Units Sold": round(units_sold, 2),
                }
            )

    demo_history = pd.DataFrame(history_rows)
    return prepare_history_frame(demo_history, base_df)


def _complete_monthly_history(monthly_history: pd.DataFrame, sku_frame: pd.DataFrame) -> pd.DataFrame:
    if monthly_history.empty:
        return monthly_history.copy()

    min_month = monthly_history["Month"].min()
    max_month = monthly_history["Month"].max()
    all_months = pd.date_range(min_month, max_month, freq="MS")

    sku_info = sku_frame[["Variant ID", "Name", "Brand", "Category"]].drop_duplicates().copy()
    if sku_info.empty:
        sku_info = monthly_history[["Variant ID", "Name", "Brand", "Category"]].drop_duplicates().copy()

    all_rows: list[pd.DataFrame] = []
    for _, sku_row in sku_info.iterrows():
        sku_history = monthly_history[monthly_history["Variant ID"] == sku_row["Variant ID"]][["Month", "Units Sold"]]
        full_sku = pd.DataFrame({"Month": all_months})
        full_sku["Variant ID"] = sku_row["Variant ID"]
        full_sku["Name"] = sku_row["Name"]
        full_sku["Brand"] = sku_row["Brand"]
        full_sku["Category"] = sku_row["Category"]
        full_sku = full_sku.merge(sku_history, on="Month", how="left")
        full_sku["Units Sold"] = full_sku["Units Sold"].fillna(0)
        all_rows.append(full_sku)

    complete_history = pd.concat(all_rows, ignore_index=True)
    complete_history["Year"] = complete_history["Month"].dt.year
    complete_history["Month Number"] = complete_history["Month"].dt.month
    complete_history["Month Label"] = complete_history["Month"].dt.strftime("%b %Y")
    return complete_history.sort_values(["Variant ID", "Month"]).reset_index(drop=True)


def _same_month_average(series: pd.Series, target_month: int) -> float:
    same_month_values = series[series.index.month == target_month].tail(3)
    return float(same_month_values.mean()) if not same_month_values.empty else float("nan")


def _six_month_trend(series: pd.Series) -> float:
    tail = series.tail(6)
    if tail.empty:
        return 0.0
    if len(tail) == 1:
        return float(tail.iloc[-1])
    x = np.arange(len(tail))
    slope, intercept = np.polyfit(x, tail.to_numpy(dtype=float), 1)
    forecast = intercept + slope * len(tail)
    return float(max(forecast, 0))


def _build_training_frame(monthly_history: pd.DataFrame) -> pd.DataFrame:
    train_df = monthly_history.copy().sort_values(["Variant ID", "Month"]).reset_index(drop=True)

    grouped = train_df.groupby("Variant ID", group_keys=False)["Units Sold"]
    for lag in range(1, 7):
        train_df[f"lag_{lag}"] = grouped.shift(lag)

    train_df["rolling_3"] = grouped.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    train_df["rolling_6"] = grouped.shift(1).rolling(6).mean().reset_index(level=0, drop=True)
    train_df["same_month_last_year"] = grouped.shift(12)
    train_df["same_month_2y"] = grouped.shift(24)
    train_df["same_month_3y"] = grouped.shift(36)
    train_df["same_month_3y_avg"] = train_df[["same_month_last_year", "same_month_2y", "same_month_3y"]].mean(axis=1)

    train_df["month_sin"] = np.sin(2 * np.pi * train_df["Month Number"] / 12)
    train_df["month_cos"] = np.cos(2 * np.pi * train_df["Month Number"] / 12)
    train_df["quarter"] = train_df["Month"].dt.quarter
    train_df["year_index"] = train_df.groupby("Variant ID").cumcount()

    six_month_features: list[float] = []
    for _, sku_group in train_df.groupby("Variant ID"):
        sku_group = sku_group.sort_values("Month")
        values = sku_group["Units Sold"].tolist()
        for idx in range(len(values)):
            history_up_to_previous = pd.Series(values[:idx], dtype=float)
            six_month_features.append(_six_month_trend(history_up_to_previous))
    train_df["six_month_trend_feature"] = six_month_features

    return train_df


def _build_next_month_feature_frame(monthly_history: pd.DataFrame, sku_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    if monthly_history.empty:
        raise ValueError("Historical data is empty after preparation.")

    complete_history = _complete_monthly_history(monthly_history, sku_frame)
    last_month = complete_history["Month"].max()
    next_month = last_month + pd.offsets.MonthBegin(1)

    feature_rows: list[dict[str, object]] = []
    for _, sku_row in sku_frame[["Variant ID", "Name", "Brand", "Category", "FG", "Per Day", "Current", "Intransit", "Booked"]].drop_duplicates().iterrows():
        sku_history = complete_history[complete_history["Variant ID"] == sku_row["Variant ID"]].sort_values("Month")
        units_series = sku_history.set_index("Month")["Units Sold"]

        row: dict[str, object] = {
            "Variant ID": sku_row["Variant ID"],
            "Name": sku_row["Name"],
            "Brand": sku_row["Brand"],
            "Category": sku_row["Category"],
            "FG": float(sku_row["FG"]),
            "Current Per Day": float(sku_row["Per Day"]),
            "Current": float(sku_row["Current"]),
            "Intransit": float(sku_row["Intransit"]),
            "Booked": float(sku_row["Booked"]),
            "Month": next_month,
            "Month Number": next_month.month,
            "quarter": next_month.quarter,
            "month_sin": np.sin(2 * np.pi * next_month.month / 12),
            "month_cos": np.cos(2 * np.pi * next_month.month / 12),
            "Days in Next Month": calendar.monthrange(next_month.year, next_month.month)[1],
            "Last Month Sales": float(units_series.iloc[-1]) if not units_series.empty else 0.0,
            "six_month_trend_feature": _six_month_trend(units_series),
            "same_month_3y_avg": _same_month_average(units_series, next_month.month),
            "same_month_last_year_sold": _same_month_recent_value(units_series, next_month.month, rank_from_latest=1),
            "same_month_2y_sold": _same_month_recent_value(units_series, next_month.month, rank_from_latest=2),
            "same_month_3y_sold": _same_month_recent_value(units_series, next_month.month, rank_from_latest=3),
            "history_month_count": len(units_series),
        }
        for lag in range(1, 7):
            row[f"lag_{lag}"] = float(units_series.iloc[-lag]) if len(units_series) >= lag else np.nan
        row["rolling_3"] = float(units_series.tail(3).mean()) if len(units_series) >= 1 else 0.0
        row["rolling_6"] = float(units_series.tail(6).mean()) if len(units_series) >= 1 else 0.0
        row["year_index"] = int(len(units_series))
        feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)
    return feature_df, next_month


def _combine_projection_components(
    ml_value: float | int | np.floating | None,
    six_month_value: float | int | np.floating | None,
    seasonal_value: float | int | np.floating | None,
    fallback_units: float,
) -> tuple[float, str]:
    components = {
        "ML": (ml_value, 0.45),
        "6M Trend": (six_month_value, 0.30),
        "3Y Seasonality": (seasonal_value, 0.25),
    }
    available_components = {
        name: (float(value), weight)
        for name, (value, weight) in components.items()
        if value is not None and pd.notna(value) and float(value) >= 0
    }
    if not available_components:
        return max(float(fallback_units), 0.0), "Current per-day fallback"

    total_weight = sum(weight for _, weight in available_components.values())
    hybrid_units = sum(value * (weight / total_weight) for value, weight in available_components.values())
    return max(float(hybrid_units), 0.0), " + ".join(available_components.keys())


def _train_monthly_model(
    train_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float | str]]:
    baseline_metrics: dict[str, float | str] = {
        "model_name": "Hybrid ensemble",
        "mae": 0.0,
        "r2": 0.0,
    }
    evaluation_frame = train_df.groupby("Variant ID").tail(1).copy()
    if not SKLEARN_AVAILABLE:
        baseline_metrics["model_name"] = "Fallback ensemble"
        evaluation_frame["ml_projection"] = np.nan
        return np.full(len(feature_df), np.nan), evaluation_frame, baseline_metrics

    feature_columns = [
        "Brand",
        "Category",
        "Month Number",
        "quarter",
        "month_sin",
        "month_cos",
        "year_index",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_5",
        "lag_6",
        "rolling_3",
        "rolling_6",
        "same_month_3y_avg",
        "six_month_trend_feature",
    ]

    model_train = train_df.dropna(subset=["lag_1", "rolling_3"]).copy()
    evaluation_frame = model_train.groupby("Variant ID").tail(1).copy()
    training_frame = model_train.drop(index=evaluation_frame.index)
    if len(training_frame) < max(24, len(feature_df)):
        baseline_metrics["model_name"] = "Fallback ensemble"
        evaluation_frame["ml_projection"] = np.nan
        return np.full(len(feature_df), np.nan), evaluation_frame, baseline_metrics

    training_frame = training_frame.sort_values(["Month", "Variant ID"]).reset_index(drop=True)
    evaluation_frame = evaluation_frame.sort_values(["Month", "Variant ID"]).reset_index(drop=True)

    X_train = training_frame[feature_columns]
    y_train = training_frame["Units Sold"]
    X_valid = evaluation_frame[feature_columns]
    y_valid = evaluation_frame["Units Sold"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["Brand", "Category"],
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                [
                    "Month Number",
                    "quarter",
                    "month_sin",
                    "month_cos",
                    "year_index",
                    "lag_1",
                    "lag_2",
                    "lag_3",
                    "lag_4",
                    "lag_5",
                    "lag_6",
                    "rolling_3",
                    "rolling_6",
                    "same_month_3y_avg",
                    "six_month_trend_feature",
                ],
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=350,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    if len(X_valid) > 0:
        valid_pred = model.predict(X_valid)
        mae = float(mean_absolute_error(y_valid, valid_pred))
        r2 = float(r2_score(y_valid, valid_pred)) if len(y_valid) > 1 else 0.0
    else:
        mae = 0.0
        r2 = 0.0

    forecast_pred = model.predict(feature_df[feature_columns])
    evaluation_frame["ml_projection"] = model.predict(evaluation_frame[feature_columns]) if len(evaluation_frame) > 0 else np.nan
    metrics = {
        "model_name": "RandomForest monthly demand model",
        "mae": mae,
        "r2": r2,
    }
    return forecast_pred, evaluation_frame, metrics


def forecast_next_month(history_df: pd.DataFrame, sku_frame: pd.DataFrame) -> ForecastArtifacts:
    monthly_history = prepare_history_frame(history_df, sku_frame)
    complete_history = _complete_monthly_history(monthly_history, sku_frame)
    train_df = _build_training_frame(complete_history)
    feature_df, next_month = _build_next_month_feature_frame(monthly_history, sku_frame)
    ml_prediction, evaluation_frame, metrics = _train_monthly_model(train_df, feature_df)

    feature_df["Last 6M Trend Forecast"] = feature_df["six_month_trend_feature"].clip(lower=0)
    feature_df["Same Month 3Y Avg"] = feature_df["same_month_3y_avg"].fillna(feature_df["rolling_6"]).clip(lower=0)
    feature_df["ML Forecast"] = np.where(np.isnan(ml_prediction), feature_df["rolling_6"], np.clip(ml_prediction, 0, None))

    hybrid_forecasts: list[float] = []
    confidence_scores: list[float] = []
    suggested_reorder: list[float] = []
    fg_cover_days: list[float] = []
    forecast_growth: list[float] = []
    basis_labels: list[str] = []

    model_mae = float(metrics.get("mae", 0.0))
    mean_monthly_volume = max(float(monthly_history["Units Sold"].mean()), 1.0)

    for _, row in feature_df.iterrows():
        hybrid_units, basis_label = _combine_projection_components(
            ml_value=row["ML Forecast"],
            six_month_value=row["Last 6M Trend Forecast"],
            seasonal_value=row["Same Month 3Y Avg"],
            fallback_units=max(row["Current Per Day"] * row["Days in Next Month"], 0),
        )
        hybrid_forecasts.append(hybrid_units)
        basis_labels.append(basis_label)

        last_month_sales = max(float(row["Last Month Sales"]), 1.0)
        forecast_growth.append(((hybrid_units - last_month_sales) / last_month_sales) if last_month_sales else 0.0)

        daily_forecast = hybrid_units / max(float(row["Days in Next Month"]), 1.0)
        fg_cover = float(row["FG"]) / daily_forecast if daily_forecast > 0 else 0.0
        fg_cover_days.append(fg_cover)

        target_next_month_inventory = hybrid_units * 1.1
        reorder_qty = max(target_next_month_inventory - (float(row["FG"]) + float(row["Current"]) + float(row["Intransit"]) + float(row["Booked"])), 0)
        suggested_reorder.append(reorder_qty)

        history_factor = min(float(row["history_month_count"]) / 36, 1.0)
        seasonality_factor = 1.0 if pd.notna(row["same_month_3y_avg"]) and row["same_month_3y_avg"] > 0 else 0.55
        stability_factor = float(max(0.2, 1 - (model_mae / mean_monthly_volume))) if SKLEARN_AVAILABLE else 0.5
        confidence = float(np.clip((0.4 * history_factor) + (0.3 * seasonality_factor) + (0.3 * stability_factor), 0.1, 0.99))
        confidence_scores.append(confidence)

    feature_df["AI Hybrid Forecast"] = hybrid_forecasts
    feature_df["AI Forecast Per Day"] = feature_df["AI Hybrid Forecast"] / feature_df["Days in Next Month"]
    feature_df["Forecast Confidence"] = confidence_scores
    feature_df["FG Cover Next Month (Days)"] = fg_cover_days
    feature_df["Suggested Reorder Qty"] = suggested_reorder
    feature_df["Forecast Growth %"] = forecast_growth
    feature_df["Forecast Basis"] = basis_labels
    feature_df["Next Month Label"] = next_month.strftime("%b %Y")
    feature_df["Forecast Risk"] = np.select(
        [
            feature_df["FG Cover Next Month (Days)"] < 10,
            feature_df["Forecast Confidence"] < 0.5,
            feature_df["Forecast Growth %"] > 0.25,
        ],
        [
            "FG risk",
            "Low confidence",
            "Demand spike",
        ],
        default="Stable",
    )

    previous_projection_lookup = (
        evaluation_frame[["Variant ID", "Month", "Units Sold", "six_month_trend_feature", "same_month_3y_avg", "ml_projection"]]
        .drop_duplicates(subset=["Variant ID"])
        .copy()
    )
    previous_projection_lookup["Previous Projection"] = previous_projection_lookup.apply(
        lambda row: _combine_projection_components(
            ml_value=row["ml_projection"],
            six_month_value=row["six_month_trend_feature"],
            seasonal_value=row["same_month_3y_avg"],
            fallback_units=float(row["Units Sold"]),
        )[0],
        axis=1,
    )
    previous_projection_lookup["Previous Month"] = previous_projection_lookup["Month"].dt.strftime("%b %Y")
    previous_projection_lookup["Actual Sold"] = previous_projection_lookup["Units Sold"]
    previous_projection_lookup["Variance"] = previous_projection_lookup["Actual Sold"] - previous_projection_lookup["Previous Projection"]
    previous_projection_lookup["Accuracy %"] = np.where(
        previous_projection_lookup["Actual Sold"] > 0,
        1 - (
            np.abs(previous_projection_lookup["Actual Sold"] - previous_projection_lookup["Previous Projection"])
            / previous_projection_lookup["Actual Sold"]
        ),
        0,
    )
    previous_projection_lookup["Accuracy %"] = previous_projection_lookup["Accuracy %"].clip(lower=0, upper=1)
    feature_df = feature_df.merge(
        previous_projection_lookup[
            ["Variant ID", "Previous Month", "Previous Projection", "Actual Sold", "Variance", "Accuracy %"]
        ],
        on="Variant ID",
        how="left",
    )
    feature_df["Same Month Last Year Sold"] = feature_df["same_month_last_year_sold"]
    feature_df["Next Month Projection"] = feature_df["AI Hybrid Forecast"]
    feature_df["YoY Change"] = feature_df["Next Month Projection"] - feature_df["Same Month Last Year Sold"]
    feature_df["YoY Change %"] = np.where(
        feature_df["Same Month Last Year Sold"] > 0,
        feature_df["YoY Change"] / feature_df["Same Month Last Year Sold"],
        np.nan,
    )
    feature_df["Daily Projection"] = feature_df["AI Forecast Per Day"]

    forecast_columns = [
        "Variant ID",
        "Name",
        "Brand",
        "Category",
        "Previous Month",
        "Previous Projection",
        "Actual Sold",
        "Variance",
        "Accuracy %",
        "Same Month Last Year Sold",
        "YoY Change",
        "YoY Change %",
        "Next Month Label",
        "Next Month Projection",
        "Daily Projection",
        "Forecast Confidence",
        "Forecast Basis",
        "AI Hybrid Forecast",
        "AI Forecast Per Day",
    ]
    forecast_df = feature_df[forecast_columns].sort_values(
        by=["Next Month Projection"],
        ascending=[False],
    ).reset_index(drop=True)

    metrics.update(
        {
            "next_month": next_month.strftime("%B %Y"),
            "history_months": int(monthly_history["Month"].nunique()),
            "forecast_method": "Hybrid 6M + same-month 3Y + ML ensemble",
            "previous_month": previous_projection_lookup["Previous Month"].mode().iloc[0]
            if not previous_projection_lookup.empty
            else "",
        }
    )
    return ForecastArtifacts(forecast=forecast_df, metrics=metrics, monthly_history=monthly_history)


def generate_ai_forecast_insights(forecast_df: pd.DataFrame) -> list[str]:
    if forecast_df.empty:
        return ["No forecast insights available because the forecast table is empty."]

    insights: list[str] = []
    total_previous_projection = float(forecast_df["Previous Projection"].fillna(0).sum())
    total_actual = float(forecast_df["Actual Sold"].fillna(0).sum())
    total_next_projection = float(forecast_df["Next Month Projection"].fillna(0).sum())
    total_same_month_last_year = float(forecast_df["Same Month Last Year Sold"].fillna(0).sum())
    total_yoy_change = total_next_projection - total_same_month_last_year
    total_yoy_pct = (total_yoy_change / total_same_month_last_year) if total_same_month_last_year > 0 else float("nan")
    avg_accuracy = float(forecast_df["Accuracy %"].fillna(0).mean())
    top_sku = forecast_df.sort_values("Next Month Projection", ascending=False).iloc[0]
    biggest_gap = forecast_df.reindex(forecast_df["Variance"].abs().sort_values(ascending=False).index).iloc[0]

    insights.append(
        f"For {biggest_gap['Previous Month']}, total projection was {total_previous_projection:,.0f} units and actual sales were {total_actual:,.0f} units."
    )
    insights.append(
        f"Average projection accuracy across SKUs was {avg_accuracy:.0%} for the last closed month."
    )
    insights.append(
        f"{top_sku['Variant ID']} has the highest next-month projection at {top_sku['Next Month Projection']:,.0f} units."
    )
    if pd.notna(total_yoy_pct):
        direction = "up" if total_yoy_change >= 0 else "down"
        insights.append(
            f"Compared with the same month last year, next-month total demand is projected {direction} by {abs(total_yoy_change):,.0f} units ({abs(total_yoy_pct):.1%})."
        )
    else:
        insights.append("Same-month-last-year comparison is not available for the current upload.")
    insights.append(
        f"The biggest last-month variance was on {biggest_gap['Variant ID']}, with actual sales differing from projection by {biggest_gap['Variance']:,.0f} units."
    )
    insights.append(
        f"Total projected demand for next month is {total_next_projection:,.0f} units across the selected SKUs."
    )
    return insights


def build_history_template(sku_frame: pd.DataFrame, months_back: int = 36) -> pd.DataFrame:
    last_closed_month = pd.Timestamp.today().normalize().replace(day=1) - pd.offsets.MonthBegin(1)
    recent_months = pd.date_range(end=last_closed_month, periods=6, freq="MS")
    forecast_month = last_closed_month + pd.offsets.MonthBegin(1)

    template_rows: list[dict[str, object]] = []
    for _, row in sku_frame[["Variant ID", "Name", "Brand", "Category"]].drop_duplicates().iterrows():
        template_row: dict[str, object] = {
            "Variant ID": row["Variant ID"],
            "Name": row["Name"],
            "Brand": row["Brand"],
            "Category": row["Category"],
        }
        for month in recent_months:
            template_row[month.strftime("%Y-%m")] = ""
        template_row["Same Month Last Year"] = ""
        template_row["Same Month 2 Years Ago"] = ""
        template_row["Same Month 3 Years Ago"] = ""
        template_rows.append(template_row)

    template_df = pd.DataFrame(template_rows)
    template_df.attrs["forecast_month"] = forecast_month.strftime("%b %Y")
    return template_df


def build_production_projection(
    forecast_df: pd.DataFrame,
    sku_frame: pd.DataFrame,
    safety_stock_days: int = 7,
) -> pd.DataFrame:
    supply_columns = [
        "Variant ID",
        "Name",
        "Brand",
        "Category",
        "ABC-Cat",
        "Case Size",
        "Current",
        "Intransit",
        "Booked",
        "Retail",
    ]
    supply_frame = sku_frame[supply_columns].drop_duplicates().copy()
    projection = forecast_df.merge(supply_frame, on=["Variant ID", "Name", "Brand", "Category"], how="left")

    projection["Case Size"] = pd.to_numeric(projection["Case Size"], errors="coerce").fillna(1).clip(lower=1)
    projection["Projected Monthly Demand"] = projection["AI Hybrid Forecast"].clip(lower=0)
    projection["Projected Daily Demand"] = projection["AI Forecast Per Day"].clip(lower=0)
    projection["Projected Demand Cases"] = np.ceil(
        projection["Projected Monthly Demand"] / projection["Case Size"]
    ).astype(int)
    projection["Available Network Stock"] = (
        projection["FG"].fillna(0)
        + projection["Current"].fillna(0)
        + projection["Intransit"].fillna(0)
    )
    projection["Safety Stock Units"] = projection["Projected Daily Demand"] * safety_stock_days
    projection["Gross Monthly Requirement"] = (
        projection["Projected Monthly Demand"] + projection["Safety Stock Units"]
    )
    projection["Net Production Requirement"] = (
        projection["Gross Monthly Requirement"] - projection["Available Network Stock"]
    ).clip(lower=0)
    projection["Suggested Production Cases"] = np.where(
        projection["Net Production Requirement"] > 0,
        np.ceil(projection["Net Production Requirement"] / projection["Case Size"]),
        0,
    ).astype(int)
    projection["Suggested Production Qty"] = (
        projection["Suggested Production Cases"] * projection["Case Size"]
    )
    projection["Expected Closing Stock"] = (
        projection["Available Network Stock"]
        + projection["Suggested Production Qty"]
        - projection["Projected Monthly Demand"]
    )
    projection["Expected Closing Cover (Days)"] = np.where(
        projection["Projected Daily Demand"] > 0,
        projection["Expected Closing Stock"] / projection["Projected Daily Demand"],
        0,
    )
    projection["Projected Production Gap %"] = np.where(
        projection["Gross Monthly Requirement"] > 0,
        projection["Net Production Requirement"] / projection["Gross Monthly Requirement"],
        0,
    )

    priority_score = (
        projection["ABC-Cat"].map({"A": 55, "B": 35, "C": 20}).fillna(15)
        + (projection["Projected Production Gap %"].clip(lower=0, upper=1) * 80)
        + ((15 - projection["FG Cover Next Month (Days)"].clip(upper=15)).clip(lower=0) * 4)
        + (projection["Forecast Growth %"].clip(lower=0, upper=0.5) * 100)
        + np.minimum(projection["Net Production Requirement"] / projection["Case Size"], 60)
        + np.where(projection["Forecast Confidence"] < 0.55, 12, 0)
    )
    projection["Production Priority Score"] = priority_score
    projection["Production Priority"] = np.select(
        [
            projection["Suggested Production Qty"] <= 0,
            (projection["Forecast Confidence"] < 0.5) & (projection["Suggested Production Qty"] > 0),
            (projection["FG Cover Next Month (Days)"] < 7)
            | ((projection["ABC-Cat"] == "A") & (projection["FG Cover Next Month (Days)"] < 10)),
            (projection["FG Cover Next Month (Days)"] < 12)
            | (projection["Projected Production Gap %"] > 0.35),
            projection["Suggested Production Qty"] > 0,
        ],
        [
            "No Action",
            "Review",
            "Critical",
            "High",
            "Planned",
        ],
        default="Monitor",
    )
    projection["Production Recommendation"] = np.select(
        [
            projection["Suggested Production Qty"] <= 0,
            projection["Production Priority"] == "Review",
            projection["Production Priority"] == "Critical",
            projection["Production Priority"] == "High",
            projection["Production Priority"] == "Planned",
        ],
        [
            "Current stock is enough for the month",
            "Review with planner because confidence is low",
            "Produce immediately and secure FG cover",
            "Schedule production early in the month",
            "Include in monthly production plan",
        ],
        default="Keep under review",
    )

    ordered_columns = [
        "Variant ID",
        "Name",
        "Brand",
        "Category",
        "ABC-Cat",
        "Next Month Label",
        "Projected Monthly Demand",
        "Projected Daily Demand",
        "Projected Demand Cases",
        "Available Network Stock",
        "FG",
        "Current",
        "Intransit",
        "Booked",
        "Safety Stock Units",
        "Gross Monthly Requirement",
        "Net Production Requirement",
        "Case Size",
        "Suggested Production Cases",
        "Suggested Production Qty",
        "Expected Closing Stock",
        "Expected Closing Cover (Days)",
        "Forecast Confidence",
        "Forecast Growth %",
        "FG Cover Next Month (Days)",
        "Forecast Basis",
        "Forecast Risk",
        "Production Priority",
        "Production Recommendation",
        "Production Priority Score",
    ]
    return projection[ordered_columns].sort_values(
        by=["Production Priority Score", "Suggested Production Qty"],
        ascending=[False, False],
    ).reset_index(drop=True)


def train_consumption_model(
    history_df: pd.DataFrame,
    sku_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    artifacts = forecast_next_month(history_df, sku_frame)
    forecast_input = artifacts.forecast[
        [
            "Variant ID",
            "Name",
            "Category",
            "Brand",
            "AI Forecast Per Day",
            "Forecast Confidence",
        ]
    ].rename(
        columns={
            "AI Forecast Per Day": "Predicted Per Day Consumption",
            "Forecast Confidence": "Confidence Score",
        }
    )
    return forecast_input, artifacts.metrics
