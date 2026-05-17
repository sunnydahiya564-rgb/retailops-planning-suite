from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Variant ID",
    "Name",
    "Brand",
    "Category",
    "ABC-Cat",
    "Current",
    "Intransit",
    "Booked",
    "Case Size",
    "FG",
    "Per Day",
    "PDA",
]


OPTIONAL_DEFAULTS = {
    "Retail": 0.0,
    "Historical Sales": 0.0,
    "Incoming Stock": 0.0,
    "Sales": 0.0,
    "Remarks": "",
    "Warehouse": "Main",
}

ABC_RANK = {"A": 0, "B": 1, "C": 2}

COLUMN_ALIASES = {
    "Variant ID (SKU)": "Variant ID",
    "variant id (sku)": "Variant ID",
    "variant id": "Variant ID",
    "sku": "Variant ID",
    "ABC Category": "ABC-Cat",
    "abc category": "ABC-Cat",
    "abc-cat": "ABC-Cat",
    "abc-cat ": "ABC-Cat",
    "Current Retail Inventory": "Current",
    "current retail inventory": "Current",
    "current": "Current",
    "Intransit Inventory": "Intransit",
    "intransit inventory": "Intransit",
    "intransit": "Intransit",
    "Booked Inventory": "Booked",
    "booked inventory": "Booked",
    "booked": "Booked",
    "FG Inventory (Main Warehouse)": "FG",
    "fg inventory (main warehouse)": "FG",
    "fg inventory": "FG",
    "fg": "FG",
    "Case size": "Case Size",
    "case size": "Case Size",
    "Per Day Consumption (Projection)": "Per Day",
    "per day consumption (projection)": "Per Day",
    "Per day (as per projection)": "Per Day",
    "per day (as per projection)": "Per Day",
    "per day": "Per Day",
    "Retail Inventory": "Retail",
    "retail inventory": "Retail",
    "retail": "Retail",
    "PDA": "PDA",
    "Brand": "Brand",
    "Category": "Category",
    "Name": "Name",
    "Remarks": "Remarks",
}


def prepare_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = normalize_column_names(df.copy())
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in clean_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    for column, default_value in OPTIONAL_DEFAULTS.items():
        if column not in clean_df.columns:
            clean_df[column] = default_value

    numeric_columns = [
        "Retail",
        "Current",
        "Intransit",
        "Booked",
        "FG",
        "Case Size",
        "Per Day",
        "PDA",
        "Historical Sales",
        "Incoming Stock",
        "Sales",
    ]

    for column in numeric_columns:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce").fillna(0)

    clean_df["Case Size"] = clean_df["Case Size"].clip(lower=1).round().astype(int)
    clean_df["Per Day"] = clean_df["Per Day"].clip(lower=0)
    clean_df["PDA"] = clean_df["PDA"].clip(lower=0)
    clean_df["Variant ID"] = clean_df["Variant ID"].astype(str)

    text_columns = ["Name", "Brand", "Category", "ABC-Cat", "Remarks", "Warehouse"]
    for column in text_columns:
        clean_df[column] = clean_df[column].fillna("Unknown").astype(str)

    return clean_df


def load_uploaded_table(uploaded_file, all_sheets: bool = False):
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")
    file_name = uploaded_file.name.lower()
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, sheet_name=None if all_sheets else 0)
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def dataframe_to_excel_bytes(sheet_name: str, df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.read()


def dataframes_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = str(sheet_name)[:31] or "Sheet1"
            df.to_excel(writer, index=False, sheet_name=safe_name)
    output.seek(0)
    return output.read()


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in df.columns:
        stripped = str(column).strip()
        alias = COLUMN_ALIASES.get(stripped, COLUMN_ALIASES.get(stripped.lower()))
        if alias:
            rename_map[column] = alias
    return df.rename(columns=rename_map)


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series:
    numerator_series = numerator if isinstance(numerator, pd.Series) else pd.Series([numerator])
    if isinstance(denominator, pd.Series):
        denominator_series = denominator.replace(0, np.nan)
    else:
        denominator_series = denominator if denominator != 0 else np.nan
    return numerator_series.div(denominator_series).replace([np.inf, -np.inf], np.nan).fillna(0)


def _priority_level(abc: str, doi: float, gap: float) -> str:
    if abc == "A" and doi < 5:
        return "Critical"
    if doi < 10 or gap > 0:
        return "High"
    if abc == "B" or doi < 20:
        return "Medium"
    return "Normal"


def _decision(move_qty: float, fg_status: str, overstock: bool, dead_stock: bool) -> str:
    if fg_status == "FG Shortage":
        return "Move partial, replenish FG"
    if overstock:
        return "Hold, monitor excess"
    if dead_stock:
        return "Do not move, review SKU"
    if move_qty > 0:
        return "Move stock"
    return "No movement required"


def calculate_inventory_plan(
    df: pd.DataFrame,
    selected_days: int,
    target_doi: int,
    low_doi_threshold: float,
    excess_doi_threshold: float,
    dead_stock_movement_threshold: float,
) -> pd.DataFrame:
    plan_df = prepare_input_frame(df)
    projection_qty = plan_df["Per Day"] * selected_days

    plan_df["Available Retail Inventory"] = (
        plan_df["Current"]
        + plan_df["Intransit"]
        + plan_df["Booked"]
        + plan_df["Incoming Stock"]
    )
    plan_df["Booked/Proj %"] = safe_divide(plan_df["Booked"], projection_qty)
    plan_df["Current DOI"] = safe_divide(plan_df["Available Retail Inventory"], plan_df["Per Day"])
    plan_df["1st Remaining"] = projection_qty - plan_df["Available Retail Inventory"]
    plan_df["Required for 20 days"] = plan_df["Per Day"] * target_doi
    plan_df["Required Gap"] = (
        plan_df["Required for 20 days"] - plan_df["Available Retail Inventory"]
    ).clip(lower=0)
    plan_df["Move"] = plan_df["Required Gap"].clip(lower=0)
    plan_df["Move in case"] = np.where(
        plan_df["Move"] > 0,
        np.ceil(plan_df["Move"] / plan_df["Case Size"]),
        0,
    ).astype(int)
    plan_df["Final Move Qty"] = plan_df["Move in case"] * plan_df["Case Size"]
    plan_df["Remaining according to case size"] = (
        plan_df["Required for 20 days"]
        - (plan_df["Available Retail Inventory"] + plan_df["Final Move Qty"])
    )
    plan_df["Suggested Partial Move"] = (
        np.floor(plan_df["FG"] / plan_df["Case Size"])
        * plan_df["Case Size"]
    )
    plan_df["FG Status"] = np.where(
        plan_df["FG"] < plan_df["Final Move Qty"],
        "FG Shortage",
        "OK",
    )
    plan_df["DOI after movement"] = safe_divide(
        plan_df["Available Retail Inventory"] + plan_df["Final Move Qty"],
        plan_df["Per Day"],
    )
    plan_df["FG DOI"] = safe_divide(plan_df["FG"], plan_df["Per Day"])
    plan_df["Projection Utilization %"] = safe_divide(
        plan_df["Booked"] + plan_df["Sales"], projection_qty
    )
    plan_df["Current DOI vs Target DOI"] = safe_divide(plan_df["Current DOI"], target_doi)
    plan_df["Overstock Alert"] = plan_df["Current DOI"] > excess_doi_threshold
    plan_df["Understock Alert"] = plan_df["Current DOI"] < low_doi_threshold
    plan_df["Dead Stock Flag"] = (
        (plan_df["Current DOI"] > excess_doi_threshold)
        & (plan_df["PDA"] <= dead_stock_movement_threshold)
    )
    plan_df["Case Break Loss"] = (plan_df["Final Move Qty"] - plan_df["Move"]).clip(lower=0)
    plan_df["Movement Efficiency %"] = safe_divide(
        plan_df["Move"], plan_df["Final Move Qty"]
    )
    plan_df["Low DOI Threshold"] = low_doi_threshold
    plan_df["Excess DOI Threshold"] = excess_doi_threshold
    plan_df["Priority Level"] = [
        _priority_level(abc, doi, gap)
        for abc, doi, gap in zip(
            plan_df["ABC-Cat"],
            plan_df["Current DOI"],
            plan_df["Required Gap"],
        )
    ]
    plan_df["Decision"] = [
        _decision(move_qty, fg_status, overstock, dead_stock)
        for move_qty, fg_status, overstock, dead_stock in zip(
            plan_df["Final Move Qty"],
            plan_df["FG Status"],
            plan_df["Overstock Alert"],
            plan_df["Dead Stock Flag"],
        )
    ]
    plan_df["Risk Flags"] = plan_df.apply(_build_risk_flags, axis=1)
    plan_df["Exception Flag"] = np.where(
        (plan_df["FG Status"] == "FG Shortage")
        | (plan_df["Understock Alert"])
        | (plan_df["Move"] > 0),
        "Action Required",
        "Monitor",
    )

    abc_sort = plan_df["ABC-Cat"].map(ABC_RANK).fillna(3)
    plan_df = plan_df.assign(_abc_sort=abc_sort)
    plan_df = plan_df.sort_values(
        by=["_abc_sort", "Current DOI", "Required Gap"],
        ascending=[True, True, False],
    ).drop(columns="_abc_sort")

    return plan_df.reset_index(drop=True)


def _build_risk_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if row["FG Status"] == "FG Shortage":
        flags.append("FG shortage")
    if row["Understock Alert"]:
        flags.append("Understock")
    if row["Overstock Alert"]:
        flags.append("Overstock")
    if row["Dead Stock Flag"]:
        flags.append("Dead stock")
    if row["FG"] <= 0:
        flags.append("Zero FG")
    return ", ".join(flags) if flags else "Healthy"


def summarize_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "total_skus": 0,
            "total_move_qty": 0.0,
            "fg_shortage_count": 0,
            "avg_doi": 0.0,
            "total_required_qty": 0.0,
            "action_sku_count": 0,
            "total_case_break_loss": 0.0,
            "avg_movement_efficiency": 0.0,
        }
    return {
        "total_skus": len(df),
        "total_move_qty": float(df["Final Move Qty"].sum()),
        "fg_shortage_count": int((df["FG Status"] == "FG Shortage").sum()),
        "avg_doi": float(df["Current DOI"].mean()),
        "total_required_qty": float(df["Required for 20 days"].sum()),
        "action_sku_count": int((df["Exception Flag"] == "Action Required").sum()),
        "total_case_break_loss": float(df["Case Break Loss"].sum()),
        "avg_movement_efficiency": float(df["Movement Efficiency %"].replace([np.inf, -np.inf], np.nan).fillna(0).mean()),
    }


def create_priority_score(df: pd.DataFrame) -> pd.Series:
    abc_component = df["ABC-Cat"].map({"A": 100, "B": 60, "C": 30}).fillna(10)
    doi_component = (30 - df["Current DOI"].clip(upper=30)).clip(lower=0) * 2
    gap_component = df["Required Gap"].clip(lower=0)
    shortage_component = np.where(df["FG Status"].eq("FG Shortage"), 25, 0)
    return abc_component + doi_component + gap_component + shortage_component


def apply_filters(
    df: pd.DataFrame,
    categories: list[str],
    brands: list[str],
    abc_categories: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    if brands:
        filtered = filtered[filtered["Brand"].isin(brands)]
    if abc_categories:
        filtered = filtered[filtered["ABC-Cat"].isin(abc_categories)]
    return filtered.reset_index(drop=True)


def build_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    export_columns = [
        "Variant ID",
        "Name",
        "Brand",
        "Category",
        "ABC-Cat",
        "Retail",
        "Current",
        "Intransit",
        "Booked",
        "Booked/Proj %",
        "Current DOI",
        "1st Remaining",
        "Required for 20 days",
        "Per Day",
        "PDA",
        "Move",
        "Case Size",
        "FG",
        "Remaining according to case size",
        "Move in case",
        "Final Move Qty",
        "Suggested Partial Move",
        "DOI after movement",
        "FG Status",
        "Priority Level",
        "Decision",
        "Risk Flags",
        "Remarks",
        "SKU Ranking Score",
    ]
    return df[export_columns].copy()


def build_totals_row(df: pd.DataFrame, selected_days: int = 20) -> pd.DataFrame:
    totals = {
        "Variant ID": "TOTAL",
        "Name": "",
        "Brand": "",
        "Category": "",
        "ABC-Cat": "",
        "Retail": df["Retail"].sum(),
        "Current": df["Current"].sum(),
        "Intransit": df["Intransit"].sum(),
        "Booked": df["Booked"].sum(),
        "Booked/Proj %": safe_divide(df["Booked"].sum(), (df["Per Day"].sum() * selected_days)).iloc[0],
        "Current DOI": df["Current DOI"].mean() if not df.empty else 0,
        "1st Remaining": df["1st Remaining"].sum(),
        "Required for 20 days": df["Required for 20 days"].sum(),
        "Per Day": df["Per Day"].sum(),
        "PDA": df["PDA"].sum(),
        "Move": df["Move"].sum(),
        "Case Size": "",
        "FG": df["FG"].sum(),
        "Remaining according to case size": df["Remaining according to case size"].sum(),
        "Move in case": df["Move in case"].sum(),
        "Final Move Qty": df["Final Move Qty"].sum(),
        "Suggested Partial Move": df["Suggested Partial Move"].sum(),
        "DOI after movement": df["DOI after movement"].mean() if not df.empty else 0,
        "FG Status": "",
        "Priority Level": "",
        "Decision": "",
        "Risk Flags": "",
        "Remarks": "",
        "SKU Ranking Score": df["SKU Ranking Score"].sum() if "SKU Ranking Score" in df.columns else 0,
    }
    return pd.DataFrame([totals])


def apply_forecast_override(base_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    override_df = base_df.copy()
    if forecast_df.empty or "Variant ID" not in forecast_df.columns or "AI Forecast Per Day" not in forecast_df.columns:
        return override_df

    forecast_lookup = forecast_df[["Variant ID", "AI Forecast Per Day", "Forecast Confidence"]].drop_duplicates()
    override_df = override_df.merge(forecast_lookup, on="Variant ID", how="left")
    override_df["Per Day"] = np.where(
        override_df["AI Forecast Per Day"].notna(),
        override_df["AI Forecast Per Day"],
        override_df["Per Day"],
    )
    override_df["PDA"] = np.where(
        override_df["AI Forecast Per Day"].notna(),
        override_df["AI Forecast Per Day"],
        override_df["PDA"],
    )
    override_df["Forecast Confidence"] = override_df["Forecast Confidence"].fillna(0)
    return override_df.drop(columns=["AI Forecast Per Day"])


def summarize_alerts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {
            "critical": 0,
            "fg_shortage": 0,
            "understock": 0,
            "overstock": 0,
            "dead_stock": 0,
            "review": 0,
        }
    return {
        "critical": int((df["Priority Level"] == "Critical").sum()),
        "fg_shortage": int((df["FG Status"] == "FG Shortage").sum()),
        "understock": int(df["Understock Alert"].sum()),
        "overstock": int(df["Overstock Alert"].sum()),
        "dead_stock": int(df["Dead Stock Flag"].sum()),
        "review": int((df["Decision"] != "No movement required").sum()),
    }


def build_action_queue(df: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Severity",
                "Variant ID",
                "Name",
                "Brand",
                "Category",
                "Current DOI",
                "Final Move Qty",
                "Suggested Partial Move",
                "FG Status",
                "Decision",
                "Priority Level",
            ]
        )

    queue = df.copy()
    queue["Severity"] = np.select(
        [
            queue["FG Status"].eq("FG Shortage") | queue["Priority Level"].eq("Critical"),
            queue["Understock Alert"] | (queue["Final Move Qty"] > 0),
            queue["Overstock Alert"] | queue["Dead Stock Flag"],
        ],
        [
            "Critical",
            "High",
            "Medium",
        ],
        default="Monitor",
    )
    severity_rank = queue["Severity"].map({"Critical": 0, "High": 1, "Medium": 2, "Monitor": 3}).fillna(4)
    score_rank = queue["SKU Ranking Score"] if "SKU Ranking Score" in queue.columns else create_priority_score(queue)
    queue = queue.assign(_severity_rank=severity_rank, _score_rank=score_rank)
    queue = queue.sort_values(
        by=["_severity_rank", "_score_rank", "Current DOI"],
        ascending=[True, False, True],
    )
    queue = queue.head(limit)
    return queue[
        [
            "Severity",
            "Variant ID",
            "Name",
            "Brand",
            "Category",
            "Current DOI",
            "Final Move Qty",
            "Suggested Partial Move",
            "FG Status",
            "Decision",
            "Priority Level",
        ]
    ].reset_index(drop=True)


def build_dimension_summary(
    df: pd.DataFrame,
    dimension: str,
    metric: str,
    top_n: int = 12,
) -> pd.DataFrame:
    if df.empty or dimension not in df.columns or metric not in df.columns:
        return pd.DataFrame(columns=[dimension, metric])
    summary = (
        df.groupby(dimension, dropna=False)[metric]
        .sum()
        .reset_index()
        .sort_values(metric, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return summary


def build_scenario_snapshot(df: pd.DataFrame, label: str) -> dict[str, float | str]:
    metrics = summarize_metrics(df)
    metrics.update(
        {
            "label": label,
            "top_sku": str(df.iloc[0]["Variant ID"]) if not df.empty else "-",
            "top_move_qty": float(df["Final Move Qty"].max()) if not df.empty else 0.0,
        }
    )
    return metrics


def generate_ai_planning_summary(plan_df: pd.DataFrame) -> list[str]:
    if plan_df.empty:
        return ["No planning insights available because the filtered table is empty."]

    insights: list[str] = []
    top_move = plan_df.sort_values("Final Move Qty", ascending=False).iloc[0]
    top_doi_risk = plan_df.sort_values("Current DOI", ascending=True).iloc[0]
    fg_shortages = int((plan_df["FG Status"] == "FG Shortage").sum())
    case_break_loss = float(plan_df["Case Break Loss"].sum())
    total_move = float(plan_df["Final Move Qty"].sum())

    insights.append(
        f"{top_move['Variant ID']} is the highest movement priority at {top_move['Final Move Qty']:,.0f} units, driven by a gap of {top_move['Required Gap']:,.0f} units."
    )
    insights.append(
        f"{top_doi_risk['Variant ID']} has the lowest current DOI at {top_doi_risk['Current DOI']:.1f} days, so it is the most urgent retail cover risk."
    )
    insights.append(
        f"There are {fg_shortages} SKUs with FG shortages against the current movement plan."
    )
    insights.append(
        f"Total planned movement is {total_move:,.0f} units, with {case_break_loss:,.0f} units of case-break loss created by case-size rounding."
    )
    if "Forecast Confidence" in plan_df.columns and (plan_df["Forecast Confidence"] > 0).any():
        low_confidence = int((plan_df["Forecast Confidence"] < 0.55).sum())
        insights.append(
            f"{low_confidence} SKUs are being planned with low-confidence AI demand inputs and may need planner review."
        )
    return insights
