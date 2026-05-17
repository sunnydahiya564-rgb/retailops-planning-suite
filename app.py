from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from model import generate_demo_history, train_consumption_model
from ui_components import (
    APP_NAME,
    PALETTE,
    PLOTLY_CONFIG,
    inject_global_styles,
    render_badges,
    render_metric_cards,
    render_page_header,
    render_panel,
    render_section_header,
    style_figure,
)
from utils import (
    REQUIRED_COLUMNS,
    apply_filters,
    apply_forecast_override,
    build_action_queue,
    build_dimension_summary,
    build_export_frame,
    build_scenario_snapshot,
    build_totals_row,
    calculate_inventory_plan,
    create_priority_score,
    dataframe_to_excel_bytes,
    generate_ai_planning_summary,
    load_uploaded_table,
    prepare_input_frame,
    summarize_alerts,
    summarize_metrics,
)


st.set_page_config(
    page_title=f"{APP_NAME} | Inventory Planning",
    page_icon="📦",
    layout="wide",
)


DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
PRIMARY_COLOR = PALETTE["accent"]
SUCCESS_COLOR = PALETTE["success"]
WARNING_COLOR = PALETTE["warning"]
DANGER_COLOR = PALETTE["danger"]


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH)


@st.cache_data(show_spinner=False)
def build_demand_preview(base_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | str]]:
    history = generate_demo_history(base_df)
    return train_consumption_model(history, base_df)


def _format_delta(
    current: float,
    baseline: float | None,
    *,
    decimals: int = 0,
    suffix: str = "",
    lower_is_better: bool = False,
    neutral: bool = False,
) -> tuple[str, str]:
    if baseline is None:
        return "Pin a benchmark to compare", PALETTE["muted"]
    delta = current - baseline
    if abs(delta) < 1e-9:
        return "Matches benchmark", PRIMARY_COLOR
    if neutral:
        color = PRIMARY_COLOR
    else:
        improved = delta < 0 if lower_is_better else delta > 0
        color = SUCCESS_COLOR if improved else DANGER_COLOR
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):,.{decimals}f}{suffix} vs benchmark", color


def render_workspace_status(
    source_name: str,
    metrics: dict[str, float],
    filtered_count: int,
    total_count: int,
    selected_days: int,
    target_doi: int,
    demand_uplift_pct: int,
    forecast_active: bool,
    benchmark_label: str | None,
) -> None:
    badges = [
        (f"Source: {source_name}", PALETTE["accent_soft"], PRIMARY_COLOR),
        (f"In focus: {filtered_count}/{total_count} SKUs", PALETTE["surface"], PALETTE["ink"]),
        (f"Projection days: {selected_days}", PALETTE["surface"], PALETTE["ink"]),
        (f"Target DOI: {target_doi} days", PALETTE["surface"], PALETTE["ink"]),
        (f"Action queue: {int(metrics['action_sku_count'])} SKUs", PALETTE["warning_soft"], "#7A4B00"),
    ]
    if demand_uplift_pct != 0:
        badges.append((f"Demand uplift: {demand_uplift_pct:+d}%", PALETTE["warning_soft"], "#7A4B00"))
    badges.append(
        (
            "Projection override live" if forecast_active else "Using planning sheet demand",
            PALETTE["success_soft"] if forecast_active else PALETTE["surface"],
            SUCCESS_COLOR if forecast_active else PALETTE["ink"],
        )
    )
    if benchmark_label:
        badges.append((f"Benchmark: {benchmark_label}", PALETTE["accent_soft"], PRIMARY_COLOR))
    render_badges(badges)


def render_kpi_cards(metrics: dict[str, float], benchmark: dict[str, float | str] | None) -> None:
    benchmark_total_move = float(benchmark["total_move_qty"]) if benchmark else None
    benchmark_action_skus = float(benchmark["action_sku_count"]) if benchmark else None
    benchmark_shortages = float(benchmark["fg_shortage_count"]) if benchmark else None
    benchmark_avg_doi = float(benchmark["avg_doi"]) if benchmark else None
    benchmark_required = float(benchmark["total_required_qty"]) if benchmark else None
    benchmark_case_loss = float(benchmark["total_case_break_loss"]) if benchmark else None

    total_move_delta, total_move_color = _format_delta(
        float(metrics["total_move_qty"]),
        benchmark_total_move,
        neutral=True,
    )
    action_delta, action_delta_color = _format_delta(
        float(metrics["action_sku_count"]),
        benchmark_action_skus,
        lower_is_better=True,
    )
    shortage_delta, shortage_delta_color = _format_delta(
        float(metrics["fg_shortage_count"]),
        benchmark_shortages,
        lower_is_better=True,
    )
    doi_delta, doi_delta_color = _format_delta(
        float(metrics["avg_doi"]),
        benchmark_avg_doi,
        decimals=1,
        suffix=" days",
    )
    required_delta, required_delta_color = _format_delta(
        float(metrics["total_required_qty"]),
        benchmark_required,
        neutral=True,
    )
    case_loss_delta, case_loss_delta_color = _format_delta(
        float(metrics["total_case_break_loss"]),
        benchmark_case_loss,
        lower_is_better=True,
    )

    render_metric_cards(
        [
            {
                "label": "SKUs in scope",
                "value": f"{int(metrics['total_skus'])}",
                "delta": f"{int(metrics['action_sku_count'])} need action",
                "delta_color": PRIMARY_COLOR,
                "caption": "Focused result set after filters and scenario edits.",
            },
            {
                "label": "Total move qty",
                "value": f"{metrics['total_move_qty']:,.0f}",
                "delta": total_move_delta,
                "delta_color": total_move_color,
                "caption": "Case-adjusted quantity recommended for movement.",
            },
            {
                "label": "Required qty",
                "value": f"{metrics['total_required_qty']:,.0f}",
                "delta": required_delta,
                "delta_color": required_delta_color,
                "caption": "Total retail requirement against the current DOI target.",
            },
            {
                "label": "FG shortages",
                "value": f"{int(metrics['fg_shortage_count'])}",
                "delta": shortage_delta,
                "delta_color": shortage_delta_color,
                "caption": "SKUs where FG cannot fully support the planned move.",
            },
            {
                "label": "Average DOI",
                "value": f"{metrics['avg_doi']:.1f} days",
                "delta": doi_delta,
                "delta_color": doi_delta_color,
                "caption": "Retail cover before movement for the selected scope.",
            },
            {
                "label": "Case break loss",
                "value": f"{metrics['total_case_break_loss']:,.0f}",
                "delta": case_loss_delta,
                "delta_color": case_loss_delta_color,
                "caption": f"Average movement efficiency {metrics['avg_movement_efficiency']:.0%}.",
            },
        ]
    )
    if benchmark:
        st.caption(
            f"Benchmark pinned as `{benchmark.get('label', 'scenario')}`. Top benchmark SKU: `{benchmark.get('top_sku', '-')}`."
        )
    else:
        st.caption("Use `Pin current view as benchmark` to compare new scenarios against a saved baseline.")


def render_alert_center(df: pd.DataFrame) -> None:
    render_section_header(
        "Alert Center",
        caption="Actionable exceptions grouped for planner review and same-day follow-up.",
    )
    alerts = summarize_alerts(df)
    left_col, right_col = st.columns([1.05, 1.95], gap="large")

    with left_col:
        render_metric_cards(
            [
                {
                    "label": "Critical alerts",
                    "value": f"{alerts['critical']}",
                    "delta": f"{alerts['fg_shortage']} FG shortages",
                    "delta_color": DANGER_COLOR,
                    "caption": "Critical SKUs combine low cover and business priority.",
                },
                {
                    "label": "Understock",
                    "value": f"{alerts['understock']}",
                    "delta": f"{alerts['overstock']} overstock",
                    "delta_color": WARNING_COLOR,
                    "caption": f"{alerts['dead_stock']} dead-stock flags currently open.",
                },
            ]
        )
        render_badges(
            [
                ("FG shortage", PALETTE["danger_soft"], DANGER_COLOR),
                ("Low DOI", PALETTE["warning_soft"], "#7A4B00"),
                ("Overstock", PALETTE["accent_soft"], PRIMARY_COLOR),
                ("Dead stock", PALETTE["surface"], PALETTE["ink"]),
            ]
        )

    with right_col:
        action_queue = build_action_queue(df)
        st.dataframe(
            action_queue,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Current DOI": st.column_config.NumberColumn("Current DOI", format="%.1f"),
                "Final Move Qty": st.column_config.NumberColumn("Final Move Qty", format="%.0f"),
                "Suggested Partial Move": st.column_config.NumberColumn("Suggested Partial Move", format="%.0f"),
            },
        )


def render_analytics(df: pd.DataFrame) -> None:
    render_section_header(
        "Decision Analytics",
        caption="Interactive drilldowns that explain where movement, gap, and stock risk are concentrated.",
    )
    control_col1, control_col2, control_col3 = st.columns([1, 1, 1.2])
    dimension = control_col1.selectbox("Drilldown by", ["Category", "Brand", "ABC-Cat"])
    metric_label = control_col2.selectbox(
        "Primary metric",
        ["Final Move Qty", "Required Gap", "Available Retail Inventory"],
    )
    top_n = control_col3.slider("Show top groups", min_value=5, max_value=20, value=10)

    dimension_summary = build_dimension_summary(df, dimension=dimension, metric=metric_label, top_n=top_n)

    chart_col1, chart_col2 = st.columns(2, gap="large")

    bar_fig = px.bar(
        dimension_summary.sort_values(metric_label, ascending=True),
        x=metric_label,
        y=dimension,
        orientation="h",
        color=metric_label,
        color_continuous_scale=["#DCE9F8", PRIMARY_COLOR],
        title=f"{dimension} by {metric_label}",
        text_auto=".2s",
    )
    bar_fig.update_traces(hovertemplate=f"{dimension}: %{{y}}<br>{metric_label}: %{{x:,.0f}}<extra></extra>")
    chart_col1.plotly_chart(style_figure(bar_fig, height=380), use_container_width=True, config=PLOTLY_CONFIG)

    bubble_fig = px.scatter(
        df,
        x="Current DOI",
        y="Required Gap",
        size="Final Move Qty",
        color="Priority Level",
        hover_name="Variant ID",
        hover_data={
            "Brand": True,
            "Category": True,
            "FG Status": True,
            "Decision": True,
            "Final Move Qty": ":,.0f",
            "Current DOI": ":.1f",
            "Required Gap": ":,.0f",
        },
        color_discrete_map={
            "Critical": DANGER_COLOR,
            "High": WARNING_COLOR,
            "Medium": PRIMARY_COLOR,
            "Normal": SUCCESS_COLOR,
        },
        title="Coverage vs Gap Map",
    )
    bubble_fig.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.35)")))
    chart_col2.plotly_chart(style_figure(bubble_fig, height=380), use_container_width=True, config=PLOTLY_CONFIG)

    lower_col1, lower_col2 = st.columns(2, gap="large")

    treemap_frame = df.nlargest(min(len(df), 30), "SKU Ranking Score")
    treemap_fig = px.treemap(
        treemap_frame,
        path=[px.Constant("Network"), "Category", "Brand", "Variant ID"],
        values="Final Move Qty",
        color="Current DOI",
        color_continuous_scale=["#FADBD8", "#FFF2CF", "#D9F2E7"],
        title="Move Qty Concentration by Category and Brand",
    )
    lower_col1.plotly_chart(style_figure(treemap_fig, height=430, show_legend=False), use_container_width=True, config=PLOTLY_CONFIG)

    decision_mix = (
        df.groupby("Decision", dropna=False)["Final Move Qty"]
        .sum()
        .reset_index()
        .sort_values("Final Move Qty", ascending=False)
    )
    donut_fig = px.pie(
        decision_mix,
        names="Decision",
        values="Final Move Qty",
        hole=0.62,
        color="Decision",
        color_discrete_sequence=[PRIMARY_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR, "#8595AE"],
        title="Movement Mix by Decision",
    )
    lower_col2.plotly_chart(style_figure(donut_fig, height=430), use_container_width=True, config=PLOTLY_CONFIG)


def render_demand_intelligence(base_df: pd.DataFrame) -> None:
    render_section_header(
        "Demand Signal",
        caption="Operational view of the next demand picture feeding the planning workspace.",
    )
    forecast_override = st.session_state.get("forecast_override")
    forecast_label = st.session_state.get("forecast_label", "")

    if isinstance(forecast_override, pd.DataFrame) and not forecast_override.empty:
        top_projection = forecast_override.nlargest(12, "Next Month Projection")
        render_metric_cards(
            [
                {
                    "label": "Projection month",
                    "value": str(forecast_label or top_projection["Next Month Label"].iloc[0]),
                    "delta": f"{len(top_projection)} top SKUs shown",
                    "delta_color": PRIMARY_COLOR,
                    "caption": "Saved from the Demand Projection page and ready to use in planning.",
                },
                {
                    "label": "Projected demand",
                    "value": f"{forecast_override['Next Month Projection'].sum():,.0f}",
                    "delta": f"{forecast_override['Forecast Confidence'].mean():.0%} average confidence",
                    "delta_color": SUCCESS_COLOR,
                    "caption": "Monthly projected units across the saved demand scope.",
                },
                {
                    "label": "Projected daily rate",
                    "value": f"{forecast_override['Daily Projection'].mean():,.1f}",
                    "delta": f"{forecast_override['Variant ID'].nunique()} projected SKUs",
                    "delta_color": PRIMARY_COLOR,
                    "caption": "Average daily demand carried back into the planning page.",
                },
            ]
        )
        projection_fig = px.bar(
            top_projection.sort_values("Next Month Projection", ascending=True),
            x="Next Month Projection",
            y="Variant ID",
            orientation="h",
            color="Forecast Confidence",
            color_continuous_scale=["#FADBD8", "#FFF2CF", "#D9F2E7"],
            title="Top Projected SKUs",
        )
        st.plotly_chart(style_figure(projection_fig, height=380), use_container_width=True, config=PLOTLY_CONFIG)
        return

    with st.spinner("Building demand preview..."):
        forecast_frame, model_metrics = build_demand_preview(base_df)
    st.caption("No saved next-month projection is active yet. The preview below uses generated history so planners can still review demand shape.")
    render_metric_cards(
        [
            {
                "label": "Model",
                "value": str(model_metrics["model_name"]),
                "delta": f"Validation MAE {float(model_metrics['mae']):.2f}",
                "delta_color": PRIMARY_COLOR,
                "caption": f"Validation R2 {float(model_metrics['r2']):.2f}",
            },
            {
                "label": "Avg predicted per day",
                "value": f"{forecast_frame['Predicted Per Day Consumption'].mean():,.1f}",
                "delta": f"{forecast_frame['Confidence Score'].mean():.0%} average confidence",
                "delta_color": SUCCESS_COLOR,
                "caption": "Preview based on generated history from the current planning sheet.",
            },
            {
                "label": "Projected SKUs",
                "value": f"{len(forecast_frame)}",
                "delta": f"{int((forecast_frame['Confidence Score'] < 0.55).sum())} need review",
                "delta_color": WARNING_COLOR,
                "caption": "Open the Demand Projection page to load actual monthly history.",
            },
        ]
    )
    preview_fig = px.bar(
        forecast_frame.nlargest(12, "Predicted Per Day Consumption").sort_values(
            "Predicted Per Day Consumption", ascending=True
        ),
        x="Predicted Per Day Consumption",
        y="Variant ID",
        orientation="h",
        color="Confidence Score",
        color_continuous_scale=["#FADBD8", "#FFF2CF", "#D9F2E7"],
        title="Predicted Per Day Consumption Preview",
    )
    st.plotly_chart(style_figure(preview_fig, height=380), use_container_width=True, config=PLOTLY_CONFIG)


def render_planning_table(filtered_df: pd.DataFrame, low_doi_threshold: float, target_doi: int) -> None:
    render_section_header(
        "Planning Sheet",
        caption="Excel-friendly execution view with the exact business logic preserved for movement planning.",
    )
    display_columns = [
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
        "FG DOI",
        "Movement Efficiency %",
        "Case Break Loss",
        "FG Status",
        "Decision",
        "Priority Level",
        "Risk Flags",
        "Remarks",
    ]
    if "Forecast Confidence" in filtered_df.columns:
        display_columns.append("Forecast Confidence")

    decision_table = filtered_df[display_columns].copy()
    format_map = {
        "Booked/Proj %": "{:.1%}",
        "Movement Efficiency %": "{:.1%}",
        "Current DOI": "{:.1f}",
        "DOI after movement": "{:.1f}",
        "FG DOI": "{:.1f}",
    }
    if "Forecast Confidence" in decision_table.columns:
        format_map["Forecast Confidence"] = "{:.0%}"

    styled_table = decision_table.style.format(format_map)

    def style_fg_status(column: pd.Series) -> list[str]:
        return [
            f"background-color: {DANGER_COLOR}; color: white" if value == "FG Shortage" else ""
            for value in column
        ]

    def style_doi(column: pd.Series) -> list[str]:
        styles: list[str] = []
        for value in column:
            if isinstance(value, (int, float)) and value < low_doi_threshold:
                styles.append(f"background-color: {PALETTE['warning_soft']}")
            elif isinstance(value, (int, float)) and value >= target_doi:
                styles.append(f"background-color: {SUCCESS_COLOR}; color: white")
            else:
                styles.append("")
        return styles

    styled_table = styled_table.apply(style_fg_status, subset=["FG Status"])
    styled_table = styled_table.apply(style_doi, subset=["Current DOI", "DOI after movement"])
    st.dataframe(styled_table, use_container_width=True, hide_index=True, height=560)


def main() -> None:
    inject_global_styles()

    with st.sidebar:
        st.header("Planning Controls")
        uploaded_file = st.file_uploader("Planning master", type=["xlsx", "xls", "csv"])
        use_sample = st.toggle("Use sample data", value=uploaded_file is None)
        st.divider()
        st.caption("Coverage rules")
        selected_days = st.number_input("Projection Days", min_value=1, max_value=180, value=20)
        target_doi = st.number_input("Target DOI", min_value=1, max_value=180, value=20)
        low_doi_threshold = st.number_input("Low DOI Threshold", min_value=0.5, max_value=60.0, value=5.0)
        excess_doi_threshold = st.number_input("Excess DOI Threshold", min_value=5.0, max_value=180.0, value=30.0)
        movement_threshold = st.number_input("Dead Stock PDA Threshold", min_value=0.0, max_value=20.0, value=1.0)
        demand_uplift_pct = st.slider("Demand uplift %", min_value=-50, max_value=100, value=0, step=5)
        st.divider()
        st.caption("Workflow")
        exception_only = st.toggle("Show exception SKUs only", value=False)
        use_ai_forecast = st.toggle(
            "Use saved next-month projection",
            value=False,
            disabled="forecast_override" not in st.session_state or st.session_state.get("forecast_override") is None,
            help="Enable this after saving a projection from the Demand Projection page.",
        )

    if uploaded_file is not None:
        raw_df = load_uploaded_table(uploaded_file)
        source_name = uploaded_file.name
    elif use_sample:
        raw_df = load_sample_data()
        source_name = "sample_data.csv"
    else:
        render_page_header(
            "Inventory Planning Workspace",
            "Load a planning master to start movement planning, DOI monitoring, and forecast-driven scenario analysis.",
            chips=["Excel upload ready", "Demand projection handoff", "Case-size execution logic"],
        )
        st.info("Upload a file or enable sample data to begin.")
        return

    try:
        base_df = prepare_input_frame(raw_df)
    except ValueError as exc:
        render_page_header(
            "Inventory Planning Workspace",
            "Load a planning master to start movement planning, DOI monitoring, and forecast-driven scenario analysis.",
            chips=["Excel upload ready", "Demand projection handoff", "Case-size execution logic"],
        )
        st.error(str(exc))
        st.write("Required columns:")
        st.code("\n".join(REQUIRED_COLUMNS))
        return

    st.session_state["planning_base_df"] = base_df.copy()
    st.session_state["planning_source_name"] = source_name

    render_page_header(
        "Inventory Planning Workspace",
        "Enterprise planning cockpit for retail coverage, movement decisions, FG risk, and scenario simulation.",
        chips=[
            source_name,
            f"{len(base_df)} SKUs loaded",
            "Demand Projection page available",
            "Case-size execution logic",
        ],
    )

    render_section_header(
        "Scenario Workspace",
        caption="Edit current stock, bookings, FG, demand, and remarks. Every change recalculates the plan instantly.",
        note="Safe to simulate. Original source data is not overwritten.",
    )
    editable_df = st.data_editor(
        base_df[
            [
                "Variant ID",
                "Name",
                "Brand",
                "Category",
                "ABC-Cat",
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
                "Remarks",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        height=320,
        key="simulation_editor",
        column_config={
            "Variant ID": st.column_config.TextColumn("Variant ID", disabled=True),
            "Name": st.column_config.TextColumn("Name", disabled=True, width="large"),
            "Brand": st.column_config.TextColumn("Brand", disabled=True),
            "Category": st.column_config.TextColumn("Category", disabled=True),
            "ABC-Cat": st.column_config.TextColumn("ABC-Cat", disabled=True),
            "Retail": st.column_config.NumberColumn("Retail", format="%.0f"),
            "Current": st.column_config.NumberColumn("Current", format="%.0f"),
            "Intransit": st.column_config.NumberColumn("Intransit", format="%.0f"),
            "Booked": st.column_config.NumberColumn("Booked", format="%.0f"),
            "FG": st.column_config.NumberColumn("FG", format="%.0f"),
            "Case Size": st.column_config.NumberColumn("Case Size", format="%.0f"),
            "Per Day": st.column_config.NumberColumn("Per Day", format="%.2f"),
            "PDA": st.column_config.NumberColumn("PDA", format="%.2f"),
            "Historical Sales": st.column_config.NumberColumn("Historical Sales", format="%.0f"),
            "Incoming Stock": st.column_config.NumberColumn("Incoming Stock", format="%.0f"),
            "Remarks": st.column_config.TextColumn("Remarks", width="medium"),
        },
    )

    edited_df = base_df.copy()
    for column in editable_df.columns:
        edited_df[column] = editable_df[column]

    if demand_uplift_pct != 0:
        uplift_factor = 1 + (demand_uplift_pct / 100)
        edited_df["Per Day"] = edited_df["Per Day"] * uplift_factor
        edited_df["PDA"] = edited_df["PDA"] * uplift_factor

    forecast_active = False
    if use_ai_forecast and isinstance(st.session_state.get("forecast_override"), pd.DataFrame):
        edited_df = apply_forecast_override(edited_df, st.session_state["forecast_override"])
        forecast_active = True
        st.success(
            f"Next-month projected demand is active using `{st.session_state.get('forecast_label', 'saved projection')}`."
        )

    category_options = sorted(edited_df["Category"].dropna().astype(str).unique().tolist())
    brand_options = sorted(edited_df["Brand"].dropna().astype(str).unique().tolist())
    abc_options = sorted(edited_df["ABC-Cat"].dropna().astype(str).unique().tolist())

    with st.sidebar:
        st.divider()
        st.header("Filters")
        selected_categories = st.multiselect("Category", category_options, default=category_options)
        selected_brands = st.multiselect("Brand", brand_options, default=brand_options)
        selected_abc = st.multiselect("ABC Category", abc_options, default=abc_options)

    plan_df = calculate_inventory_plan(
        edited_df,
        selected_days=selected_days,
        target_doi=target_doi,
        low_doi_threshold=low_doi_threshold,
        excess_doi_threshold=excess_doi_threshold,
        dead_stock_movement_threshold=movement_threshold,
    )
    plan_df["SKU Ranking Score"] = create_priority_score(plan_df)

    filtered_df = apply_filters(
        plan_df,
        categories=selected_categories,
        brands=selected_brands,
        abc_categories=selected_abc,
    )
    if exception_only:
        filtered_df = filtered_df[filtered_df["Exception Flag"] == "Action Required"].reset_index(drop=True)

    benchmark = st.session_state.get("planning_benchmark")
    toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([1, 1, 4])
    if toolbar_col1.button("Pin current view as benchmark", use_container_width=True):
        benchmark = build_scenario_snapshot(
            filtered_df,
            label=pd.Timestamp.now().strftime("%d %b %H:%M"),
        )
        st.session_state["planning_benchmark"] = benchmark
    if toolbar_col2.button("Clear benchmark", use_container_width=True):
        benchmark = None
        st.session_state["planning_benchmark"] = None
    toolbar_col3.caption(
        "Use the benchmark to compare demand uplifts, FG edits, or saved demand projections against a fixed planning baseline."
    )

    metrics = summarize_metrics(filtered_df)
    render_workspace_status(
        source_name=source_name,
        metrics=metrics,
        filtered_count=len(filtered_df),
        total_count=len(plan_df),
        selected_days=selected_days,
        target_doi=target_doi,
        demand_uplift_pct=demand_uplift_pct,
        forecast_active=forecast_active,
        benchmark_label=str(benchmark.get("label")) if isinstance(benchmark, dict) else None,
    )

    overview_tab, planner_tab, demand_tab = st.tabs(
        ["Executive Dashboard", "Planning Sheet", "Demand Signal"]
    )

    with overview_tab:
        render_kpi_cards(metrics, benchmark if isinstance(benchmark, dict) else None)
        render_alert_center(filtered_df)
        render_analytics(filtered_df)
        render_panel(
            "AI Planning Brief",
            generate_ai_planning_summary(filtered_df),
            caption="Heuristic planner commentary distilled from the current inventory and movement plan.",
        )

    with planner_tab:
        render_planning_table(filtered_df, low_doi_threshold=low_doi_threshold, target_doi=target_doi)
        st.caption("Totals row")
        totals_df = build_totals_row(filtered_df, selected_days=selected_days)[
            [
                "Variant ID",
                "Retail",
                "Current",
                "Intransit",
                "Booked",
                "1st Remaining",
                "Required for 20 days",
                "Per Day",
                "PDA",
                "Move",
                "FG",
                "Remaining according to case size",
                "Move in case",
                "Final Move Qty",
                "Suggested Partial Move",
            ]
        ]
        st.dataframe(totals_df, use_container_width=True, hide_index=True)
        export_df = build_export_frame(filtered_df)
        st.download_button(
            label="Download Excel plan",
            data=dataframe_to_excel_bytes("Inventory Plan", export_df),
            file_name="inventory_optimization_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with demand_tab:
        render_demand_intelligence(base_df)
        st.info(
            "Use the Demand Projection page to upload monthly history, validate last-month projection accuracy, and save next-month demand back into this planning workspace."
        )


if __name__ == "__main__":
    main()
