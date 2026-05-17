from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import (
    build_history_template,
    build_multisheet_history_template,
    forecast_next_month,
    generate_ai_forecast_insights,
    generate_demo_history,
)
from ui_components import (
    APP_NAME,
    PALETTE,
    PLOTLY_CONFIG,
    inject_global_styles,
    render_metric_cards,
    render_page_header,
    render_panel,
    render_section_header,
    style_figure,
)
from utils import (
    dataframe_to_excel_bytes,
    dataframes_to_excel_bytes,
    load_uploaded_table,
    prepare_input_frame,
)


st.set_page_config(page_title=f"{APP_NAME} | Demand Projection", page_icon="📈", layout="wide")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
PRIMARY_COLOR = PALETTE["accent"]
SUCCESS_COLOR = PALETTE["success"]
WARNING_COLOR = PALETTE["warning"]
DANGER_COLOR = PALETTE["danger"]


@st.cache_data(show_spinner=False)
def load_sample_planning_data() -> pd.DataFrame:
    return prepare_input_frame(pd.read_csv(SAMPLE_DATA_PATH))


@st.cache_data(show_spinner=False)
def build_cached_projection(history_input, planning_df: pd.DataFrame):
    artifacts = forecast_next_month(history_input, planning_df)
    return artifacts.forecast.copy(), dict(artifacts.metrics), artifacts.monthly_history.copy()


def resolve_planning_frame(uploaded_planning_file) -> pd.DataFrame:
    if uploaded_planning_file is not None:
        return prepare_input_frame(load_uploaded_table(uploaded_planning_file))
    if isinstance(st.session_state.get("planning_base_df"), pd.DataFrame):
        return prepare_input_frame(st.session_state["planning_base_df"])
    return load_sample_planning_data()


def render_overview_cards(forecast_df: pd.DataFrame) -> None:
    previous_month = (
        forecast_df["Previous Month"].dropna().iloc[0]
        if forecast_df["Previous Month"].notna().any()
        else "-"
    )
    total_previous_projection = forecast_df["Previous Projection"].fillna(0).sum()
    total_actual = forecast_df["Actual Sold"].fillna(0).sum()
    total_same_month_last_year = forecast_df["Same Month Last Year Sold"].fillna(0).sum()
    total_next_projection = forecast_df["Next Month Projection"].fillna(0).sum()
    avg_accuracy = forecast_df["Accuracy %"].fillna(0).mean()
    total_yoy_change = total_next_projection - total_same_month_last_year
    total_yoy_pct = (total_yoy_change / total_same_month_last_year) if total_same_month_last_year > 0 else 0

    render_metric_cards(
        [
            {
                "label": "Last closed month",
                "value": previous_month,
                "delta": f"Average accuracy {avg_accuracy:.0%}",
                "delta_color": PRIMARY_COLOR,
                "caption": "Month used to compare the last projection against actual sold.",
            },
            {
                "label": "Previous projection",
                "value": f"{total_previous_projection:,.0f}",
                "delta": f"Actual sold {total_actual:,.0f}",
                "delta_color": WARNING_COLOR,
                "caption": "Uses your upload value when provided, otherwise an estimated prior projection.",
            },
            {
                "label": "Variance",
                "value": f"{(total_actual - total_previous_projection):,.0f}",
                "delta": f"{avg_accuracy:.0%} accuracy",
                "delta_color": SUCCESS_COLOR if total_actual >= total_previous_projection else DANGER_COLOR,
                "caption": "Difference between last projection and actual sold across the current filters.",
            },
            {
                "label": "Same month last year",
                "value": f"{total_same_month_last_year:,.0f}",
                "delta": f"{total_yoy_change:,.0f} ({total_yoy_pct:.1%}) vs next month",
                "delta_color": SUCCESS_COLOR if total_yoy_change >= 0 else DANGER_COLOR,
                "caption": "Lift or dip relative to the same month last year.",
            },
            {
                "label": "Next month projection",
                "value": f"{total_next_projection:,.0f}",
                "delta": f"{forecast_df['Forecast Confidence'].mean():.0%} average confidence",
                "delta_color": SUCCESS_COLOR,
                "caption": "Hybrid projection using recent trend, seasonality, and ML.",
            },
        ]
    )


def render_projection_overview(monthly_history: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    render_section_header(
        "Projection Overview",
        caption="Clear comparison of what was projected, what sold, and what the model recommends next.",
    )
    sku_options = forecast_df["Variant ID"].tolist()
    selected_sku = st.selectbox("Focus SKU", sku_options)
    selected_row = forecast_df[forecast_df["Variant ID"] == selected_sku].iloc[0]
    sku_history = monthly_history[monthly_history["Variant ID"] == selected_sku].copy().sort_values("Month")
    recent_history = sku_history.tail(6)
    next_month_ts = pd.to_datetime(f"01 {selected_row['Next Month Label']}")

    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=recent_history["Month"],
                y=recent_history["Units Sold"],
                mode="lines+markers",
                name="Actual Sold",
                line=dict(color=PRIMARY_COLOR, width=3),
            )
        )
        trend_fig.add_trace(
            go.Scatter(
                x=[recent_history["Month"].max()] if not recent_history.empty else [next_month_ts],
                y=[selected_row["Previous Projection"]],
                mode="markers",
                marker=dict(color=WARNING_COLOR, size=12, symbol="diamond"),
                name="Previous Projection",
            )
        )
        trend_fig.add_trace(
            go.Scatter(
                x=[next_month_ts],
                y=[selected_row["Next Month Projection"]],
                mode="markers+text",
                marker=dict(color=SUCCESS_COLOR, size=13),
                text=[f"{selected_row['Next Month Projection']:.0f}"],
                textposition="top center",
                name="Next Month Projection",
            )
        )
        trend_fig.update_layout(
            title=f"{selected_sku} demand trend",
            xaxis_title="Month",
            yaxis_title="Units",
        )
        st.plotly_chart(style_figure(trend_fig, height=410), use_container_width=True, config=PLOTLY_CONFIG)

    with right_col:
        driver_df = pd.DataFrame(
            {
                "Signal": [
                    "Previous Projection",
                    "Actual Sold",
                    "Same Month Last Year",
                    "Next Month Projection",
                ],
                "Units": [
                    selected_row["Previous Projection"],
                    selected_row["Actual Sold"],
                    selected_row["Same Month Last Year Sold"],
                    selected_row["Next Month Projection"],
                ],
            }
        )
        driver_fig = px.bar(
            driver_df,
            x="Signal",
            y="Units",
            color="Signal",
            color_discrete_map={
                "Previous Projection": WARNING_COLOR,
                "Actual Sold": PRIMARY_COLOR,
                "Same Month Last Year": DANGER_COLOR,
                "Next Month Projection": SUCCESS_COLOR,
            },
            title="Projection story",
        )
        driver_fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        st.plotly_chart(style_figure(driver_fig, height=410, show_legend=False), use_container_width=True, config=PLOTLY_CONFIG)

    lower_col1, lower_col2 = st.columns(2, gap="large")

    top_projection_df = forecast_df.nlargest(12, "Next Month Projection")[
        ["Variant ID", "Previous Projection", "Actual Sold", "Next Month Projection"]
    ].copy()
    comparison_melted = top_projection_df.melt(
        id_vars="Variant ID",
        value_vars=["Previous Projection", "Actual Sold", "Next Month Projection"],
        var_name="Measure",
        value_name="Units",
    )
    compare_fig = px.bar(
        comparison_melted,
        x="Variant ID",
        y="Units",
        color="Measure",
        barmode="group",
        title="Top SKUs: previous projection vs actual vs next month",
        color_discrete_map={
            "Previous Projection": WARNING_COLOR,
            "Actual Sold": PRIMARY_COLOR,
            "Next Month Projection": SUCCESS_COLOR,
        },
    )
    lower_col1.plotly_chart(style_figure(compare_fig, height=400), use_container_width=True, config=PLOTLY_CONFIG)

    if "Projection Qty" in monthly_history.columns and monthly_history["Projection Qty"].notna().any():
        monthly_compare = (
            monthly_history.groupby("Month", dropna=False)[["Projection Qty", "Units Sold"]]
            .sum(min_count=1)
            .reset_index()
            .sort_values("Month")
        )
        monthly_compare["Month Label"] = monthly_compare["Month"].dt.strftime("%b %Y")
        monthly_compare["Accuracy %"] = monthly_compare.apply(
            lambda row: max(0, 1 - (abs(row["Units Sold"] - row["Projection Qty"]) / row["Units Sold"]))
            if row["Units Sold"] > 0 and pd.notna(row["Projection Qty"])
            else pd.NA,
            axis=1,
        )
        history_fig = go.Figure()
        history_fig.add_trace(
            go.Bar(
                x=monthly_compare["Month Label"],
                y=monthly_compare["Projection Qty"],
                name="Projection",
                marker_color=WARNING_COLOR,
            )
        )
        history_fig.add_trace(
            go.Bar(
                x=monthly_compare["Month Label"],
                y=monthly_compare["Units Sold"],
                name="Actual Sold",
                marker_color=PRIMARY_COLOR,
            )
        )
        history_fig.add_trace(
            go.Scatter(
                x=monthly_compare["Month Label"],
                y=(monthly_compare["Accuracy %"].fillna(0) * monthly_compare["Units Sold"].max()),
                name="Accuracy trend",
                mode="lines+markers",
                line=dict(color=SUCCESS_COLOR, width=3),
                yaxis="y2",
                hovertemplate="Accuracy: %{customdata:.0%}<extra></extra>",
                customdata=monthly_compare["Accuracy %"].fillna(0),
            )
        )
        history_fig.update_layout(
            title="Monthly projection vs actual trend",
            barmode="group",
            yaxis_title="Units",
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Scaled accuracy"),
        )
        lower_col2.plotly_chart(style_figure(history_fig, height=400), use_container_width=True, config=PLOTLY_CONFIG)
    else:
        yoy_df = forecast_df[["Variant ID", "YoY Change %"]].copy().dropna().sort_values("YoY Change %", ascending=False).head(15)
        yoy_fig = px.bar(
            yoy_df,
            x="YoY Change %",
            y="Variant ID",
            orientation="h",
            color="YoY Change %",
            color_continuous_scale=["#D93025", "#FFF2CF", "#1C8C5E"],
            title="Same-month last year rise or dip",
        )
        yoy_fig.update_traces(hovertemplate="SKU: %{y}<br>YoY change: %{x:.1%}<extra></extra>")
        lower_col2.plotly_chart(style_figure(yoy_fig, height=400), use_container_width=True, config=PLOTLY_CONFIG)


def render_projection_table(forecast_df: pd.DataFrame) -> None:
    render_section_header(
        "Projection Table",
        caption="SKU-level next month projection with a simple business view for planners and leadership.",
    )
    display_columns = [
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
    ]
    display_df = forecast_df[display_columns].copy()
    st.dataframe(
        display_df.style.format(
            {
                "Accuracy %": "{:.0%}",
                "YoY Change %": "{:.1%}",
                "Forecast Confidence": "{:.0%}",
                "Daily Projection": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=560,
    )


def main() -> None:
    inject_global_styles()
    render_page_header(
        "Demand Projection",
        "Upload six months of monthly sales in either one simple sheet or one sheet per month, compare projection vs actual, and generate a sharper next-month projection.",
        chips=[
            "Wide-format upload supported",
            "Monthly workbook supported",
            "Previous projection comparison",
            "Same-month last year analysis",
        ],
    )

    with st.sidebar:
        st.header("Inputs")
        planning_file = st.file_uploader("Planning master", type=["xlsx", "xls", "csv"], key="projection_planning")
        historical_file = st.file_uploader("Monthly sales history", type=["xlsx", "xls", "csv"], key="projection_history")
        use_sample_history = st.toggle("Use sample history", value=historical_file is None)
        clear_saved_projection = st.button("Clear saved projection")

    if clear_saved_projection:
        st.session_state["forecast_override"] = None
        st.session_state["forecast_label"] = ""
        st.success("Saved projection has been cleared.")

    try:
        planning_df = resolve_planning_frame(planning_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state["planning_base_df"] = planning_df.copy()

    if historical_file is not None:
        history_input = load_uploaded_table(historical_file, all_sheets=True)
        if isinstance(history_input, dict):
            history_source = f"{historical_file.name} ({len(history_input)} sheets)"
        else:
            history_source = historical_file.name
    elif use_sample_history:
        history_input = generate_demo_history(planning_df)
        history_source = "generated sample history"
    else:
        st.info("Upload monthly sales history or enable sample history to continue.")
        return

    with st.spinner("Building next-month projection from monthly history..."):
        forecast_df, metrics, monthly_history = build_cached_projection(history_input, planning_df)

    category_options = sorted(forecast_df["Category"].dropna().astype(str).unique().tolist())
    brand_options = sorted(forecast_df["Brand"].dropna().astype(str).unique().tolist())
    with st.sidebar:
        st.header("Filters")
        selected_categories = st.multiselect("Category", category_options, default=category_options)
        selected_brands = st.multiselect("Brand", brand_options, default=brand_options)

    if selected_categories:
        forecast_df = forecast_df[forecast_df["Category"].isin(selected_categories)].reset_index(drop=True)
    if selected_brands:
        forecast_df = forecast_df[forecast_df["Brand"].isin(selected_brands)].reset_index(drop=True)

    if forecast_df.empty:
        st.warning("No SKUs match the current filters.")
        return

    render_overview_cards(forecast_df)
    render_panel(
        "Projection Brief",
        generate_ai_forecast_insights(forecast_df),
        caption=f"History source: {history_source}. Method: {metrics.get('forecast_method', 'Hybrid ensemble')}.",
    )

    with st.expander("Upload formats", expanded=True):
        st.write("Recommended format: one Excel workbook with one sheet per month.")
        st.code("Sheet name: 2026-04\nColumns: Variant ID, Name, Brand, Category, Projection, Actual Sold")
        st.write("Alternate simple format: one row per SKU with the last 6 months as columns.")
        st.code("Variant ID, Name, Brand, Category, Previous Projection, 2025-11, 2025-12, 2026-01, 2026-02, 2026-03, 2026-04, Same Month Last Year")
        st.write("Minimum format:")
        st.code("Variant ID, 2025-11, 2025-12, 2026-01, 2026-02, 2026-03, 2026-04")
        st.write("Also supported:")
        st.code("Month, Variant ID, Units Sold")
        st.write("Optional columns that improve accuracy:")
        st.code("Previous Projection, Same Month Last Year, Same Month 2 Years Ago, Same Month 3 Years Ago")

    overview_tab, sku_tab, upload_tab = st.tabs(["Overview", "Projection Table", "Upload Guide"])

    with overview_tab:
        render_projection_overview(monthly_history, forecast_df)

    with sku_tab:
        render_projection_table(forecast_df)
        action_col1, action_col2 = st.columns(2)
        if action_col1.button("Use next-month projection in Planning page", type="primary"):
            st.session_state["forecast_override"] = forecast_df.copy()
            st.session_state["forecast_label"] = forecast_df["Next Month Label"].iloc[0]
            st.success("Projection saved. Go back to the Planning page and enable `Use saved next-month projection`.")

        export_df = forecast_df[
            [
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
            ]
        ].rename(columns={"Forecast Confidence": "Confidence"})
        action_col2.download_button(
            label="Download projection file",
            data=dataframe_to_excel_bytes("Demand Projection", export_df),
            file_name="next_month_projection.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with upload_tab:
        template_df = build_history_template(planning_df)
        multi_sheet_template = build_multisheet_history_template(planning_df)
        forecast_month_label = template_df.attrs.get("forecast_month", "next month")
        render_section_header(
            "Upload Guide",
            caption=f"Recommended: use the monthly workbook template with one tab per month. The single-sheet file is still available as a backup option. Both lead into a projection for {forecast_month_label}.",
        )
        download_col1, download_col2 = st.columns(2)
        download_col1.download_button(
            label="Download recommended monthly workbook template",
            data=dataframes_to_excel_bytes(multi_sheet_template),
            file_name="monthly_projection_workbook_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        download_col2.download_button(
            label="Download alternate single-sheet template",
            data=dataframe_to_excel_bytes("History Template", template_df),
            file_name="easy_demand_projection_upload.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.dataframe(monthly_history, use_container_width=True, hide_index=True, height=420)


if __name__ == "__main__":
    main()
