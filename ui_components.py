from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import plotly.graph_objects as go
import streamlit as st


APP_NAME = "SupplyPilot Control Tower"

PALETTE = {
    "ink": "#10223A",
    "muted": "#6B7A90",
    "surface": "#F4F7FB",
    "surface_alt": "#FFFFFF",
    "accent": "#1F4E79",
    "accent_soft": "#DCE9F8",
    "success": "#1C8C5E",
    "success_soft": "#D9F2E7",
    "warning": "#E7A814",
    "warning_soft": "#FFF2CF",
    "danger": "#D93025",
    "danger_soft": "#FADBD8",
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "toggleSpikelines",
        "autoScale2d",
    ],
}


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light dark;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(31, 78, 121, 0.10), transparent 30%),
                radial-gradient(circle at top left, rgba(28, 140, 94, 0.08), transparent 28%);
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1440px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(107, 122, 144, 0.18);
        }

        .ct-page-header {
            padding: 1.55rem 1.7rem 1.35rem;
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(16, 34, 58, 0.96), rgba(31, 78, 121, 0.94));
            color: #F5F8FC;
            box-shadow: 0 20px 45px rgba(16, 34, 58, 0.18);
            margin-bottom: 1rem;
        }

        .ct-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            opacity: 0.8;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .ct-title {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0;
        }

        .ct-subtitle {
            margin-top: 0.55rem;
            font-size: 0.98rem;
            color: rgba(245, 248, 252, 0.88);
            max-width: 980px;
        }

        .ct-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.95rem;
        }

        .ct-chip {
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.15);
            font-size: 0.78rem;
            font-weight: 600;
        }

        .ct-section-head {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 1rem;
            margin: 0.25rem 0 0.8rem;
        }

        .ct-section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #10223A;
            margin: 0;
        }

        .ct-section-copy {
            color: #6B7A90;
            font-size: 0.88rem;
            margin-top: 0.18rem;
        }

        .ct-section-note {
            color: #1F4E79;
            background: rgba(31, 78, 121, 0.08);
            border-radius: 999px;
            padding: 0.38rem 0.82rem;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .ct-metric-card {
            border-radius: 20px;
            padding: 1rem 1.05rem 0.95rem;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(107, 122, 144, 0.16);
            box-shadow: 0 10px 28px rgba(16, 34, 58, 0.08);
            min-height: 132px;
        }

        .ct-metric-label {
            color: #6B7A90;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .ct-metric-value {
            color: #10223A;
            font-size: 1.58rem;
            font-weight: 700;
            line-height: 1.1;
            margin-top: 0.4rem;
        }

        .ct-metric-delta {
            font-size: 0.8rem;
            font-weight: 700;
            margin-top: 0.55rem;
        }

        .ct-metric-caption {
            color: #6B7A90;
            font-size: 0.78rem;
            margin-top: 0.35rem;
            line-height: 1.35;
        }

        .ct-panel {
            border-radius: 22px;
            padding: 1rem 1.05rem 1.05rem;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(107, 122, 144, 0.16);
            box-shadow: 0 10px 28px rgba(16, 34, 58, 0.06);
        }

        .ct-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }

        .ct-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.35rem 0.72rem;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .ct-list {
            margin: 0;
            padding-left: 1rem;
            color: #24364F;
        }

        .ct-list li {
            margin-bottom: 0.35rem;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.4rem;
        }

        [data-testid="stTabs"] [role="tab"] {
            border-radius: 999px;
            padding-left: 1rem;
            padding-right: 1rem;
            border: 1px solid rgba(107, 122, 144, 0.18);
            background: rgba(255, 255, 255, 0.72);
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: rgba(31, 78, 121, 0.10);
            color: #1F4E79;
            border-color: rgba(31, 78, 121, 0.22);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(107, 122, 144, 0.16);
            padding: 0.85rem 0.95rem;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(16, 34, 58, 0.05);
        }

        [data-testid="stDataFrame"], [data-testid="stMarkdownContainer"] table {
            border-radius: 16px;
        }

        @media (prefers-color-scheme: dark) {
            .ct-section-title,
            .ct-metric-value,
            .ct-list {
                color: #F2F5FA;
            }

            .ct-section-copy,
            .ct-metric-caption,
            .ct-metric-label {
                color: rgba(242, 245, 250, 0.72);
            }

            .ct-metric-card,
            .ct-panel,
            [data-testid="stMetric"],
            [data-testid="stTabs"] [role="tab"] {
                background: rgba(16, 34, 58, 0.72);
                border-color: rgba(255, 255, 255, 0.09);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
            }

            .ct-section-note {
                background: rgba(111, 168, 220, 0.18);
                color: #D8E9FF;
            }

            [data-testid="stTabs"] [aria-selected="true"] {
                background: rgba(111, 168, 220, 0.18);
                color: #D8E9FF;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    chips: Sequence[str] | None = None,
    eyebrow: str = APP_NAME,
) -> None:
    chip_html = ""
    if chips:
        chip_html = '<div class="ct-chip-row">' + "".join(
            f'<span class="ct-chip">{chip}</span>' for chip in chips
        ) + "</div>"
    st.markdown(
        f"""
        <section class="ct-page-header">
            <div class="ct-eyebrow">{eyebrow}</div>
            <h1 class="ct-title">{title}</h1>
            <div class="ct-subtitle">{subtitle}</div>
            {chip_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, caption: str | None = None, note: str | None = None) -> None:
    copy_html = f'<div class="ct-section-copy">{caption}</div>' if caption else ""
    note_html = f'<div class="ct-section-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="ct-section-head">
            <div>
                <div class="ct-section-title">{title}</div>
                {copy_html}
            </div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: Sequence[Mapping[str, str | None]]) -> None:
    if not metrics:
        return
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        delta = metric.get("delta") or ""
        caption = metric.get("caption") or ""
        delta_color = metric.get("delta_color") or PALETTE["accent"]
        column.markdown(
            f"""
            <div class="ct-metric-card">
                <div class="ct-metric-label">{metric.get("label", "")}</div>
                <div class="ct-metric-value">{metric.get("value", "")}</div>
                <div class="ct-metric-delta" style="color: {delta_color};">{delta}</div>
                <div class="ct-metric-caption">{caption}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_badges(items: Iterable[tuple[str, str, str]]) -> None:
    badge_html = "".join(
        f'<span class="ct-badge" style="background:{background}; color:{text_color};">{label}</span>'
        for label, background, text_color in items
    )
    st.markdown(f'<div class="ct-badge-row">{badge_html}</div>', unsafe_allow_html=True)


def render_panel(title: str, items: Sequence[str], caption: str | None = None) -> None:
    render_section_header(title, caption=caption)
    list_html = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="ct-panel">
            <ul class="ct-list">{list_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(
    fig: go.Figure,
    *,
    title: str | None = None,
    height: int = 360,
    show_legend: bool = True,
) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ink"], size=13),
        margin=dict(l=18, r=18, t=54, b=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        showlegend=show_legend,
        hoverlabel=dict(
            bgcolor=PALETTE["surface_alt"],
            font=dict(color=PALETTE["ink"]),
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(107, 122, 144, 0.18)", zeroline=False)
    return fig
