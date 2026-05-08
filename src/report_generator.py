# src/report_generator.py

from datetime import datetime
from html import escape

import pandas as pd


def format_report_value(value, value_type: str = "text") -> str:
    """
    Formate une valeur pour l'affichage dans le rapport HTML.
    """

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        pass

    if value_type == "percent":
        return f"{value:.2%}"

    if value_type == "price":
        return f"{value:.2f} $"

    if value_type == "number":
        return f"{value:.2f}"

    return escape(str(value))


def dataframe_to_html_table(df: pd.DataFrame) -> str:
    """
    Convertit un DataFrame en tableau HTML stylisé.
    """

    if df is None or df.empty:
        return "<p>No data available.</p>"

    html = "<table>"
    html += "<thead><tr>"

    for column in df.columns:
        html += f"<th>{escape(str(column))}</th>"

    html += "</tr></thead>"
    html += "<tbody>"

    for _, row in df.iterrows():
        html += "<tr>"

        for value in row:
            html += f"<td>{escape(str(value))}</td>"

        html += "</tr>"

    html += "</tbody></table>"

    return html


def generate_research_report_html(
    executive_summary: dict,
    executive_summary_table: pd.DataFrame,
    signal_breakdown: pd.DataFrame | None = None,
    signal_backtest_metrics: pd.DataFrame | None = None,
    monte_carlo_summary: pd.DataFrame | None = None,
    monte_carlo_risk_table: pd.DataFrame | None = None,
    project_title: str = "Oil Quant Research Platform",
    start_date: str = "",
    end_date: str = "",
    monte_carlo_horizon: int | None = None,
    risk_data_mode: str = "",
) -> str:
    """
    Génère un rapport HTML complet à partir des résultats du dashboard.
    """

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    signal = executive_summary.get("signal", "N/A")
    signal_description = executive_summary.get("signal_description", "N/A")
    total_score = executive_summary.get("total_score", "N/A")
    target_exposure = executive_summary.get("target_exposure", None)

    regime = executive_summary.get("regime", "N/A")
    risk_level = executive_summary.get("risk_level", "N/A")

    median_simulated_price = executive_summary.get("median_simulated_price", None)
    probability_of_loss = executive_summary.get("probability_of_loss", None)

    historical_var_95 = executive_summary.get("historical_var_95", None)
    monte_carlo_var_95 = executive_summary.get("monte_carlo_var_95", None)
    annualized_volatility = executive_summary.get("annualized_volatility", None)

    interpretation = executive_summary.get("interpretation", "No interpretation available.")

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{escape(project_title)} - Research Report</title>

        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #0B0F19;
                color: #F9FAFB;
                font-family: Arial, sans-serif;
            }}

            .page {{
                max-width: 1100px;
                margin: 0 auto;
                padding: 48px 56px;
            }}

            .header {{
                border-bottom: 1px solid rgba(212, 175, 55, 0.35);
                padding-bottom: 24px;
                margin-bottom: 32px;
            }}

            h1 {{
                margin: 0;
                font-size: 38px;
                letter-spacing: -0.04em;
            }}

            h2 {{
                margin-top: 38px;
                margin-bottom: 16px;
                font-size: 26px;
            }}

            h3 {{
                margin-bottom: 10px;
                font-size: 18px;
                color: #D4AF37;
            }}

            p {{
                color: #D1D5DB;
                line-height: 1.65;
                font-size: 15px;
            }}

            .subtitle {{
                color: #D1D5DB;
                font-size: 16px;
                margin-top: 12px;
                max-width: 850px;
            }}

            .meta {{
                color: #9CA3AF;
                font-size: 13px;
                margin-top: 16px;
            }}

            .grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
                margin-top: 24px;
            }}

            .card {{
                background: linear-gradient(145deg, #111827, #020617);
                border: 1px solid rgba(212, 175, 55, 0.25);
                border-radius: 18px;
                padding: 20px;
                min-height: 120px;
            }}

            .label {{
                color: #9CA3AF;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.09em;
                margin-bottom: 12px;
            }}

            .value {{
                color: #F9FAFB;
                font-size: 28px;
                font-weight: 800;
                letter-spacing: -0.03em;
                margin-bottom: 8px;
            }}

            .caption {{
                color: #D4AF37;
                font-size: 13px;
                line-height: 1.45;
            }}

            .insight {{
                background: rgba(17, 24, 39, 0.90);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                padding: 24px;
                margin-top: 24px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 14px;
                overflow: hidden;
                border-radius: 14px;
                font-size: 14px;
            }}

            th {{
                background: #1F2937;
                color: #F9FAFB;
                text-align: left;
                padding: 12px;
                border: 1px solid #374151;
            }}

            td {{
                background: #111827;
                color: #D1D5DB;
                padding: 11px 12px;
                border: 1px solid #374151;
            }}

            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid rgba(148, 163, 184, 0.20);
                color: #9CA3AF;
                font-size: 12px;
                line-height: 1.6;
            }}

            @media print {{
                body {{
                    background: white;
                    color: black;
                }}

                .page {{
                    padding: 24px;
                }}

                .card, .insight, td, th {{
                    -webkit-print-color-adjust: exact;
                    print-color-adjust: exact;
                }}
            }}
        </style>
    </head>

    <body>
        <div class="page">

            <div class="header">
                <h1>{escape(project_title)}</h1>
                <p class="subtitle">
                    Quantitative oil market research report generated from the Python dashboard.
                </p>
                <p class="meta">
                    Report generated: {escape(report_date)}<br>
                    Data period: {escape(start_date)} to {escape(end_date)}<br>
                    Risk data mode: {escape(risk_data_mode)}<br>
                    Monte Carlo horizon: {monte_carlo_horizon if monte_carlo_horizon is not None else "N/A"} days
                </p>
            </div>

            <h2>Executive Summary</h2>

            <div class="grid">
                <div class="card">
                    <div class="label">Research Signal</div>
                    <div class="value">{escape(str(signal))}</div>
                    <div class="caption">{escape(str(signal_description))}</div>
                </div>

                <div class="card">
                    <div class="label">Market Regime</div>
                    <div class="value">{escape(str(regime))}</div>
                    <div class="caption">Current market regime</div>
                </div>

                <div class="card">
                    <div class="label">Risk Level</div>
                    <div class="value">{escape(str(risk_level))}</div>
                    <div class="caption">Global risk classification</div>
                </div>

                <div class="card">
                    <div class="label">Target Exposure</div>
                    <div class="value">{format_report_value(target_exposure, "percent")}</div>
                    <div class="caption">Total score: {escape(str(total_score))}</div>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <div class="label">Median MC Price</div>
                    <div class="value">{format_report_value(median_simulated_price, "price")}</div>
                    <div class="caption">Median simulated WTI price</div>
                </div>

                <div class="card">
                    <div class="label">Probability of Loss</div>
                    <div class="value">{format_report_value(probability_of_loss, "percent")}</div>
                    <div class="caption">Simulated probability of ending below current price</div>
                </div>

                <div class="card">
                    <div class="label">Historical VaR 95%</div>
                    <div class="value">{format_report_value(historical_var_95, "percent")}</div>
                    <div class="caption">Historical daily downside risk</div>
                </div>

                <div class="card">
                    <div class="label">Monte Carlo VaR 95%</div>
                    <div class="value">{format_report_value(monte_carlo_var_95, "percent")}</div>
                    <div class="caption">Simulated downside risk at selected horizon</div>
                </div>
            </div>

            <div class="insight">
                <h3>Automatic Market Interpretation</h3>
                <p>{escape(str(interpretation))}</p>
            </div>

            <h2>Detailed Summary Table</h2>
            {dataframe_to_html_table(executive_summary_table)}

            <h2>Signal Breakdown</h2>
            {dataframe_to_html_table(signal_breakdown)}

            <h2>Signal Backtest Metrics</h2>
            {dataframe_to_html_table(signal_backtest_metrics)}

            <h2>Monte Carlo Summary</h2>
            {dataframe_to_html_table(monte_carlo_summary)}

            <h2>Monte Carlo Risk Table</h2>
            {dataframe_to_html_table(monte_carlo_risk_table)}

            <h2>Risk Snapshot</h2>
            <div class="grid">
                <div class="card">
                    <div class="label">Annualized Volatility</div>
                    <div class="value">{format_report_value(annualized_volatility, "percent")}</div>
                    <div class="caption">Annualized volatility based on selected risk data</div>
                </div>

                <div class="card">
                    <div class="label">Historical VaR 95%</div>
                    <div class="value">{format_report_value(historical_var_95, "percent")}</div>
                    <div class="caption">Historical Value-at-Risk</div>
                </div>

                <div class="card">
                    <div class="label">Monte Carlo VaR 95%</div>
                    <div class="value">{format_report_value(monte_carlo_var_95, "percent")}</div>
                    <div class="caption">Scenario-based Value-at-Risk</div>
                </div>

                <div class="card">
                    <div class="label">Risk Level</div>
                    <div class="value">{escape(str(risk_level))}</div>
                    <div class="caption">Global risk classification</div>
                </div>
            </div>

            <div class="footer">
                This report is generated automatically for research and educational purposes.
                It is not investment advice. Model outputs depend on data quality, historical assumptions,
                volatility estimation, and user-selected dashboard parameters.
            </div>

        </div>
    </body>
    </html>
    """

    return html