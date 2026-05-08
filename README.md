# Oil Quant Research Platform
https://oil-quant-research-platform-hxtzkr4s9swv633pjbcqbf.streamlit.app

Professional Python and Streamlit platform for quantitative oil market research.

This project analyzes the crude oil market through WTI and Brent prices, Brent-WTI spread, returns, volatility, drawdowns, futures curve structure, market drivers, Monte Carlo simulations, machine learning volatility forecasting, market regime detection, signal generation, backtesting, risk management and automated research reporting.

The objective is to build a complete quantitative research dashboard similar to what could be used by a commodities analyst, quant researcher or trading desk assistant.

---

## Project Overview

The platform focuses on the oil market and provides a full analytical workflow:

- Market overview of WTI, Brent and Brent-WTI spread
- Returns, volatility and drawdown analysis
- WTI futures curve analysis
- Market drivers analysis
- Machine learning volatility forecasting
- Monte Carlo price scenario simulation
- Market regime detection
- Multi-factor signal engine
- Historical backtesting
- Risk dashboard with VaR, CVaR and stress tests
- Automated HTML research report export

The project is built in Python with a modular architecture and an interactive Streamlit interface.

---

## Key Features

### 1. Market Overview

The dashboard displays historical WTI and Brent prices, calculates the Brent-WTI spread, and tracks the latest available market levels.

Main outputs:

- Latest WTI price
- Latest Brent price
- Latest Brent-WTI spread
- Historical price chart
- Historical spread chart

---

### 2. Returns and Risk Analysis

The project computes daily returns, rolling annualized volatility and drawdowns.

Main metrics:

- Current price
- Total performance
- Annualized volatility
- Maximum drawdown
- Daily returns
- Rolling volatility
- Historical drawdowns

---

### 3. Futures Curve Analysis

The futures curve module analyzes the structure of WTI futures contracts.

It identifies whether the curve is in:

- Contango
- Backwardation

It also computes:

- Curve slope
- Annualized roll yield
- Price change versus front contract
- Percentage change versus front contract

This section helps understand market expectations, storage pressure, supply-demand tension and carry structure.

---

### 4. Market Drivers

The dashboard connects oil price movements with external market drivers such as:

- VIX
- US Dollar Index
- S&P 500
- Energy sector ETF
- Oil ETF
- US 10Y yield

The module computes:

- Historical correlations with WTI
- Rolling correlations
- Normalized driver performance
- Driver summary statistics

---

### 5. Volatility Forecasting

A machine learning model is used to forecast future realized volatility of WTI.

The current implementation uses a Random Forest Regressor trained on features such as:

- Past returns
- Rolling volatility
- Momentum
- Lagged returns

Outputs include:

- Predicted future volatility
- Realized future volatility
- Model metrics
- Feature importance
- Latest predictions

---

### 6. Monte Carlo Scenario Lab

The Monte Carlo module simulates thousands of possible future WTI price paths.

It uses a Geometric Brownian Motion framework and can use either:

- Historical realized volatility
- Machine learning predicted volatility

Outputs include:

- Simulated price paths
- Percentile scenario cone
- Median simulated price
- Probability of loss
- Monte Carlo VaR
- Monte Carlo CVaR
- Final simulated price distribution

---

### 7. Market Regime Detection

The market regime module classifies the current oil market environment.

Possible regimes include:

- Bullish Trend
- Bearish Trend
- High Volatility
- Stress Regime
- Neutral

The classification uses indicators such as:

- 20-day momentum
- 60-day momentum
- Realized volatility
- Drawdown
- VIX level when available
- Dollar momentum when available

---

### 8. Signal Engine

The Signal Engine combines multiple components into a research signal.

Inputs include:

- Market regime
- Momentum
- Predicted volatility
- Futures curve regime
- Drawdown

Outputs include:

- Research signal
- Signal description
- Total score
- Target exposure
- Signal breakdown
- Historical signal distribution

Example signals:

- Strong Bullish
- Bullish
- Neutral
- Defensive
- Stress Defensive

---

### 9. Signal Engine Backtest

The platform backtests the historical Signal Engine exposure against a simple WTI Buy & Hold strategy.

Backtest outputs include:

- Signal strategy return
- Buy & Hold return
- Average exposure
- Sharpe ratio
- Maximum drawdown
- Annualized volatility
- Historical exposure chart
- Signal distribution

---

### 10. Risk Dashboard

The risk dashboard provides a complete view of downside risk.

It includes:

- Historical Value-at-Risk
- Historical Conditional Value-at-Risk
- Worst daily return
- Return distribution
- Distribution statistics
- Stress test scenarios
- Theoretical PnL under shocks

The dashboard supports two data modes:

- Cleaned Data
- Raw Data

Cleaned Data removes non-positive oil prices and caps extreme returns to make risk metrics more stable.

---

### 11. Executive Summary

The Executive Summary automatically synthesizes the full dashboard into a clear research-style output.

It displays:

- Current research signal
- Current market regime
- Risk level
- Target exposure
- Median Monte Carlo price
- Probability of loss
- Historical VaR
- Monte Carlo VaR
- Automatic market interpretation

This section is designed to quickly communicate the main conclusion of the quantitative analysis.

---

### 12. Research Report Export

The dashboard can generate an HTML research report directly from the Executive Summary.

The report includes:

- Executive summary
- Current signal
- Risk metrics
- Monte Carlo summary
- Signal breakdown
- Backtest metrics
- Automatic interpretation

The HTML report can be opened in a browser and exported to PDF.

---

## Project Structure

```text
oil-quant-research-platform/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│       └── wti_futures_curve.csv
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── backtest.py
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── futures_curve.py
│   ├── indicators.py
│   ├── market_drivers.py
│   ├── monte_carlo_simulator.py
│   ├── regime_detection.py
│   ├── report_generator.py
│   ├── research_summary.py
│   ├── risk.py
│   ├── signal_backtest.py
│   ├── signal_engine.py
│   └── volatility_model.py
│
└── .streamlit/
    └── config.toml
```

---

## Technologies Used

The project uses the following Python libraries:

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- yfinance
- scikit-learn

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/oil-quant-research-platform.git
cd oil-quant-research-platform
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```powershell
streamlit run app.py
```

The dashboard will open locally in your browser.

---

## Data

The project uses historical market data retrieved with Python and external financial data sources.

The futures curve is loaded from:

```text
data/raw/wti_futures_curve.csv
```

The expected structure of this file is:

```text
contract,maturity,price
CL1,2026-06-01,94.80
CL2,2026-07-01,94.10
CL3,2026-08-01,93.45
```

The file can be updated manually to reflect the current WTI futures curve.

---

## How to Use the Dashboard

After launching the application, use the sidebar to configure:

- Start date
- End date
- Rolling volatility window
- Rolling correlation window
- Volatility forecasting horizon
- Monte Carlo horizon
- Number of simulations
- Signal target volatility
- Maximum signal exposure
- Risk data mode
- Backtest parameters

Then navigate through the dashboard sections:

1. Executive Summary
2. Market Overview
3. Returns & Risk
4. Futures Curve
5. Market Drivers
6. Volatility Forecasting
7. Monte Carlo Lab
8. Market Regime
9. Signal Engine
10. Strategy Backtest
11. Risk Dashboard
12. Data

---

## Quantitative Methods

The project implements several quantitative finance concepts:

### Returns

```text
r_t = P_t / P_{t-1} - 1
```

### Annualized Volatility

```text
sigma_annualized = sigma_daily * sqrt(252)
```

### Drawdown

```text
DD_t = P_t / max(P_0, ..., P_t) - 1
```

### Historical VaR

```text
VaR_alpha = Quantile_{1-alpha}(R)
```

### Historical CVaR

```text
CVaR_alpha = E[R | R <= VaR_alpha]
```

### Futures Curve Slope

```text
Slope = F_long / F_front - 1
```

### Geometric Brownian Motion

```text
S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*epsilon)
```

---

## Example Research Interpretation

The platform automatically generates an interpretation such as:

```text
The model indicates a fragile market environment.
Capital preservation is prioritized over aggressive return seeking.
A low WTI exposure is consistent with the current regime.
```

This makes the dashboard useful not only as a visualization tool, but also as a research decision-support system.

---

## Current Status

The project currently includes:

- Functional Streamlit dashboard
- Modular Python source code
- Futures curve module
- Market driver module
- Volatility forecasting module
- Monte Carlo simulator
- Market regime detection
- Signal Engine
- Signal Engine backtest
- Risk dashboard
- Executive Summary
- HTML research report export

---

## Possible Future Improvements

Potential next steps:

- Add live futures curve data
- Add options-implied volatility data
- Add crude oil inventory data
- Add macroeconomic data from FRED
- Add portfolio optimization
- Add Bayesian regime detection
- Add deep learning volatility model
- Add PDF export directly from Python
- Deploy the dashboard online
- Add authentication for private use

---

## Disclaimer

This project is for educational and research purposes only.

It is not financial advice, investment advice or a trading recommendation. The results depend on historical data, model assumptions, parameter choices and data quality. Financial markets involve risk, and past performance does not guarantee future results.

---

## Author

Developed by Julien Ruiz.

Project focused on quantitative finance, commodities research, risk analytics and Python-based financial dashboards.
