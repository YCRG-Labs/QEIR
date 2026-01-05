# Replication Package: Fiscal Thresholds and QE Effectiveness

This folder contains the replication code for the main results in Table 2 of the paper.

## Requirements

```bash
pip install pandas numpy fredapi statsmodels python-dotenv
```

## Setup

1. Get a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a `.env` file in the project root with:
   ```
   FRED_API_KEY=your_key_here
   ```

## Running the Replication

```bash
python replication/main_results.py
```

## Expected Output

```
======================================================================
TABLE 2: QE EFFECTIVENESS BY FISCAL REGIME
======================================================================

Sample Period: 2009-03-01 to 2015-02-28
Threshold: 0.161
Frequency: monthly

----------------------------------------------------------------------
Regime                Coefficient   Std. Error    p-value        N
----------------------------------------------------------------------
Low Debt                  -9.29***         2.59     0.0003       62
High Debt                 -3.71           2.29     0.1052       10
----------------------------------------------------------------------

Attenuation: 60.0%
Total Observations: 72

======================================================================
Notes: Robust standard errors (HC1). *** p<0.01, ** p<0.05, * p<0.10
======================================================================
```

## Key Results

| Metric | Value |
|--------|-------|
| Threshold | 0.161 |
| Low-debt effect | -9.3 bps (p < 0.001) |
| High-debt effect | -3.7 bps (p = 0.105) |
| Attenuation | 60% |
| N (low/high) | 62/10 |

## Data Sources

All data is publicly available from FRED:
- `DGS10`: 10-Year Treasury Constant Maturity Rate
- `A091RC1Q027SBEA`: Federal Government Interest Payments
- `FGRECPT`: Federal Government Current Receipts

## Methodology

1. Construct QE shocks from daily yield changes around FOMC announcements
2. Aggregate to monthly frequency
3. Split sample by fiscal constraint threshold (interest/revenue ratio)
4. Estimate separate regressions for low-debt and high-debt regimes
5. Report with heteroskedasticity-robust standard errors (HC1)
