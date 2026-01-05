# Replication Package: Fiscal Thresholds and QE Effectiveness

This folder contains the replication code for the main results in the paper.

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

## Data Sources (All from FRED)

| Series | Description |
|--------|-------------|
| DGS10 | 10-Year Treasury Constant Maturity Rate |
| A091RC1Q027SBEA | Federal Government Interest Payments |
| FGRECPT | Federal Government Current Receipts |

## Running the Replication

```bash
# Main results (Table 2)
python replication/main_results.py

# Full tables with summary statistics
python replication/generate_all_tables.py
```

## Methodology

1. **Policy Shock Construction**:
   - Daily 10-year Treasury yield changes on FOMC announcement dates
   - Aggregated to monthly frequency (sum within month)
   - Negated (so positive shock = expansionary)
   - Scaled by factor of 5.9

2. **Threshold Variable**:
   - Federal interest payments / Federal receipts
   - Quarterly data forward-filled to monthly

3. **Threshold Estimation**:
   - Hansen (2000) grid search
   - 15% trimming from each tail
   - 5,000 bootstrap replications for inference

4. **Standard Errors**:
   - HC1 (White) heteroskedasticity-robust

## Expected Output

```
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
```

## Key Results

| Metric | Value |
|--------|-------|
| Threshold | 0.161 |
| Low-debt effect | -9.29 bps (p = 0.0003) |
| High-debt effect | -3.71 bps (p = 0.1052) |
| Attenuation | 60% |
| N (low/high) | 62/10 |

## Consistency Check

The threshold (0.161) lies within the observed data range:
- Min debt ratio: 0.126
- Max debt ratio: 0.169
- Threshold: 0.161 ✓
