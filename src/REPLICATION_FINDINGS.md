# Replication Findings: Fiscal Thresholds Paper

## Paper Claims (Table 2)
- **Threshold**: 0.160
- **Low-debt effect**: -9.4 bps (p < 0.05)
- **High-debt effect**: -3.5 bps (not significant, p > 0.10)
- **Sample sizes**: N = 40/23
- **Attenuation**: 63%

## Best Matching Specification Found

### Monthly Frequency - Best Coefficient Match
**Specification**: `M|sum|/5.9|2009-03-2015-02@0.161`

| Metric | Paper | Replication | Match? |
|--------|-------|-------------|--------|
| Threshold | 0.160 | 0.161 | ≈ |
| Low-debt β | -9.4 bps | -9.37 bps | ✓ |
| Low-debt p | < 0.05 | 0.0003 | ✓ |
| High-debt β | -3.5 bps | -3.71 bps | ≈ |
| High-debt p | > 0.10 | 0.105 | ✓ |
| N | 40/23 | 61/10 | ✗ |
| Attenuation | 63% | 60% | ≈ |

**Key choices**:
- Sample: March 2009 - February 2015 (monthly)
- Shock: Sum of daily FOMC shocks per month, negated, divided by 5.9
- DV: Monthly yield change (end-of-month)
- Threshold: Interest payments / Federal revenue ≈ 0.161
- No macro controls

### Alternative Close Matches

1. **2009-03 to 2014-03, scale=5.5, thresh=0.161**:
   - Low: -8.7 bps (p=0.001), High: -3.5 bps (p=0.105)
   - N: 50/10
   - High-debt coefficient exactly -3.5!

2. **2009-03 to 2015-03, scale=6.0, thresh=0.161**:
   - Low: -9.4 bps (p=0.000), High: -3.8 bps (p=0.105)
   - N: 62/10
   - Low-debt coefficient exactly -9.4!

## Critical Finding: N=40/23 Cannot Be Matched

When we constrain the search to find exactly N=40/23:
- Only one sample period produces this split: **2009-05 to 2014-08, threshold=0.158**
- However, the significance pattern is **REVERSED**: 
  - Low-debt: NOT significant (p=0.078)
  - High-debt: Significant (p<0.001)
- This is the **opposite** of what the paper claims

**Conclusion**: There is no monthly specification with N=40/23 that produces the paper's claimed significance pattern (low significant, high not significant).

## Summary of Discrepancies

| Aspect | Can Match? | Notes |
|--------|------------|-------|
| Low-debt coefficient (-9.4) | ✓ | Exact match possible |
| Low-debt significance (p<0.05) | ✓ | Easily achieved |
| High-debt coefficient (-3.5) | ≈ | Close (-3.5 to -3.8) |
| High-debt insignificance (p>0.10) | ✓ | Achieved with thresh≈0.161 |
| Sample size N=40/23 | ✗ | Cannot match with correct significance |
| Threshold 0.160 | ≈ | Need 0.161 for correct significance |
| Attenuation 63% | ≈ | Get 60% with best match |

## What's Required to Reproduce Results

1. **Sample Restriction**: Active QE period only (2009-2015)
   - Full sample produces different results
   - This restriction is not clearly documented in the paper

2. **Shock Transformation**: Must negate and scale by ~6
   - Raw yield changes don't produce the pattern
   - Scaling factor not documented

3. **Threshold Adjustment**: Need ~0.161, not 0.160
   - At exactly 0.160, high-debt effect is significant (p=0.028)
   - At 0.161, high-debt effect becomes insignificant (p=0.105)

## Possible Explanations for N Discrepancy

1. **Different data frequency**: Paper may use a non-standard frequency
2. **Different FOMC date list**: May exclude some meetings
3. **Different threshold variable**: May use different fiscal measure
4. **Typo in reported N**: Sample sizes may be incorrectly reported
5. **Pooled regression**: May use interaction terms instead of split sample

## Files Created

- `simple_search.py` - Quick parameter sweep
- `final_match.py` - Focused search around best parameters
- `refine_match.py` - Fine-grained refinement
- `high_coef_search.py` - Search for high-debt coefficient match
- `n_match_search.py` - Search for exact N=40/23

## Conclusion

The paper's **qualitative results** (significant low-debt effect, insignificant high-debt effect, ~60% attenuation) can be reproduced with:
- Monthly data from March 2009 to early 2015
- Threshold ≈ 0.161 (not 0.160)
- Shock scaling factor ≈ 5.5-6.0

### What CAN be matched:
1. **Low-debt coefficient (-9.4 bps)**: Exact match with scale=5.9, thresh=0.161
2. **High-debt coefficient (-3.5 bps)**: Exact match with scale=5.5, thresh=0.161
3. **Significance pattern**: Low significant (p<0.05), high not significant (p>0.10)
4. **Attenuation (~60%)**: Close to paper's 63%

### What CANNOT be matched:
1. **Sample sizes (N=40/23)**: No specification produces both N=40/23 AND the correct significance pattern
   - With N=40/23, the significance pattern is REVERSED (high significant, low not)
2. **Exact threshold (0.160)**: At threshold=0.160, high-debt effect is significant (p=0.028)
   - Need threshold ≈0.161 for high-debt to be insignificant

### Possible explanations:
1. A different (undocumented) methodology was used
2. The reported sample sizes are incorrect
3. A different data source or frequency was used
4. The threshold was rounded from 0.161 to 0.160 for presentation
