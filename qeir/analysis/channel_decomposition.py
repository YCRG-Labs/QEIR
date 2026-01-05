"""
Structural Channel Decomposition for QE Transmission Mechanisms

This module implements the structural decomposition methodology to quantify
the relative importance of interest rate versus market distortion channels
in QE transmission to investment.

Following the two-step approach:
1. Estimate effects of QE on channels (interest rates and distortions)
2. Estimate investment response to each channel
3. Decompose total effects into channel contributions
4. Calculate channel shares

Requirements: 7.1-7.7
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass
class ChannelDecompositionConfig:
    """Configuration for structural channel decomposition"""
    
    # Target channel shares
    target_distortion_share: float = 0.65  # 65% distortion channel
    target_rate_share: float = 0.35  # 35% rate channel
    share_tolerance: float = 0.05  # ±5% tolerance
    
    # Estimation settings
    max_horizon: int = 12  # Maximum horizon for decomposition (quarters)
    use_hac_errors: bool = True  # Use HAC standard errors
    hac_lags: int = 4  # Lags for Newey-West
    
    # Validation settings
    min_observations: int = 20  # Minimum observations required
    validate_shares: bool = True  # Validate shares sum to 1


class StructuralChannelDecomposer:
    """
    Implements structural decomposition of QE transmission channels.
    
    This class performs a two-step structural decomposition to quantify
    the relative importance of interest rate and market distortion channels
    in transmitting QE effects to investment.
    
    The methodology follows:
    Step 1: Estimate effects of QE on channels
        rt = αr + βr·QEt + γr·Zt + εr,t
        Dt = αD + βD·QEt + γD·Zt + εD,t
    
    Step 2: Estimate investment response to channels
        Δ^h It+h = αI,h + ρh·rt + κh·Dt + γI,h·Zt + εI,h,t
    
    Step 3: Decompose total effects
        ψ^rate_h = ρh × βr
        ψ^dist_h = κh × βD
        ψh = ψ^rate_h + ψ^dist_h
    
    Step 4: Calculate channel shares
        Distortion Share = Σ(h=0 to H) ψ^dist_h / Σ(h=0 to H) ψh
    
    Requirements: 7.1, 7.2
    """
    
    def __init__(self, config: Optional[ChannelDecompositionConfig] = None):
        """
        Initialize structural channel decomposer.
        
        Args:
            config: Configuration for channel decomposition
        """
        self.config = config or ChannelDecompositionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for intermediate results
        self.channel_effects_results: Optional[Dict[str, Any]] = None
        self.investment_response_results: Optional[Dict[int, Dict[str, Any]]] = None
        
    def validate_data(self,
                     qe_shocks: pd.Series,
                     interest_rates: pd.Series,
                     distortion_index: pd.Series,
                     investment_growth: pd.Series,
                     controls: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
        """
        Validate input data for channel decomposition.
        
        Args:
            qe_shocks: QE shock series
            interest_rates: Interest rate series
            distortion_index: Market distortion index
            investment_growth: Investment growth series
            controls: Optional control variables
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for None inputs
        if qe_shocks is None or interest_rates is None or distortion_index is None or investment_growth is None:
            return False, "One or more required series is None"
        
        # Check for empty series
        if len(qe_shocks) == 0 or len(interest_rates) == 0 or len(distortion_index) == 0 or len(investment_growth) == 0:
            return False, "One or more series is empty"
        
        # Find common dates
        common_dates = qe_shocks.index.intersection(interest_rates.index)\
                                      .intersection(distortion_index.index)\
                                      .intersection(investment_growth.index)
        
        if controls is not None:
            common_dates = common_dates.intersection(controls.index)
        
        # Check minimum observations
        if len(common_dates) < self.config.min_observations:
            return False, f"Insufficient overlapping observations: {len(common_dates)} < {self.config.min_observations}"
        
        # Check for missing values
        if qe_shocks.loc[common_dates].isnull().any():
            return False, "QE shocks contain missing values"
        
        if interest_rates.loc[common_dates].isnull().any():
            return False, "Interest rates contain missing values"
        
        if distortion_index.loc[common_dates].isnull().any():
            return False, "Distortion index contains missing values"
        
        if investment_growth.loc[common_dates].isnull().any():
            return False, "Investment growth contains missing values"
        
        if controls is not None:
            if isinstance(controls, pd.DataFrame):
                if controls.loc[common_dates].isnull().any().any():
                    return False, "Controls contain missing values"
            else:
                if controls.loc[common_dates].isnull().any():
                    return False, "Controls contain missing values"
        
        # Check for sufficient variation
        if qe_shocks.loc[common_dates].std() < 1e-10:
            return False, "QE shocks have insufficient variation"
        
        if interest_rates.loc[common_dates].std() < 1e-10:
            return False, "Interest rates have insufficient variation"
        
        if distortion_index.loc[common_dates].std() < 1e-10:
            return False, "Distortion index has insufficient variation"
        
        return True, ""

    def estimate_channel_effects(self,
                                 qe_shocks: pd.Series,
                                 interest_rates: pd.Series,
                                 distortion_index: pd.Series,
                                 controls: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Step 1: Estimate effects of QE on transmission channels.
        
        Estimates two regressions:
        1. rt = αr + βr·QEt + γr·Zt + εr,t
        2. Dt = αD + βD·QEt + γD·Zt + εD,t
        
        Where:
        - rt: Interest rates
        - Dt: Market distortion index
        - QEt: QE shocks (instrumented)
        - Zt: Control variables
        
        Args:
            qe_shocks: QE shock series (instrumented)
            interest_rates: Interest rate series
            distortion_index: Market distortion index
            controls: Optional control variables DataFrame
            
        Returns:
            Tuple of (rate_channel_results, distortion_channel_results)
            Each dict contains: beta, se, pvalue, rsquared, observations
            
        Requirements: 7.1, 7.2
        """
        self.logger.info("Step 1: Estimating channel effects")
        
        # Validate data
        is_valid, error_msg = self.validate_data(
            qe_shocks, interest_rates, distortion_index,
            interest_rates,  # Dummy for investment_growth validation
            controls
        )
        
        if not is_valid:
            raise ValueError(f"Data validation failed: {error_msg}")
        
        # Align all series to common dates
        common_dates = qe_shocks.index.intersection(interest_rates.index)\
                                      .intersection(distortion_index.index)
        
        if controls is not None:
            common_dates = common_dates.intersection(controls.index)
        
        qe_aligned = qe_shocks.loc[common_dates]
        rates_aligned = interest_rates.loc[common_dates]
        distortions_aligned = distortion_index.loc[common_dates]
        
        # Prepare control variables
        if controls is not None:
            controls_aligned = controls.loc[common_dates]
            # Combine QE shocks with controls
            X_rate = pd.concat([qe_aligned, controls_aligned], axis=1)
            X_distortion = pd.concat([qe_aligned, controls_aligned], axis=1)
        else:
            X_rate = qe_aligned.to_frame(name='qe_shocks')
            X_distortion = qe_aligned.to_frame(name='qe_shocks')
        
        # Add constant
        X_rate_with_const = sm.add_constant(X_rate)
        X_distortion_with_const = sm.add_constant(X_distortion)
        
        # Regression 1: Interest rates on QE shocks
        # rt = αr + βr·QEt + γr·Zt + εr,t
        self.logger.info("Estimating rate channel: rt = αr + βr·QEt + γr·Zt + εr,t")
        
        if self.config.use_hac_errors:
            rate_model = sm.OLS(rates_aligned.values, X_rate_with_const).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': self.config.hac_lags}
            )
        else:
            rate_model = sm.OLS(rates_aligned.values, X_rate_with_const).fit()
        
        # Extract βr (coefficient on QE shocks - first variable after constant)
        rate_channel_results = {
            'beta': rate_model.params[1],  # βr
            'se': rate_model.bse[1],
            'pvalue': rate_model.pvalues[1],
            'tstat': rate_model.tvalues[1],
            'rsquared': rate_model.rsquared,
            'observations': len(rates_aligned),
            'model': rate_model,
            'equation': 'rt = αr + βr·QEt + γr·Zt + εr,t'
        }
        
        self.logger.info(f"Rate channel βr = {rate_channel_results['beta']:.4f} "
                        f"(se={rate_channel_results['se']:.4f}, p={rate_channel_results['pvalue']:.4f})")
        
        # Regression 2: Distortion index on QE shocks
        # Dt = αD + βD·QEt + γD·Zt + εD,t
        self.logger.info("Estimating distortion channel: Dt = αD + βD·QEt + γD·Zt + εD,t")
        
        if self.config.use_hac_errors:
            distortion_model = sm.OLS(distortions_aligned.values, X_distortion_with_const).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': self.config.hac_lags}
            )
        else:
            distortion_model = sm.OLS(distortions_aligned.values, X_distortion_with_const).fit()
        
        # Extract βD (coefficient on QE shocks - first variable after constant)
        distortion_channel_results = {
            'beta': distortion_model.params[1],  # βD
            'se': distortion_model.bse[1],
            'pvalue': distortion_model.pvalues[1],
            'tstat': distortion_model.tvalues[1],
            'rsquared': distortion_model.rsquared,
            'observations': len(distortions_aligned),
            'model': distortion_model,
            'equation': 'Dt = αD + βD·QEt + γD·Zt + εD,t'
        }
        
        self.logger.info(f"Distortion channel βD = {distortion_channel_results['beta']:.4f} "
                        f"(se={distortion_channel_results['se']:.4f}, p={distortion_channel_results['pvalue']:.4f})")
        
        # Store results
        self.channel_effects_results = {
            'rate_channel': rate_channel_results,
            'distortion_channel': distortion_channel_results
        }
        
        return rate_channel_results, distortion_channel_results

    def estimate_investment_response(self,
                                    investment_growth: pd.Series,
                                    interest_rates: pd.Series,
                                    distortion_index: pd.Series,
                                    controls: Optional[pd.DataFrame] = None,
                                    horizons: Optional[List[int]] = None) -> Dict[int, Dict[str, Any]]:
        """
        Step 2: Estimate investment response to transmission channels.
        
        Estimates local projection regressions for each horizon h:
        Δ^h It+h = αI,h + ρh·rt + κh·Dt + γI,h·Zt + εI,h,t
        
        Where:
        - Δ^h It+h: h-period ahead investment growth
        - rt: Interest rates
        - Dt: Market distortion index
        - Zt: Control variables
        - ρh: Investment response to rate channel at horizon h
        - κh: Investment response to distortion channel at horizon h
        
        Args:
            investment_growth: Investment growth series
            interest_rates: Interest rate series
            distortion_index: Market distortion index
            controls: Optional control variables DataFrame
            horizons: List of horizons to estimate (default: 0 to max_horizon)
            
        Returns:
            Dict mapping horizon to estimation results
            Each result contains: rho (ρh), kappa (κh), and their standard errors
            
        Requirements: 7.3
        """
        self.logger.info("Step 2: Estimating investment response to channels")
        
        # Use default horizons if not specified
        if horizons is None:
            horizons = list(range(self.config.max_horizon + 1))
        
        # Validate data
        is_valid, error_msg = self.validate_data(
            interest_rates,  # Use as dummy for qe_shocks
            interest_rates,
            distortion_index,
            investment_growth,
            controls
        )
        
        if not is_valid:
            raise ValueError(f"Data validation failed: {error_msg}")
        
        # Align all series to common dates
        common_dates = investment_growth.index.intersection(interest_rates.index)\
                                              .intersection(distortion_index.index)
        
        if controls is not None:
            common_dates = common_dates.intersection(controls.index)
        
        investment_aligned = investment_growth.loc[common_dates]
        rates_aligned = interest_rates.loc[common_dates]
        distortions_aligned = distortion_index.loc[common_dates]
        
        # Prepare control variables
        if controls is not None:
            controls_aligned = controls.loc[common_dates]
        else:
            controls_aligned = None
        
        # Storage for results by horizon
        results_by_horizon = {}
        
        # Estimate for each horizon
        for h in horizons:
            self.logger.info(f"Estimating horizon h={h}")
            
            # Create h-period ahead investment growth
            # Δ^h It+h = It+h - It
            investment_h = investment_aligned.shift(-h)
            
            # Drop observations where we don't have future values
            valid_idx = investment_h.dropna().index
            
            if len(valid_idx) < self.config.min_observations:
                self.logger.warning(f"Insufficient observations for horizon {h}: {len(valid_idx)}")
                continue
            
            # Align all variables to valid index
            y_h = investment_h.loc[valid_idx]
            rates_h = rates_aligned.loc[valid_idx]
            distortions_h = distortions_aligned.loc[valid_idx]
            
            # Prepare regressors: rt and Dt
            X_h = pd.DataFrame({
                'interest_rates': rates_h,
                'distortion_index': distortions_h
            })
            
            # Add controls if provided
            if controls_aligned is not None:
                controls_h = controls_aligned.loc[valid_idx]
                X_h = pd.concat([X_h, controls_h], axis=1)
            
            # Add constant
            X_h_with_const = sm.add_constant(X_h)
            
            # Estimate: Δ^h It+h = αI,h + ρh·rt + κh·Dt + γI,h·Zt + εI,h,t
            if self.config.use_hac_errors:
                model_h = sm.OLS(y_h.values, X_h_with_const).fit(
                    cov_type='HAC',
                    cov_kwds={'maxlags': self.config.hac_lags}
                )
            else:
                model_h = sm.OLS(y_h.values, X_h_with_const).fit()
            
            # Extract coefficients
            # ρh: coefficient on interest_rates (first variable after constant)
            # κh: coefficient on distortion_index (second variable after constant)
            rho_h = model_h.params[1]  # ρh
            rho_se = model_h.bse[1]
            rho_pvalue = model_h.pvalues[1]
            
            kappa_h = model_h.params[2]  # κh
            kappa_se = model_h.bse[2]
            kappa_pvalue = model_h.pvalues[2]
            
            # Store results
            results_by_horizon[h] = {
                'horizon': h,
                'rho': rho_h,  # ρh: Investment response to rate channel
                'rho_se': rho_se,
                'rho_pvalue': rho_pvalue,
                'rho_tstat': model_h.tvalues[1],
                'kappa': kappa_h,  # κh: Investment response to distortion channel
                'kappa_se': kappa_se,
                'kappa_pvalue': kappa_pvalue,
                'kappa_tstat': model_h.tvalues[2],
                'rsquared': model_h.rsquared,
                'observations': len(y_h),
                'model': model_h,
                'equation': f'Δ^{h} It+{h} = αI,{h} + ρ{h}·rt + κ{h}·Dt + γI,{h}·Zt + εI,{h},t'
            }
            
            self.logger.info(f"Horizon {h}: ρ{h} = {rho_h:.4f} (se={rho_se:.4f}), "
                           f"κ{h} = {kappa_h:.4f} (se={kappa_se:.4f})")
        
        # Store results
        self.investment_response_results = results_by_horizon
        
        self.logger.info(f"Estimated investment response for {len(results_by_horizon)} horizons")
        
        return results_by_horizon

    def decompose_total_effects(self,
                               channel_effects: Tuple[Dict[str, Any], Dict[str, Any]],
                               investment_responses: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 3: Decompose total effect into channel contributions.
        
        Computes channel-specific contributions using the formulas:
        - Rate channel: ψ^rate_h = ρh × βr
        - Distortion channel: ψ^dist_h = κh × βD
        - Total effect: ψh = ψ^rate_h + ψ^dist_h
        
        Where:
        - ρh: Investment response to rate channel at horizon h (from Step 2)
        - βr: Effect of QE on interest rates (from Step 1)
        - κh: Investment response to distortion channel at horizon h (from Step 2)
        - βD: Effect of QE on distortion index (from Step 1)
        
        Args:
            channel_effects: Tuple of (rate_channel_results, distortion_channel_results) from Step 1
            investment_responses: Dict of investment response results by horizon from Step 2
            
        Returns:
            Dict with decomposition results including:
            - rate_contributions: Dict[int, float] - ψ^rate_h by horizon
            - distortion_contributions: Dict[int, float] - ψ^dist_h by horizon
            - total_effects: Dict[int, float] - ψh by horizon
            
        Requirements: 7.4, 7.5
        """
        self.logger.info("Step 3: Decomposing total effects into channel contributions")
        
        rate_channel_results, distortion_channel_results = channel_effects
        
        # Extract βr and βD from Step 1
        beta_r = rate_channel_results['beta']  # βr: Effect of QE on interest rates
        beta_d = distortion_channel_results['beta']  # βD: Effect of QE on distortion index
        
        self.logger.info(f"Using βr = {beta_r:.4f}, βD = {beta_d:.4f}")
        
        # Initialize storage for contributions
        rate_contributions = {}  # ψ^rate_h
        distortion_contributions = {}  # ψ^dist_h
        total_effects = {}  # ψh
        
        # Compute contributions for each horizon
        for h, response_results in investment_responses.items():
            # Extract ρh and κh from Step 2
            rho_h = response_results['rho']  # ρh: Investment response to rate channel
            kappa_h = response_results['kappa']  # κh: Investment response to distortion channel
            
            # Compute channel contributions
            # ψ^rate_h = ρh × βr
            psi_rate_h = rho_h * beta_r
            
            # ψ^dist_h = κh × βD
            psi_dist_h = kappa_h * beta_d
            
            # Total effect: ψh = ψ^rate_h + ψ^dist_h
            psi_h = psi_rate_h + psi_dist_h
            
            # Store contributions
            rate_contributions[h] = psi_rate_h
            distortion_contributions[h] = psi_dist_h
            total_effects[h] = psi_h
            
            self.logger.info(f"Horizon {h}: ψ^rate_{h} = {psi_rate_h:.4f}, "
                           f"ψ^dist_{h} = {psi_dist_h:.4f}, ψ{h} = {psi_h:.4f}")
        
        # Compile decomposition results
        decomposition_results = {
            'rate_contributions': rate_contributions,  # ψ^rate_h by horizon
            'distortion_contributions': distortion_contributions,  # ψ^dist_h by horizon
            'total_effects': total_effects,  # ψh by horizon
            'beta_r': beta_r,  # βr from Step 1
            'beta_d': beta_d,  # βD from Step 1
            'horizons': sorted(investment_responses.keys())
        }
        
        self.logger.info(f"Decomposed effects for {len(investment_responses)} horizons")
        
        return decomposition_results

    def calculate_channel_shares(self,
                                 rate_contributions: Dict[int, float],
                                 distortion_contributions: Dict[int, float],
                                 horizon: Optional[int] = None) -> Dict[str, float]:
        """
        Step 4: Calculate channel shares from cumulative contributions.
        
        Computes the share of total QE effect attributable to each channel:
        - Distortion Share = Σ(h=0 to H) ψ^dist_h / Σ(h=0 to H) ψh
        - Rate Share = Σ(h=0 to H) ψ^rate_h / Σ(h=0 to H) ψh
        
        Validates that:
        1. Shares sum to approximately 1.0 (within tolerance)
        2. Distortion share is approximately 65% (within ±5%)
        3. Rate share is approximately 35% (within ±5%)
        
        Args:
            rate_contributions: Dict mapping horizon to ψ^rate_h
            distortion_contributions: Dict mapping horizon to ψ^dist_h
            horizon: Maximum horizon to include (default: config.max_horizon)
            
        Returns:
            Dict with:
            - distortion_share: Share of effect from distortion channel
            - rate_share: Share of effect from rate channel
            - cumulative_rate_effect: Σ ψ^rate_h
            - cumulative_distortion_effect: Σ ψ^dist_h
            - cumulative_total_effect: Σ ψh
            - shares_sum_to_one: bool
            - meets_target_shares: bool
            
        Requirements: 7.6
        """
        self.logger.info("Step 4: Calculating channel shares")
        
        # Use default horizon if not specified
        if horizon is None:
            horizon = self.config.max_horizon
        
        # Get horizons to include (0 to H)
        horizons_to_include = [h for h in rate_contributions.keys() if h <= horizon]
        
        if len(horizons_to_include) == 0:
            raise ValueError(f"No horizons available up to horizon {horizon}")
        
        self.logger.info(f"Computing shares over horizons 0 to {horizon} ({len(horizons_to_include)} horizons)")
        
        # Compute cumulative effects
        # Σ(h=0 to H) ψ^rate_h
        cumulative_rate_effect = sum(rate_contributions[h] for h in horizons_to_include)
        
        # Σ(h=0 to H) ψ^dist_h
        cumulative_distortion_effect = sum(distortion_contributions[h] for h in horizons_to_include)
        
        # Σ(h=0 to H) ψh = Σ(h=0 to H) (ψ^rate_h + ψ^dist_h)
        cumulative_total_effect = cumulative_rate_effect + cumulative_distortion_effect
        
        self.logger.info(f"Cumulative rate effect: {cumulative_rate_effect:.4f}")
        self.logger.info(f"Cumulative distortion effect: {cumulative_distortion_effect:.4f}")
        self.logger.info(f"Cumulative total effect: {cumulative_total_effect:.4f}")
        
        # Calculate shares
        if abs(cumulative_total_effect) < 1e-10:
            self.logger.warning("Cumulative total effect is near zero, cannot compute shares")
            return {
                'distortion_share': 0.0,
                'rate_share': 0.0,
                'cumulative_rate_effect': cumulative_rate_effect,
                'cumulative_distortion_effect': cumulative_distortion_effect,
                'cumulative_total_effect': cumulative_total_effect,
                'shares_sum_to_one': False,
                'meets_target_shares': False,
                'horizons_included': horizons_to_include
            }
        
        # Distortion Share = Σ(h=0 to H) ψ^dist_h / Σ(h=0 to H) ψh
        distortion_share = cumulative_distortion_effect / cumulative_total_effect
        
        # Rate Share = Σ(h=0 to H) ψ^rate_h / Σ(h=0 to H) ψh
        rate_share = cumulative_rate_effect / cumulative_total_effect
        
        self.logger.info(f"Distortion share: {distortion_share:.2%}")
        self.logger.info(f"Rate share: {rate_share:.2%}")
        
        # Validation 1: Shares sum to 1.0 (within tolerance)
        shares_sum = distortion_share + rate_share
        shares_sum_to_one = abs(shares_sum - 1.0) < 0.01
        
        if not shares_sum_to_one:
            self.logger.warning(f"Shares do not sum to 1.0: {shares_sum:.4f}")
        else:
            self.logger.info(f"✓ Shares sum to 1.0: {shares_sum:.4f}")
        
        # Validation 2: Check if shares meet targets (65% distortion, 35% rate, ±5%)
        if self.config.validate_shares:
            distortion_target = self.config.target_distortion_share
            rate_target = self.config.target_rate_share
            tolerance = self.config.share_tolerance
            
            distortion_in_range = abs(distortion_share - distortion_target) <= tolerance
            rate_in_range = abs(rate_share - rate_target) <= tolerance
            
            meets_target_shares = distortion_in_range and rate_in_range
            
            if meets_target_shares:
                self.logger.info(f"✓ Shares meet targets: distortion={distortion_share:.2%} "
                               f"(target={distortion_target:.2%}±{tolerance:.2%}), "
                               f"rate={rate_share:.2%} (target={rate_target:.2%}±{tolerance:.2%})")
            else:
                self.logger.warning(f"Shares outside target range: distortion={distortion_share:.2%} "
                                  f"(target={distortion_target:.2%}±{tolerance:.2%}), "
                                  f"rate={rate_share:.2%} (target={rate_target:.2%}±{tolerance:.2%})")
        else:
            meets_target_shares = True  # Skip validation if disabled
        
        # Compile results
        results = {
            'distortion_share': distortion_share,
            'rate_share': rate_share,
            'cumulative_rate_effect': cumulative_rate_effect,
            'cumulative_distortion_effect': cumulative_distortion_effect,
            'cumulative_total_effect': cumulative_total_effect,
            'shares_sum_to_one': shares_sum_to_one,
            'meets_target_shares': meets_target_shares,
            'horizons_included': horizons_to_include,
            'max_horizon': horizon
        }
        
        return results

    def run_full_decomposition(self,
                              qe_shocks: pd.Series,
                              interest_rates: pd.Series,
                              distortion_index: pd.Series,
                              investment_growth: pd.Series,
                              controls: Optional[pd.DataFrame] = None,
                              horizons: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run complete structural channel decomposition (all 4 steps).
        
        This is a convenience method that executes all four steps:
        1. Estimate channel effects (QE → rates, QE → distortions)
        2. Estimate investment responses (rates → investment, distortions → investment)
        3. Decompose total effects into channel contributions
        4. Calculate channel shares
        
        Args:
            qe_shocks: QE shock series (instrumented)
            interest_rates: Interest rate series
            distortion_index: Market distortion index
            investment_growth: Investment growth series
            controls: Optional control variables DataFrame
            horizons: List of horizons to estimate (default: 0 to max_horizon)
            
        Returns:
            Dict with complete decomposition results including:
            - channel_effects: Results from Step 1
            - investment_responses: Results from Step 2
            - decomposition: Results from Step 3
            - channel_shares: Results from Step 4
        """
        self.logger.info("Running full structural channel decomposition")
        
        # Step 1: Estimate channel effects
        rate_channel, distortion_channel = self.estimate_channel_effects(
            qe_shocks=qe_shocks,
            interest_rates=interest_rates,
            distortion_index=distortion_index,
            controls=controls
        )
        
        # Step 2: Estimate investment responses
        investment_responses = self.estimate_investment_response(
            investment_growth=investment_growth,
            interest_rates=interest_rates,
            distortion_index=distortion_index,
            controls=controls,
            horizons=horizons
        )
        
        # Step 3: Decompose total effects
        decomposition = self.decompose_total_effects(
            channel_effects=(rate_channel, distortion_channel),
            investment_responses=investment_responses
        )
        
        # Step 4: Calculate channel shares
        channel_shares = self.calculate_channel_shares(
            rate_contributions=decomposition['rate_contributions'],
            distortion_contributions=decomposition['distortion_contributions'],
            horizon=self.config.max_horizon
        )
        
        # Compile complete results
        full_results = {
            'channel_effects': {
                'rate_channel': rate_channel,
                'distortion_channel': distortion_channel
            },
            'investment_responses': investment_responses,
            'decomposition': decomposition,
            'channel_shares': channel_shares,
            'config': self.config
        }
        
        self.logger.info("Full structural channel decomposition completed")
        self.logger.info(f"Final shares: Distortion={channel_shares['distortion_share']:.2%}, "
                        f"Rate={channel_shares['rate_share']:.2%}")
        
        return full_results
