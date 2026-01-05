"""
Enhanced Data Construction Module

Implements the detailed data construction procedures described in the paper,
including specific FRED series codes, missing value handling, and composite measures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class EnhancedDataConstructor:
    """
    Enhanced data construction following paper specifications
    """
    
    def __init__(self):
        """Initialize the data constructor"""
        self.fred_series_codes = {
            # QE and Treasury data
            'fed_treasury_holdings': 'TREAST',
            'total_treasury_debt': 'GFDEBTN',
            'ten_year_yield': 'GS10',
            'corporate_aaa': 'AAA',
            'corporate_baa': 'BAA',
            'mortgage_30y': 'MORTGAGE30US',
            
            # Fiscal variables
            'federal_interest_payments': 'A091RC1Q027SBEA',
            'federal_revenues': 'FGRECPT',
            
            # Confidence measures
            'consumer_confidence': 'UMCSENT',
            'business_confidence': 'BSCICP03USM665S',
            'financial_stress': 'NFCI',
            
            # Investment and real variables
            'private_investment': 'GPDIC1',
            'gdp_deflator': 'GDPDEF',
            
            # Exchange rates
            'dollar_index': 'DTWEXBGS',
            'eur_usd': 'DEXUSEU',
            'jpy_usd': 'DEXJPUS',
            'gbp_usd': 'DEXUSUK',
            'chf_usd': 'DEXSZUS'
        }
        
        self.constructed_variables = {}
        
    def construct_qe_intensity(self, 
                             fed_holdings: pd.Series,
                             total_debt: pd.Series) -> pd.Series:
        """
        Construct QE intensity measure as described in paper
        
        Parameters:
        -----------
        fed_holdings : pd.Series
            Federal Reserve Treasury holdings (TREAST)
        total_debt : pd.Series
            Total outstanding Treasury debt (GFDEBTN)
            
        Returns:
        --------
        pd.Series
            QE intensity measure (gamma_t)
        """
        # Handle missing values in early 2008 with linear interpolation
        fed_holdings_filled = fed_holdings.interpolate(method='linear', limit=3)
        total_debt_filled = total_debt.interpolate(method='linear', limit=3)
        
        # Calculate ratio
        qe_intensity = fed_holdings_filled / total_debt_filled
        
        # Ensure non-negative and bounded
        qe_intensity = qe_intensity.clip(lower=0, upper=1)
        
        return qe_intensity
    
    def construct_debt_service_burden(self,
                                    interest_payments: pd.Series,
                                    federal_revenues: pd.Series) -> pd.Series:
        """
        Construct debt service burden measure (d_t)
        
        Parameters:
        -----------
        interest_payments : pd.Series
            Federal interest payments (quarterly, A091RC1Q027SBEA)
        federal_revenues : pd.Series
            Federal revenues (monthly, FGRECPT)
            
        Returns:
        --------
        pd.Series
            Debt service burden ratio
        """
        # Convert quarterly interest payments to monthly using cubic spline
        if interest_payments.index.freq != 'M':
            # Create monthly index
            monthly_index = pd.date_range(
                start=interest_payments.index.min(),
                end=interest_payments.index.max(),
                freq='M'
            )
            
            # Interpolate using cubic spline
            f = interpolate.interp1d(
                interest_payments.index.astype(np.int64),
                interest_payments.values,
                kind='cubic',
                fill_value='extrapolate'
            )
            
            interest_monthly = pd.Series(
                f(monthly_index.astype(np.int64)),
                index=monthly_index
            )
        else:
            interest_monthly = interest_payments
        
        # Align indices and calculate ratio
        common_index = interest_monthly.index.intersection(federal_revenues.index)
        
        debt_service_burden = (
            interest_monthly.loc[common_index] / 
            federal_revenues.loc[common_index]
        )
        
        return debt_service_burden
    
    def construct_confidence_composite(self,
                                     consumer_conf: pd.Series,
                                     business_conf: pd.Series,
                                     financial_stress: pd.Series,
                                     method: str = 'equal_weight') -> pd.Series:
        """
        Construct confidence composite measure (lambda_t)
        
        Parameters:
        -----------
        consumer_conf : pd.Series
            Consumer confidence (UMCSENT)
        business_conf : pd.Series
            Business confidence (BSCICP03USM665S)
        financial_stress : pd.Series
            Financial stress index (NFCI, inverted)
        method : str, default='equal_weight'
            Method for combining: 'equal_weight' or 'pca'
            
        Returns:
        --------
        pd.Series
            Confidence composite measure
        """
        # Align all series to common index
        common_index = (consumer_conf.index
                       .intersection(business_conf.index)
                       .intersection(financial_stress.index))
        
        # Extract aligned data
        cons_aligned = consumer_conf.loc[common_index]
        bus_aligned = business_conf.loc[common_index]
        stress_aligned = -financial_stress.loc[common_index]  # Invert stress index
        
        # Standardize all components
        scaler = StandardScaler()
        components = np.column_stack([
            scaler.fit_transform(cons_aligned.values.reshape(-1, 1)).flatten(),
            scaler.fit_transform(bus_aligned.values.reshape(-1, 1)).flatten(),
            scaler.fit_transform(stress_aligned.values.reshape(-1, 1)).flatten()
        ])
        
        if method == 'equal_weight':
            # Equal weighting
            confidence_composite = np.mean(components, axis=1)
        elif method == 'pca':
            # PCA-based weighting
            pca = PCA(n_components=1)
            confidence_composite = pca.fit_transform(components).flatten()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return pd.Series(confidence_composite, index=common_index)
    
    def construct_liquidity_component(self,
                                    bid_ask_spreads: Dict[str, pd.Series]) -> pd.Series:
        """
        Construct liquidity component from Treasury bid-ask spreads.
        
        Parameters:
        -----------
        bid_ask_spreads : Dict[str, pd.Series]
            Dictionary with keys like '2Y', '5Y', '10Y', '30Y' containing
            bid-ask spread series for different maturities
            
        Returns:
        --------
        pd.Series
            Standardized liquidity component (Liqt)
        """
        if not bid_ask_spreads:
            raise ValueError("bid_ask_spreads dictionary cannot be empty")
        
        # Align all spread series to common index
        common_index = None
        for maturity, spreads in bid_ask_spreads.items():
            if common_index is None:
                common_index = spreads.index
            else:
                common_index = common_index.intersection(spreads.index)
        
        if len(common_index) == 0:
            raise ValueError("No common dates found across bid-ask spread series")
        
        # Extract aligned spreads and compute average across maturities
        aligned_spreads = []
        for maturity, spreads in bid_ask_spreads.items():
            aligned = spreads.loc[common_index]
            # Handle missing values with forward fill (max 2 periods)
            aligned = aligned.fillna(method='ffill', limit=2)
            # If still missing, use backward fill
            aligned = aligned.fillna(method='bfill', limit=2)
            aligned_spreads.append(aligned)
        
        # Average across maturities to get aggregate liquidity measure
        avg_spreads = pd.concat(aligned_spreads, axis=1).mean(axis=1)
        
        # Normalize using StandardScaler
        scaler = StandardScaler()
        liquidity_component = scaler.fit_transform(
            avg_spreads.values.reshape(-1, 1)
        ).flatten()
        
        return pd.Series(liquidity_component, index=common_index, name='liquidity_component')
    
    def construct_balance_sheet_component(self,
                                         dealer_assets: pd.Series,
                                         dealer_capital: pd.Series) -> pd.Series:
        """
        Construct balance sheet utilization from H.4.1 data.
        
        Parameters:
        -----------
        dealer_assets : pd.Series
            Primary dealer total assets from FRBNY Primary Dealer Statistics
        dealer_capital : pd.Series
            Primary dealer capital from FRBNY Primary Dealer Statistics
            
        Returns:
        --------
        pd.Series
            Standardized balance sheet component (BStt)
        """
        # Align both series to common index
        common_index = dealer_assets.index.intersection(dealer_capital.index)
        
        if len(common_index) == 0:
            raise ValueError("No common dates found between dealer assets and capital series")
        
        assets_aligned = dealer_assets.loc[common_index]
        capital_aligned = dealer_capital.loc[common_index]
        
        # Handle missing values with forward fill (max 2 periods)
        assets_aligned = assets_aligned.fillna(method='ffill', limit=2)
        capital_aligned = capital_aligned.fillna(method='ffill', limit=2)
        
        # If still missing, use backward fill
        assets_aligned = assets_aligned.fillna(method='bfill', limit=2)
        capital_aligned = capital_aligned.fillna(method='bfill', limit=2)
        
        # Compute ratio of assets to capital (balance sheet utilization)
        # Avoid division by zero
        balance_sheet_utilization = assets_aligned / capital_aligned.replace(0, np.nan)
        
        # Drop any remaining NaN values
        balance_sheet_utilization = balance_sheet_utilization.dropna()
        
        if len(balance_sheet_utilization) == 0:
            raise ValueError("Balance sheet utilization calculation resulted in no valid data")
        
        # Normalize using StandardScaler
        scaler = StandardScaler()
        balance_sheet_component = scaler.fit_transform(
            balance_sheet_utilization.values.reshape(-1, 1)
        ).flatten()
        
        return pd.Series(balance_sheet_component, 
                        index=balance_sheet_utilization.index, 
                        name='balance_sheet_component')
    
    def construct_concentration_component(self,
                                         dealer_market_shares: pd.DataFrame) -> pd.Series:
        """
        Construct HHI from dealer market shares.
        
        Parameters:
        -----------
        dealer_market_shares : pd.DataFrame
            DataFrame with dealer shares by quarter (rows=dates, cols=dealers)
            Each row should sum to approximately 1.0
            
        Returns:
        --------
        pd.Series
            Standardized concentration component (Conct)
        """
        if dealer_market_shares.empty:
            raise ValueError("dealer_market_shares DataFrame cannot be empty")
        
        # Validate that shares sum to approximately 1 for each period
        row_sums = dealer_market_shares.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=0.01):
            warnings.warn(
                f"Market shares do not sum to 1.0 for all periods. "
                f"Range: [{row_sums.min():.4f}, {row_sums.max():.4f}]. "
                f"Normalizing shares to sum to 1.0."
            )
            # Normalize each row to sum to 1
            dealer_market_shares = dealer_market_shares.div(row_sums, axis=0)
        
        # Calculate HHI for each period: HHI = Σ(share_i²)
        hhi = (dealer_market_shares ** 2).sum(axis=1)
        
        # Validate HHI bounds: 1/N ≤ HHI ≤ 1
        n_dealers = dealer_market_shares.shape[1]
        min_hhi = 1.0 / n_dealers
        
        if (hhi < min_hhi - 1e-6).any() or (hhi > 1.0 + 1e-6).any():
            warnings.warn(
                f"HHI values outside expected bounds [1/N={min_hhi:.4f}, 1.0]. "
                f"Actual range: [{hhi.min():.4f}, {hhi.max():.4f}]"
            )
        
        # Clip to valid range to handle numerical precision issues
        hhi = hhi.clip(lower=min_hhi, upper=1.0)
        
        # Normalize using StandardScaler
        scaler = StandardScaler()
        concentration_component = scaler.fit_transform(
            hhi.values.reshape(-1, 1)
        ).flatten()
        
        return pd.Series(concentration_component, 
                        index=hhi.index, 
                        name='concentration_component')
    
    def construct_market_distortions(self,
                                   bid_ask_spreads: Optional[pd.Series] = None,
                                   dealer_balance_sheet: Optional[pd.Series] = None,
                                   market_concentration: Optional[pd.Series] = None,
                                   # New three-component interface
                                   liquidity_component: Optional[pd.Series] = None,
                                   balance_sheet_component: Optional[pd.Series] = None,
                                   concentration_component: Optional[pd.Series] = None,
                                   use_three_component: bool = True) -> pd.Series:
        """
        Construct market distortions measure (mu_t)
        
        This method supports both the legacy two-component interface and the new
        three-component interface for backward compatibility.
        
        Parameters:
        -----------
        bid_ask_spreads : pd.Series, optional
            [LEGACY] Treasury bid-ask spreads from TRACE data
        dealer_balance_sheet : pd.Series, optional
            [LEGACY] Primary dealer balance sheet utilization
        market_concentration : pd.Series, optional
            [LEGACY] Market concentration measures from SIFMA
        liquidity_component : pd.Series, optional
            [NEW] Pre-computed standardized liquidity component (Liqt)
        balance_sheet_component : pd.Series, optional
            [NEW] Pre-computed standardized balance sheet component (BStt)
        concentration_component : pd.Series, optional
            [NEW] Pre-computed standardized concentration component (Conct)
        use_three_component : bool, default=True
            If True, use new three-component method. If False, use legacy method.
            
        Returns:
        --------
        pd.Series
            Market distortions composite measure (Dt)
        """
        # Determine which interface to use
        new_components_provided = any([
            liquidity_component is not None,
            balance_sheet_component is not None,
            concentration_component is not None
        ])
        
        legacy_components_provided = any([
            bid_ask_spreads is not None,
            dealer_balance_sheet is not None,
            market_concentration is not None
        ])
        
        if new_components_provided and use_three_component:
            # Use new three-component method
            components = []
            component_names = []
            
            if liquidity_component is not None:
                components.append(liquidity_component)
                component_names.append('liquidity')
            
            if balance_sheet_component is not None:
                components.append(balance_sheet_component)
                component_names.append('balance_sheet')
            
            if concentration_component is not None:
                components.append(concentration_component)
                component_names.append('concentration')
            
            if not components:
                raise ValueError("At least one component must be provided")
            
            # Align all components to common index
            common_index = components[0].index
            for comp in components[1:]:
                common_index = common_index.intersection(comp.index)
            
            if len(common_index) == 0:
                raise ValueError("No common dates found across components")
            
            # Validate that components are properly normalized (should have ~0 mean, ~1 std)
            for i, comp in enumerate(components):
                aligned = comp.loc[common_index]
                mean_val = aligned.mean()
                std_val = aligned.std()
                
                if abs(mean_val) > 0.1:
                    warnings.warn(
                        f"Component {component_names[i]} has mean {mean_val:.4f}, "
                        f"expected ~0. Component may not be properly normalized."
                    )
                
                if abs(std_val - 1.0) > 0.2:
                    warnings.warn(
                        f"Component {component_names[i]} has std {std_val:.4f}, "
                        f"expected ~1. Component may not be properly normalized."
                    )
            
            # Extract aligned components
            aligned_components = [comp.loc[common_index] for comp in components]
            
            # Combine as equal-weighted average: Dt = (Liqt + BStt + Conct) / 3
            market_distortions = np.mean(aligned_components, axis=0)
            
            return pd.Series(market_distortions, index=common_index, name='market_distortions')
        
        elif legacy_components_provided:
            # Use legacy two-component method for backward compatibility
            warnings.warn(
                "Using legacy two-component market distortion construction. "
                "Consider using the new three-component method with "
                "construct_liquidity_component(), construct_balance_sheet_component(), "
                "and construct_concentration_component().",
                DeprecationWarning
            )
            
            components = []
            component_names = []
            
            if bid_ask_spreads is not None:
                components.append(bid_ask_spreads)
                component_names.append('bid_ask_spreads')
                
            if dealer_balance_sheet is not None:
                components.append(dealer_balance_sheet)
                component_names.append('dealer_balance_sheet')
                
            if market_concentration is not None:
                components.append(market_concentration)
                component_names.append('market_concentration')
            
            if not components:
                # Create synthetic distortion measure if no data available
                warnings.warn("No market distortion data provided, creating synthetic measure")
                dates = pd.date_range('2008-01-01', '2023-12-31', freq='M')
                synthetic_distortions = pd.Series(
                    np.random.normal(0.1, 0.05, len(dates)),
                    index=dates
                )
                return synthetic_distortions.clip(lower=0)
            
            # Align all components
            common_index = components[0].index
            for comp in components[1:]:
                common_index = common_index.intersection(comp.index)
            
            # Standardize and combine with equal weights
            scaler = StandardScaler()
            standardized_components = []
            
            for comp in components:
                aligned_comp = comp.loc[common_index]
                # Handle missing values with forward fill (max 2 periods)
                aligned_comp = aligned_comp.fillna(method='ffill', limit=2)
                
                standardized = scaler.fit_transform(
                    aligned_comp.values.reshape(-1, 1)
                ).flatten()
                standardized_components.append(standardized)
            
            # Equal-weighted average
            market_distortions = np.mean(standardized_components, axis=0)
            
            return pd.Series(market_distortions, index=common_index)
        
        else:
            raise ValueError(
                "No market distortion components provided. "
                "Provide either new components (liquidity_component, balance_sheet_component, "
                "concentration_component) or legacy components (bid_ask_spreads, "
                "dealer_balance_sheet, market_concentration)."
            )
    
    def construct_investment_growth(self,
                                  private_investment: pd.Series,
                                  gdp_deflator: pd.Series) -> pd.Series:
        """
        Construct real private investment growth (I_t)
        
        Parameters:
        -----------
        private_investment : pd.Series
            Real private fixed investment (GPDIC1)
        gdp_deflator : pd.Series
            GDP deflator (GDPDEF)
            
        Returns:
        --------
        pd.Series
            Investment growth rate (log differences)
        """
        # Ensure both series are aligned
        common_index = private_investment.index.intersection(gdp_deflator.index)
        
        investment_aligned = private_investment.loc[common_index]
        deflator_aligned = gdp_deflator.loc[common_index]
        
        # Real investment (already real in GPDIC1, but double-check deflation)
        real_investment = investment_aligned / (deflator_aligned / 100)
        
        # Calculate log growth rates
        investment_growth = np.log(real_investment).diff()
        
        return investment_growth
    
    def handle_missing_values(self,
                            data: pd.DataFrame,
                            method: str = 'mixed') -> pd.DataFrame:
        """
        Handle missing values using methods described in paper
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with potential missing values
        method : str, default='mixed'
            Method for handling missing values
            
        Returns:
        --------
        pd.DataFrame
            Data with missing values handled
        """
        data_filled = data.copy()
        
        for column in data_filled.columns:
            if data_filled[column].isna().any():
                if method == 'mixed':
                    # Use forward fill for market data (max 2 periods)
                    if any(market_term in column.lower() 
                          for market_term in ['spread', 'yield', 'rate', 'price']):
                        data_filled[column] = data_filled[column].fillna(
                            method='ffill', limit=2
                        )
                    # Use linear interpolation for macro data (max 3 periods)
                    else:
                        data_filled[column] = data_filled[column].interpolate(
                            method='linear', limit=3
                        )
                elif method == 'interpolate':
                    data_filled[column] = data_filled[column].interpolate(
                        method='linear', limit=3
                    )
                elif method == 'forward_fill':
                    data_filled[column] = data_filled[column].fillna(
                        method='ffill', limit=2
                    )
        
        return data_filled
    
    def construct_all_variables(self, raw_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Construct all variables as described in the paper
        
        Parameters:
        -----------
        raw_data : Dict[str, pd.Series]
            Dictionary of raw data series with FRED codes as keys
            
        Returns:
        --------
        pd.DataFrame
            Complete dataset with all constructed variables
        """
        constructed_data = {}
        
        # QE intensity
        if 'TREAST' in raw_data and 'GFDEBTN' in raw_data:
            constructed_data['qe_intensity'] = self.construct_qe_intensity(
                raw_data['TREAST'], raw_data['GFDEBTN']
            )
        
        # Debt service burden
        if 'A091RC1Q027SBEA' in raw_data and 'FGRECPT' in raw_data:
            constructed_data['debt_service_burden'] = self.construct_debt_service_burden(
                raw_data['A091RC1Q027SBEA'], raw_data['FGRECPT']
            )
        
        # Confidence composite
        if all(code in raw_data for code in ['UMCSENT', 'BSCICP03USM665S', 'NFCI']):
            constructed_data['confidence_composite'] = self.construct_confidence_composite(
                raw_data['UMCSENT'], raw_data['BSCICP03USM665S'], raw_data['NFCI']
            )
        
        # Investment growth
        if 'GPDIC1' in raw_data and 'GDPDEF' in raw_data:
            constructed_data['investment_growth'] = self.construct_investment_growth(
                raw_data['GPDIC1'], raw_data['GDPDEF']
            )
        
        # Add raw variables that don't need construction
        for code, series in raw_data.items():
            if code in ['GS10', 'AAA', 'BAA', 'MORTGAGE30US', 'DTWEXBGS', 
                       'DEXUSEU', 'DEXJPUS', 'DEXUSUK', 'DEXSZUS']:
                constructed_data[code] = series
        
        # Combine into DataFrame
        result_df = pd.DataFrame(constructed_data)
        
        # Handle missing values
        result_df = self.handle_missing_values(result_df)
        
        return result_df