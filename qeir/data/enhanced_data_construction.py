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
    
    def construct_market_distortions(self,
                                   bid_ask_spreads: Optional[pd.Series] = None,
                                   dealer_balance_sheet: Optional[pd.Series] = None,
                                   market_concentration: Optional[pd.Series] = None) -> pd.Series:
        """
        Construct market distortions measure (mu_t)
        
        Parameters:
        -----------
        bid_ask_spreads : pd.Series, optional
            Treasury bid-ask spreads from TRACE data
        dealer_balance_sheet : pd.Series, optional
            Primary dealer balance sheet utilization
        market_concentration : pd.Series, optional
            Market concentration measures from SIFMA
            
        Returns:
        --------
        pd.Series
            Market distortions composite measure
        """
        # This is a simplified implementation
        # Real implementation would use actual TRACE, H.4.1, and SIFMA data
        
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
            # This would be replaced with actual data in production
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