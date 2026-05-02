#!/usr/bin/env python3
"""
Analyze and impute missing profit values using financial relationships and ratios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_missing_profit():
    """Load dataset and analyze missing profit cases."""
    print("🔍 ANALYZING MISSING PROFIT VALUES")
    print("=" * 50)
    
    # Load dataset
    df = pd.read_csv('OPTIMIZED_PD_MODEL_ENHANCED/data/train-2025.csv', low_memory=False)
    
    # Find rows with missing profit
    missing_profit = df[df['profit'].isnull()].copy()
    print(f"Total missing profit values: {len(missing_profit)}")
    
    # Show available financial data for these cases
    print("\n📊 AVAILABLE FINANCIAL DATA FOR MISSING PROFIT CASES:")
    print("-" * 60)
    
    # Key columns for profit calculation
    profit_related_cols = [
        'id', 'fs_year', 'rev_operating', 'COGS', 'prof_operations', 
        'inc_financing', 'exp_financing', 'prof_financing', 
        'inc_extraord', 'taxes', 'profit', 'ebitda', 'roa', 'roe',
        'asst_tot', 'eqty_tot'
    ]
    
    missing_subset = missing_profit[profit_related_cols].copy()
    print(missing_subset.to_string(index=False))
    
    # Check data availability for imputation
    print(f"\n🎯 DATA AVAILABILITY FOR IMPUTATION:")
    print("-" * 40)
    
    available_data = {}
    for col in ['prof_operations', 'inc_financing', 'exp_financing', 'prof_financing', 
                'inc_extraord', 'taxes', 'ebitda', 'roa', 'roe']:
        available = missing_profit[col].notna().sum()
        available_data[col] = available
        print(f"{col:20}: {available:2d}/{len(missing_profit):2d} available")
    
    return df, missing_profit, missing_subset

def calculate_profit_relationships(df):
    """Analyze relationships between profit and other financial variables."""
    print(f"\n🧮 PROFIT CALCULATION RELATIONSHIPS:")
    print("-" * 45)
    
    # Remove rows with missing profit for analysis
    clean_df = df.dropna(subset=['profit']).copy()
    
    # Method 1: Operating Profit + Financing + Extraordinary - Taxes
    clean_df['calc_profit_method1'] = (
        clean_df['prof_operations'].fillna(0) + 
        clean_df['prof_financing'].fillna(0) + 
        clean_df['inc_extraord'].fillna(0) - 
        clean_df['taxes'].fillna(0)
    )
    
    # Method 2: EBITDA-based estimation
    # First establish EBITDA to profit relationship
    ebitda_profit_corr = clean_df[['ebitda', 'profit']].corr().iloc[0,1]
    print(f"EBITDA-Profit correlation: {ebitda_profit_corr:.4f}")
    
    # Method 3: ROA-based calculation (ROA = Net Income / Total Assets)
    clean_df['calc_profit_roa'] = clean_df['roa'] * clean_df['asst_tot'] / 100
    
    # Method 4: ROE-based calculation (ROE = Net Income / Equity)
    clean_df['calc_profit_roe'] = clean_df['roe'] * clean_df['eqty_tot'] / 100
    
    # Compare methods
    methods = ['calc_profit_method1', 'calc_profit_roa', 'calc_profit_roe']
    
    print(f"\n📈 METHOD ACCURACY COMPARISON:")
    print("-" * 35)
    
    for method in methods:
        if method in clean_df.columns:
            # Calculate correlation with actual profit
            mask = clean_df[['profit', method]].notna().all(axis=1)
            if mask.sum() > 100:  # Need sufficient data points
                corr = clean_df.loc[mask, ['profit', method]].corr().iloc[0,1]
                mae = np.abs(clean_df.loc[mask, 'profit'] - clean_df.loc[mask, method]).mean()
                print(f"{method:20}: Corr={corr:.4f}, MAE={mae:,.0f}")
    
    return clean_df

def impute_missing_profits(df, missing_profit):
    """Impute missing profit values using best available method."""
    print(f"\n🔧 IMPUTING MISSING PROFIT VALUES:")
    print("-" * 35)
    
    imputed_values = []
    methods_used = []
    
    for idx, row in missing_profit.iterrows():
        profit_estimate = None
        method_used = "None"
        
        # Method 1: ROA-based (most reliable for profitability ratios)
        if pd.notna(row['roa']) and pd.notna(row['asst_tot']):
            profit_estimate = row['roa'] * row['asst_tot'] / 100
            method_used = "ROA-based"
        
        # Method 2: ROE-based (if ROA not available)
        elif pd.notna(row['roe']) and pd.notna(row['eqty_tot']):
            profit_estimate = row['roe'] * row['eqty_tot'] / 100  
            method_used = "ROE-based"
        
        # Method 3: Component-based calculation
        elif (pd.notna(row['prof_operations']) or pd.notna(row['ebitda'])):
            components = 0
            if pd.notna(row['prof_operations']):
                components += row['prof_operations']
            elif pd.notna(row['ebitda']):
                # Rough approximation: operating profit ≈ EBITDA * 0.7
                components += row['ebitda'] * 0.7
                
            # Add/subtract other components if available
            if pd.notna(row['prof_financing']):
                components += row['prof_financing']
            if pd.notna(row['inc_extraord']):
                components += row['inc_extraord']
            if pd.notna(row['taxes']):
                components -= row['taxes']
            
            profit_estimate = components
            method_used = "Component-based"
        
        # Method 4: Industry/sector median (last resort)
        else:
            # Use sector median profit margin if available
            sector_median = df.groupby('ateco_sector')['profit'].median()
            if row['ateco_sector'] in sector_median.index:
                if pd.notna(row['rev_operating']):
                    # Estimate using sector median profit margin
                    sector_profit_margin = sector_median[row['ateco_sector']] / df.groupby('ateco_sector')['rev_operating'].median()[row['ateco_sector']]
                    profit_estimate = row['rev_operating'] * sector_profit_margin
                    method_used = "Sector-based"
        
        imputed_values.append(profit_estimate)
        methods_used.append(method_used)
        
        print(f"ID {row['id']:8d} ({row['fs_year']}): {method_used:15s} -> {profit_estimate:>10,.0f}" if profit_estimate else f"ID {row['id']:8d} ({row['fs_year']}): {'No method':15s} -> {'N/A':>10s}")
    
    return imputed_values, methods_used

def main():
    """Main execution function."""
    # Load and analyze data
    df, missing_profit, missing_subset = load_and_analyze_missing_profit()
    
    # Analyze profit calculation relationships
    clean_df = calculate_profit_relationships(df)
    
    # Impute missing values
    imputed_values, methods_used = impute_missing_profits(df, missing_profit)
    
    # Create summary report
    print(f"\n📋 IMPUTATION SUMMARY:")
    print("-" * 25)
    
    method_counts = pd.Series(methods_used).value_counts()
    for method, count in method_counts.items():
        print(f"{method:15s}: {count:2d} cases")
    
    # Show successful imputations
    successful_imputations = [v for v in imputed_values if v is not None]
    if successful_imputations:
        print(f"\n✅ Successfully imputed {len(successful_imputations)}/{len(missing_profit)} missing values")
        print(f"   Range: {min(successful_imputations):,.0f} to {max(successful_imputations):,.0f}")
        print(f"   Mean:  {np.mean(successful_imputations):,.0f}")
    
    # Create updated dataset
    df_updated = df.copy()
    missing_indices = missing_profit.index
    
    for i, (idx, imputed_val) in enumerate(zip(missing_indices, imputed_values)):
        if imputed_val is not None:
            df_updated.loc[idx, 'profit'] = imputed_val
            # Add a flag to track imputed values
            if 'profit_imputed' not in df_updated.columns:
                df_updated['profit_imputed'] = False
            df_updated.loc[idx, 'profit_imputed'] = True
    
    # Save updated dataset
    output_path = 'OPTIMIZED_PD_MODEL_ENHANCED/data/train-2025_profit_imputed.csv'
    df_updated.to_csv(output_path, index=False)
    print(f"\n💾 Updated dataset saved to: {output_path}")
    print(f"   Original missing profit values: {df['profit'].isnull().sum()}")
    print(f"   Remaining missing values: {df_updated['profit'].isnull().sum()}")
    
    return df_updated

if __name__ == "__main__":
    df_updated = main()
