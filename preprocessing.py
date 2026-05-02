#!/usr/bin/env python3
"""
OPTIMIZED Preprocessing Pipeline for PD Model (v3.0)
Streamlined to 12+3 high-impact features based on statistical significance
All features computed from accounting relationships
Missing values: If can compute from components → compute, else → zero (conservative)
NO arbitrary constants, NO median/percentile imputation, ONLY financial logic

VERSION: 3.0 - Optimized to 15 features (12 proven + 3 new strategic additions)
REMOVED: 7 non-significant/multicollinear features
ADDED: 3 strategic features (tangible asset ratio, debt maturity, log assets)
"""

import pandas as pd
import numpy as np


class Preprocessing_Pipeline:
    @staticmethod
    def train_preprocessing(df):
        """
        TRAIN: Fit preprocessing parameters and transform data with optimized features
        Returns (transformed_df, preprocessing_details)
        """
        df = df.copy()

        # ===================== CORE COMPUTATIONS =====================
        df["asst_tot_approx"] = (
            df["asst_intang_fixed"].fillna(0) +
            df["asst_tang_fixed"].fillna(0) +
            df["asst_fixed_fin"].fillna(0) +
            df["asst_current"].fillna(0)
        )
        df["asst_tot"] = df["asst_tot"].fillna(df["asst_tot_approx"])

        df["liab_tot_approx"] = (
            df["liab_lt"].fillna(0) +
            df["debt_bank_st"].fillna(0) +
            df["debt_bank_lt"].fillna(0) +
            df["debt_fin_st"].fillna(0) +
            df["debt_fin_lt"].fillna(0) +
            df["AP_st"].fillna(0) +
            df["AP_lt"].fillna(0) +
            df["liab_lt_emp"].fillna(0)
        )
        df["eqty_tot"] = df["eqty_tot"].fillna(df["asst_tot"] - df["liab_tot_approx"])

        # Core 12 features (proven)
        
        # FEATURE 1: LEVERAGE_W (Core leverage) - z=17.46, p<0.001
        leverage = df["liab_tot_approx"] / df["asst_tot"] * 100
        leverage = leverage.mask(df["asst_tot"] == 0, pd.NA).fillna(0)
        leverage_p1 = np.percentile(leverage.dropna(), 1)
        leverage_p99 = np.percentile(leverage.dropna(), 99)
        df["leverage_w"] = leverage.clip(lower=leverage_p1, upper=leverage_p99)

        # FEATURE 2: ROE_W (Profitability) - z=17.99, p<0.001
        roe = df["prof_operations"] / df["eqty_tot"] * 100
        roe = roe.mask(df["eqty_tot"] == 0, pd.NA)
        roe = roe.mask((df["prof_operations"] == 0) & (df["eqty_tot"] != 0), 0.0)
        roe_calc = roe.round(2)
        missing_roe = df["roe"].isna().astype(int)
        roe_merged = df["roe"].fillna(roe_calc).fillna(0)
        roe_p1 = np.percentile(roe_merged.dropna(), 1)
        roe_p99 = np.percentile(roe_merged.dropna(), 99)
        df["roe_w"] = roe_merged.clip(lower=roe_p1, upper=roe_p99)

        # FEATURE 3: ROA_W (Asset efficiency) - z=-17.73, p<0.001
        roa = df["prof_operations"] / df["asst_tot"] * 100
        roa = roa.mask(df["asst_tot"] == 0, pd.NA)
        roa = roa.mask((df["prof_operations"] == 0) & (df["asst_tot"] != 0), 0.0)
        roa_calc = roa.round(2)
        roa_merged = df["roa"].fillna(roa_calc).fillna(0)
        roa_p1 = np.percentile(roa_merged.dropna(), 1)
        roa_p99 = np.percentile(roa_merged.dropna(), 99)
        df["roa_w"] = roa_merged.clip(lower=roa_p1, upper=roa_p99)

        # FEATURE 4: CASH_TO_ASSETS_W (Liquidity buffer) - z=-1.87, p=0.062
        cash_to_assets = df["cash_and_equiv"].fillna(0) / df["asst_tot"] * 100
        cash_to_assets = cash_to_assets.mask(df["asst_tot"] == 0, pd.NA).fillna(0)
        cta_p1 = np.percentile(cash_to_assets.dropna(), 1)
        cta_p99 = np.percentile(cash_to_assets.dropna(), 99)
        df["cash_to_assets_w"] = cash_to_assets.clip(lower=cta_p1, upper=cta_p99)

        # FEATURE 5: ASSET_TURNOVER_W (Efficiency) - z=4.34, p<0.001
        asset_turnover = df["rev_operating"].fillna(0) / df["asst_tot"]
        asset_turnover = asset_turnover.mask(df["asst_tot"] == 0, pd.NA).fillna(0)
        at_p1 = np.percentile(asset_turnover.dropna(), 1)
        at_p99 = np.percentile(asset_turnover.dropna(), 99)
        df["asset_turnover_w"] = asset_turnover.clip(lower=at_p1, upper=at_p99)

        # FEATURE 6: INTEREST_COVERAGE_W (Debt service ability) - z=1.75, p=0.081
        interest_exp = df["exp_financing"].fillna(0)
        ebitda = df["ebitda"].fillna(0)
        interest_coverage = ebitda / interest_exp.replace(0, np.nan)
        interest_coverage = interest_coverage.fillna(0)
        ic_p1 = np.percentile(interest_coverage.dropna(), 1)
        ic_p99 = np.percentile(interest_coverage.dropna(), 99)
        df["interest_coverage_w"] = interest_coverage.clip(lower=ic_p1, upper=ic_p99)

        # FEATURE 7: EBITDA_MARGIN_W (Profitability margin) - z=-1.07, p=0.286
        revenue = df["rev_operating"].fillna(0)
        ebitda_margin = (df["ebitda"].fillna(0) / revenue.replace(0, np.nan)) * 100
        ebitda_margin = ebitda_margin.fillna(0)
        em_p1 = np.percentile(ebitda_margin.dropna(), 1)
        em_p99 = np.percentile(ebitda_margin.dropna(), 99)
        df["ebitda_margin_w"] = ebitda_margin.clip(lower=em_p1, upper=em_p99)

        # FEATURE 8: WORKING_CAPITAL_RATIO_W (Operational efficiency) - z=-10.65, p<0.001
        working_capital = df["wc_net"].fillna(0)
        wc_ratio = working_capital / df["asst_tot"] * 100
        wc_ratio = wc_ratio.mask(df["asst_tot"] == 0, pd.NA).fillna(0)
        wcr_p1 = np.percentile(wc_ratio.dropna(), 1)
        wcr_p99 = np.percentile(wc_ratio.dropna(), 99)
        df["working_capital_ratio_w"] = wc_ratio.clip(lower=wcr_p1, upper=wcr_p99)

        # FEATURE 9: MISSING_ROE_FLAG (Missing data pattern) - z=39.08, p<0.001 ⭐⭐⭐
        df["missing_roe_flag"] = missing_roe

        # FEATURE 10: CFROA_W (Cash Flow Return on Assets) - z=-31.03, p<0.001 ⭐⭐⭐
        cfroa = (df["cf_operations"].fillna(0) / df["asst_tot"].replace(0, np.nan)) * 100
        cfroa = cfroa.fillna(0)
        cfroa_p1 = np.percentile(cfroa.dropna(), 1)
        cfroa_p99 = np.percentile(cfroa.dropna(), 99)
        df["cfroa_w"] = cfroa.clip(lower=cfroa_p1, upper=cfroa_p99)

        # FEATURE 11: AR_TURNOVER_W (Receivables Turnover) - z=-7.42, p<0.001
        ar_turnover = df["rev_operating"].fillna(0) / df["AR"].replace(0, np.nan)
        ar_turnover = ar_turnover.fillna(0)
        art_p1 = np.percentile(ar_turnover.dropna(), 1)
        art_p99 = np.percentile(ar_turnover.dropna(), 99)
        df["ar_turnover_w"] = ar_turnover.clip(lower=art_p1, upper=art_p99)

        # FEATURE 12: FIN_DEBT_RATIO_W (Financial Debt to Assets) - z=-6.47, p<0.001
        financial_debt = (
            df["debt_bank_st"].fillna(0) + 
            df["debt_bank_lt"].fillna(0) + 
            df["debt_fin_st"].fillna(0) + 
            df["debt_fin_lt"].fillna(0)
        )
        fin_debt_ratio = (financial_debt / df["asst_tot"].replace(0, np.nan)) * 100
        fin_debt_ratio = fin_debt_ratio.fillna(0)
        fdr_p1 = np.percentile(fin_debt_ratio.dropna(), 1)
        fdr_p99 = np.percentile(fin_debt_ratio.dropna(), 99)
        df["fin_debt_ratio_w"] = fin_debt_ratio.clip(lower=fdr_p1, upper=fdr_p99)

        # Strategic additions (3 new)
        
        # FEATURE 13: TANGIBLE_ASSET_RATIO_W (Collateral value - critical for credit risk!)
        tangible_ratio = (df["asst_tang_fixed"].fillna(0) / df["asst_tot"].replace(0, np.nan)) * 100
        tangible_ratio = tangible_ratio.fillna(0)
        tang_p1 = np.percentile(tangible_ratio.dropna(), 1)
        tang_p99 = np.percentile(tangible_ratio.dropna(), 99)
        df["tangible_asset_ratio_w"] = tangible_ratio.clip(lower=tang_p1, upper=tang_p99)

        # FEATURE 14: DEBT_MATURITY_RATIO_W (Debt structure - refinancing risk!)
        total_debt = (
            df["debt_bank_st"].fillna(0) + df["debt_bank_lt"].fillna(0) +
            df["debt_fin_st"].fillna(0) + df["debt_fin_lt"].fillna(0)
        )
        long_term_debt = df["debt_bank_lt"].fillna(0) + df["debt_fin_lt"].fillna(0)
        debt_maturity = (long_term_debt / total_debt.replace(0, np.nan)) * 100
        debt_maturity = debt_maturity.fillna(0)  # No debt = 0
        dm_p1 = np.percentile(debt_maturity.dropna(), 1)
        dm_p99 = np.percentile(debt_maturity.dropna(), 99)
        df["debt_maturity_ratio_w"] = debt_maturity.clip(lower=dm_p1, upper=dm_p99)

        # FEATURE 15: LOG_ASSETS (Size control - small firms riskier!)
        log_assets = np.log1p(df["asst_tot"].fillna(0))
        la_p1 = np.percentile(log_assets.dropna(), 1)
        la_p99 = np.percentile(log_assets.dropna(), 99)
        df["log_assets"] = log_assets.clip(lower=la_p1, upper=la_p99)

        # Store preprocessing parameters
        details = {
            # Core 12 features
            "leverage_p1": float(leverage_p1), "leverage_p99": float(leverage_p99),
            "roe_p1": float(roe_p1), "roe_p99": float(roe_p99),
            "roa_p1": float(roa_p1), "roa_p99": float(roa_p99),
            "cta_p1": float(cta_p1), "cta_p99": float(cta_p99),
            "at_p1": float(at_p1), "at_p99": float(at_p99),
            "ic_p1": float(ic_p1), "ic_p99": float(ic_p99),
            "em_p1": float(em_p1), "em_p99": float(em_p99),
            "wcr_p1": float(wcr_p1), "wcr_p99": float(wcr_p99),
            "cfroa_p1": float(cfroa_p1), "cfroa_p99": float(cfroa_p99),
            "art_p1": float(art_p1), "art_p99": float(art_p99),
            "fdr_p1": float(fdr_p1), "fdr_p99": float(fdr_p99),
            # Strategic additions
            "tang_p1": float(tang_p1), "tang_p99": float(tang_p99),
            "dm_p1": float(dm_p1), "dm_p99": float(dm_p99),
            "la_p1": float(la_p1), "la_p99": float(la_p99),
        }

        print(f"OPTIMIZED preprocessing TRAIN complete: 15 features (12 proven + 3 strategic), NO imputation, {len(df):,} records")
        return df, details

    @staticmethod
    def transform_preprocessing(df, details):
        """
        TRANSFORM: Apply learned preprocessing parameters to new data
        """
        df = df.copy()

        # Core computations
        df["asst_tot_approx"] = (
            df["asst_intang_fixed"].fillna(0) +
            df["asst_tang_fixed"].fillna(0) +
            df["asst_fixed_fin"].fillna(0) +
            df["asst_current"].fillna(0)
        )
        df["asst_tot"] = df["asst_tot"].fillna(df["asst_tot_approx"])

        df["liab_tot_approx"] = (
            df["liab_lt"].fillna(0) +
            df["debt_bank_st"].fillna(0) +
            df["debt_bank_lt"].fillna(0) +
            df["debt_fin_st"].fillna(0) +
            df["debt_fin_lt"].fillna(0) +
            df["AP_st"].fillna(0) +
            df["AP_lt"].fillna(0) +
            df["liab_lt_emp"].fillna(0)
        )
        df["eqty_tot"] = df["eqty_tot"].fillna(df["asst_tot"] - df["liab_tot_approx"])

        # Apply learned transformations
        
        # FEATURE 1: LEVERAGE_W
        leverage = df["liab_tot_approx"] / df["asst_tot"] * 100
        leverage = leverage.mask(df["asst_tot"] == 0, pd.NA).fillna(0)
        df["leverage_w"] = leverage.clip(lower=details["leverage_p1"], upper=details["leverage_p99"])

        # FEATURE 2: ROE_W
        roe = df["prof_operations"] / df["eqty_tot"] * 100
        roe = roe.mask(df["eqty_tot"] == 0, pd.NA)
        roe = roe.mask((df["prof_operations"] == 0) & (df["eqty_tot"] != 0), 0.0)
        roe_calc = roe.round(2)
        missing_roe = df["roe"].isna().astype(int)
        roe_merged = df["roe"].fillna(roe_calc).fillna(0)
        df["roe_w"] = roe_merged.clip(lower=details["roe_p1"], upper=details["roe_p99"])

        # FEATURE 3: ROA_W
        roa = df["prof_operations"] / df["asst_tot"] * 100
        roa = roa.mask(df["asst_tot"] == 0, pd.NA)
        roa = roa.mask((df["prof_operations"] == 0) & (df["asst_tot"] != 0), 0.0)
        roa_calc = roa.round(2)
        roa_merged = df["roa"].fillna(roa_calc).fillna(0)
        df["roa_w"] = roa_merged.clip(lower=details["roa_p1"], upper=details["roa_p99"])

        # FEATURE 4: CASH_TO_ASSETS_W
        cash_to_assets = df["cash_and_equiv"].fillna(0) / df["asst_tot"] * 100
        cash_to_assets = cash_to_assets.mask(df["asst_tot"] == 0, pd.NA).fillna(0)
        df["cash_to_assets_w"] = cash_to_assets.clip(lower=details["cta_p1"], upper=details["cta_p99"])

        # FEATURE 5: ASSET_TURNOVER_W
        asset_turnover = df["rev_operating"].fillna(0) / df["asst_tot"]
        asset_turnover = asset_turnover.mask(df["asst_tot"] == 0, pd.NA).fillna(0)
        df["asset_turnover_w"] = asset_turnover.clip(lower=details["at_p1"], upper=details["at_p99"])

        # FEATURE 6: INTEREST_COVERAGE_W
        interest_exp = df["exp_financing"].fillna(0)
        ebitda = df["ebitda"].fillna(0)
        interest_coverage = ebitda / interest_exp.replace(0, np.nan)
        interest_coverage = interest_coverage.fillna(0)
        df["interest_coverage_w"] = interest_coverage.clip(lower=details["ic_p1"], upper=details["ic_p99"])

        # FEATURE 7: EBITDA_MARGIN_W
        revenue = df["rev_operating"].fillna(0)
        ebitda_margin = (df["ebitda"].fillna(0) / revenue.replace(0, np.nan)) * 100
        ebitda_margin = ebitda_margin.fillna(0)
        df["ebitda_margin_w"] = ebitda_margin.clip(lower=details["em_p1"], upper=details["em_p99"])

        # FEATURE 8: WORKING_CAPITAL_RATIO_W
        working_capital = df["wc_net"].fillna(0)
        wc_ratio = working_capital / df["asst_tot"] * 100
        wc_ratio = wc_ratio.mask(df["asst_tot"] == 0, pd.NA).fillna(0)
        df["working_capital_ratio_w"] = wc_ratio.clip(lower=details["wcr_p1"], upper=details["wcr_p99"])

        # FEATURE 9: MISSING_ROE_FLAG
        df["missing_roe_flag"] = missing_roe

        # FEATURE 10: CFROA_W
        cfroa = (df["cf_operations"].fillna(0) / df["asst_tot"].replace(0, np.nan)) * 100
        cfroa = cfroa.fillna(0)
        df["cfroa_w"] = cfroa.clip(lower=details["cfroa_p1"], upper=details["cfroa_p99"])

        # FEATURE 11: AR_TURNOVER_W
        ar_turnover = df["rev_operating"].fillna(0) / df["AR"].replace(0, np.nan)
        ar_turnover = ar_turnover.fillna(0)
        df["ar_turnover_w"] = ar_turnover.clip(lower=details["art_p1"], upper=details["art_p99"])

        # FEATURE 12: FIN_DEBT_RATIO_W
        financial_debt = (
            df["debt_bank_st"].fillna(0) + 
            df["debt_bank_lt"].fillna(0) + 
            df["debt_fin_st"].fillna(0) + 
            df["debt_fin_lt"].fillna(0)
        )
        fin_debt_ratio = (financial_debt / df["asst_tot"].replace(0, np.nan)) * 100
        fin_debt_ratio = fin_debt_ratio.fillna(0)
        df["fin_debt_ratio_w"] = fin_debt_ratio.clip(lower=details["fdr_p1"], upper=details["fdr_p99"])

        # FEATURE 13: TANGIBLE_ASSET_RATIO_W
        tangible_ratio = (df["asst_tang_fixed"].fillna(0) / df["asst_tot"].replace(0, np.nan)) * 100
        tangible_ratio = tangible_ratio.fillna(0)
        df["tangible_asset_ratio_w"] = tangible_ratio.clip(lower=details["tang_p1"], upper=details["tang_p99"])

        # FEATURE 14: DEBT_MATURITY_RATIO_W
        total_debt = (
            df["debt_bank_st"].fillna(0) + df["debt_bank_lt"].fillna(0) +
            df["debt_fin_st"].fillna(0) + df["debt_fin_lt"].fillna(0)
        )
        long_term_debt = df["debt_bank_lt"].fillna(0) + df["debt_fin_lt"].fillna(0)
        debt_maturity = (long_term_debt / total_debt.replace(0, np.nan)) * 100
        debt_maturity = debt_maturity.fillna(0)
        df["debt_maturity_ratio_w"] = debt_maturity.clip(lower=details["dm_p1"], upper=details["dm_p99"])

        # FEATURE 15: LOG_ASSETS
        log_assets = np.log1p(df["asst_tot"].fillna(0))
        df["log_assets"] = log_assets.clip(lower=details["la_p1"], upper=details["la_p99"])

        return df
