"""
Data models for fundamental financial data.
"""

import argparse
import os
import requests
import yaml
import pandas as pd
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict


# ============ 指標計算策略模式 ============

class FinancialRatioCalculator(ABC):
    """財務比率計算器基類"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """指標名稱"""
        pass
    
    @property
    @abstractmethod
    def required_fields(self) -> List[List[str]]:
        """
        必需的欄位關鍵字列表
        每個子列表代表一個必需欄位的可能名稱
        """
        pass
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        """計算指標"""
        pass
    
    def find_column(self, df: pd.DataFrame, keywords: List[str], 
                    exclude: List[str] = None) -> Optional[str]:
        """
        在 DataFrame 中尋找匹配的欄位
        
        Args:
            df: DataFrame
            keywords: 關鍵字列表（全部必須匹配）
            exclude: 排除的關鍵字
        
        Returns:
            匹配的欄位名稱，或 None
        """
        exclude = exclude or []
        
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            
            # 檢查是否包含所有關鍵字
            has_all_keywords = all(
                kw.lower().replace('_', '').replace(' ', '') in col_lower 
                for kw in keywords
            )
            
            # 檢查是否包含排除的關鍵字
            has_excluded = any(
                ex.lower().replace('_', '').replace(' ', '') in col_lower 
                for ex in exclude
            )
            
            if has_all_keywords and not has_excluded:
                return col
        
        return None
    
    def safe_divide(self, numerator: pd.Series, denominator: pd.Series, 
                    fill_value: float = np.nan) -> pd.Series:
        """安全除法，處理除以零的情況"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result = result.replace([np.inf, -np.inf], fill_value)
        return result


class CurrentRatioCalculator(FinancialRatioCalculator):
    """流動比率 = 總流動資產 / 總流動負債"""
    
    @property
    def name(self) -> str:
        return "Current Ratio"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [
            ['current', 'assets', 'total'],
            ['current', 'liabilities', 'total']
        ]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        assets_col = self.find_column(df, ['totalcurrentassets']) or \
                     self.find_column(df, ['current', 'assets'])
        liab_col = self.find_column(df, ['totalcurrentliabilities']) or \
                   self.find_column(df, ['current', 'liabilities'])
        
        if assets_col and liab_col:
            return self.safe_divide(
                pd.to_numeric(df[assets_col], errors='coerce'),
                pd.to_numeric(df[liab_col], errors='coerce')
            )
        return pd.Series([np.nan] * len(df), index=df.index)


class AcidTestRatioCalculator(FinancialRatioCalculator):
    """速動比率 = (現金 + 短期投資 + 應收帳款) / 總流動負債"""
    
    @property
    def name(self) -> str:
        return "Acid Test Ratio"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['cash'], ['current', 'liabilities']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        # 尋找各個欄位
        cash_col = self.find_column(df, ['cash', 'carrying']) or \
                   self.find_column(df, ['cash', 'equivalents'])
        short_inv_col = self.find_column(df, ['shortterm', 'investments']) or \
                        self.find_column(df, ['cashandshortterm'])
        receivables_col = self.find_column(df, ['receivables', 'net']) or \
                          self.find_column(df, ['currentnetreceivables'])
        liab_col = self.find_column(df, ['totalcurrentliabilities']) or \
                   self.find_column(df, ['current', 'liabilities'])
        
        if liab_col:
            cash = pd.to_numeric(df[cash_col], errors='coerce').fillna(0) if cash_col else 0
            short_inv = pd.to_numeric(df[short_inv_col], errors='coerce').fillna(0) if short_inv_col else 0
            receivables = pd.to_numeric(df[receivables_col], errors='coerce').fillna(0) if receivables_col else 0
            liabilities = pd.to_numeric(df[liab_col], errors='coerce')
            
            quick_assets = cash + short_inv + receivables
            return self.safe_divide(quick_assets, liabilities)
        
        return pd.Series([np.nan] * len(df), index=df.index)


class OperatingCashFlowRatioCalculator(FinancialRatioCalculator):
    """營業現金流量比率 = 營業現金流量 / 總流動負債"""
    
    @property
    def name(self) -> str:
        return "Operating Cash Flow Ratio"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['operating', 'cashflow'], ['current', 'liabilities']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        ocf_col = self.find_column(df, ['operatingcashflow']) or \
                  self.find_column(df, ['operating', 'cash'])
        liab_col = self.find_column(df, ['totalcurrentliabilities']) or \
                   self.find_column(df, ['current', 'liabilities'])
        
        if ocf_col and liab_col:
            return self.safe_divide(
                pd.to_numeric(df[ocf_col], errors='coerce'),
                pd.to_numeric(df[liab_col], errors='coerce')
            )
        return pd.Series([np.nan] * len(df), index=df.index)


class DebtRatioCalculator(FinancialRatioCalculator):
    """負債比率 = 總負債 / 總資產"""
    
    @property
    def name(self) -> str:
        return "Debt Ratio"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['total', 'liabilities'], ['total', 'assets']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        liab_col = self.find_column(df, ['totalliabilities'], exclude=['current'])
        assets_col = self.find_column(df, ['totalassets'], exclude=['current'])
        
        if liab_col and assets_col:
            return self.safe_divide(
                pd.to_numeric(df[liab_col], errors='coerce'),
                pd.to_numeric(df[assets_col], errors='coerce')
            )
        return pd.Series([np.nan] * len(df), index=df.index)


class DebtToEquityRatioCalculator(FinancialRatioCalculator):
    """負債權益比率 = 總負債 / 總股東權益"""
    
    @property
    def name(self) -> str:
        return "Debt to Equity Ratio"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['total', 'liabilities'], ['shareholder', 'equity']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        liab_col = self.find_column(df, ['totalliabilities'], exclude=['current'])
        equity_col = self.find_column(df, ['totalshareholderequity']) or \
                     self.find_column(df, ['shareholder', 'equity'])
        
        if liab_col and equity_col:
            return self.safe_divide(
                pd.to_numeric(df[liab_col], errors='coerce'),
                pd.to_numeric(df[equity_col], errors='coerce')
            )
        return pd.Series([np.nan] * len(df), index=df.index)


class InterestCoverageRatioCalculator(FinancialRatioCalculator):
    """利息保障倍數 = EBIT / 利息費用"""
    
    @property
    def name(self) -> str:
        return "Interest Coverage Ratio"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['ebit'], ['interest']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        # 尋找 EBIT 欄位
        ebit_col = self.find_column(df, ['income_ebit']) or \
                   self.find_column(df, ['ebit'], exclude=['ebitda'])
        
        # 尋找利息費用欄位 - 嘗試多種可能的欄位名稱
        interest_col = self.find_column(df, ['income_interestexpense']) or \
                       self.find_column(df, ['income_interestanddebtexpense']) or \
                       self.find_column(df, ['interestexpense']) or \
                       self.find_column(df, ['interestanddebtexpense'])
        
        # 檢查利息欄位是否有有效數據
        if interest_col and interest_col in df.columns:
            interest_data = pd.to_numeric(df[interest_col], errors='coerce')
            # 如果全為 NaN 或 0，則視為無效
            if interest_data.isna().all() or (interest_data == 0).all():
                interest_col = None
        
        if ebit_col and interest_col:
            ebit = pd.to_numeric(df[ebit_col], errors='coerce')
            interest = pd.to_numeric(df[interest_col], errors='coerce')
            interest = interest.replace(0, np.nan)
            return self.safe_divide(ebit, interest.abs())
        
        # ========== 備用方法：使用 EBIT - EBT 估算利息 ==========
        ebt_col = self.find_column(df, ['income_incomebeforetax']) or \
                  self.find_column(df, ['incomebeforetax'])
        
        if ebit_col and ebt_col:
            ebit = pd.to_numeric(df[ebit_col], errors='coerce')
            ebt = pd.to_numeric(df[ebt_col], errors='coerce')
            
            # 估算利息費用 = EBIT - EBT
            estimated_interest = (ebit - ebt).abs()
            
            # 只有當估算值大於 0 時才計算
            valid_interest = estimated_interest.replace(0, np.nan)
            
            if not valid_interest.isna().all():
                print(f"  ℹ Interest Coverage: 使用 EBIT-EBT 估算利息費用")
                return self.safe_divide(ebit, valid_interest)
        # ========================================================
        
        # 無法計算，返回 NaN
        return pd.Series([np.nan] * len(df), index=df.index)


class AssetTurnoverRatioCalculator(FinancialRatioCalculator):
    """資產周轉率 = 總營收 / 總資產"""
    
    @property
    def name(self) -> str:
        return "Asset Turnover Ratio"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['total', 'revenue'], ['total', 'assets']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        revenue_col = self.find_column(df, ['totalrevenue']) or \
                      self.find_column(df, ['revenue'])
        assets_col = self.find_column(df, ['totalassets'], exclude=['current'])
        
        if revenue_col and assets_col:
            return self.safe_divide(
                pd.to_numeric(df[revenue_col], errors='coerce'),
                pd.to_numeric(df[assets_col], errors='coerce')
            )
        return pd.Series([np.nan] * len(df), index=df.index)


class InventoryTurnoverRatioCalculator(FinancialRatioCalculator):
    """存貨周轉率 = 銷貨成本 / 存貨"""
    
    @property
    def name(self) -> str:
        return "Inventory Turnover Ratio"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['cost', 'revenue'], ['inventory']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        cogs_col = self.find_column(df, ['costofrevenue']) or \
                   self.find_column(df, ['costofgoodssold']) or \
                   self.find_column(df, ['cost', 'goods'])
        inventory_col = self.find_column(df, ['inventory'], exclude=['days'])
        
        if cogs_col and inventory_col:
            cogs = pd.to_numeric(df[cogs_col], errors='coerce')
            inventory = pd.to_numeric(df[inventory_col], errors='coerce').replace(0, np.nan)
            return self.safe_divide(cogs, inventory)
        return pd.Series([np.nan] * len(df), index=df.index)


class DaySalesInInventoryCalculator(FinancialRatioCalculator):
    """存貨銷售天數 = 365 / 存貨周轉率"""
    
    @property
    def name(self) -> str:
        return "Day Sales in Inventory Ratio"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['inventory', 'turnover']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        if 'Inventory Turnover Ratio' in df.columns:
            inv_turnover = pd.to_numeric(df['Inventory Turnover Ratio'], errors='coerce')
            return self.safe_divide(pd.Series([365] * len(df), index=df.index), inv_turnover)
        return pd.Series([np.nan] * len(df), index=df.index)


class ReturnOnAssetsCalculator(FinancialRatioCalculator):
    """資產報酬率 = 淨利 / 總資產"""
    
    @property
    def name(self) -> str:
        return "Return on Ratio"  # 保持與原始名稱一致
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['net', 'income'], ['total', 'assets']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        ni_col = self.find_column(df, ['netincome'], exclude=['comprehensive'])
        assets_col = self.find_column(df, ['totalassets'], exclude=['current'])
        
        if ni_col and assets_col:
            return self.safe_divide(
                pd.to_numeric(df[ni_col], errors='coerce'),
                pd.to_numeric(df[assets_col], errors='coerce')
            )
        return pd.Series([np.nan] * len(df), index=df.index)


class ReturnOnEquityCalculator(FinancialRatioCalculator):
    """股東權益報酬率 = 淨利 / 總股東權益"""
    
    @property
    def name(self) -> str:
        return "Return on Equity Ratio"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['net', 'income'], ['shareholder', 'equity']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        ni_col = self.find_column(df, ['netincome'], exclude=['comprehensive'])
        equity_col = self.find_column(df, ['totalshareholderequity']) or \
                     self.find_column(df, ['shareholder', 'equity'])
        
        if ni_col and equity_col:
            return self.safe_divide(
                pd.to_numeric(df[ni_col], errors='coerce'),
                pd.to_numeric(df[equity_col], errors='coerce')
            )
        return pd.Series([np.nan] * len(df), index=df.index)


class GrossProfitMarginCalculator(FinancialRatioCalculator):
    """毛利率 = 毛利 / 總營收 * 100"""
    
    @property
    def name(self) -> str:
        return "Gross Profit Margin"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['gross', 'profit'], ['total', 'revenue']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        gp_col = self.find_column(df, ['grossprofit'])
        revenue_col = self.find_column(df, ['totalrevenue']) or \
                      self.find_column(df, ['revenue'])
        
        if gp_col and revenue_col:
            return self.safe_divide(
                pd.to_numeric(df[gp_col], errors='coerce'),
                pd.to_numeric(df[revenue_col], errors='coerce')
            ) * 100
        return pd.Series([np.nan] * len(df), index=df.index)


class NetProfitMarginCalculator(FinancialRatioCalculator):
    """淨利率 = 淨利 / 總營收 * 100"""
    
    @property
    def name(self) -> str:
        return "Net Profit Margin"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['net', 'income'], ['total', 'revenue']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        ni_col = self.find_column(df, ['netincome'], exclude=['comprehensive'])
        revenue_col = self.find_column(df, ['totalrevenue']) or \
                      self.find_column(df, ['revenue'])
        
        if ni_col and revenue_col:
            return self.safe_divide(
                pd.to_numeric(df[ni_col], errors='coerce'),
                pd.to_numeric(df[revenue_col], errors='coerce')
            ) * 100
        return pd.Series([np.nan] * len(df), index=df.index)


class OperatingMarginCalculator(FinancialRatioCalculator):
    """營業利潤率 = 營業收入 / 總營收 * 100"""
    
    @property
    def name(self) -> str:
        return "Operating Margin"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['operating', 'income'], ['total', 'revenue']]
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        oi_col = self.find_column(df, ['operatingincome']) or \
                 self.find_column(df, ['operating', 'income'])
        revenue_col = self.find_column(df, ['totalrevenue']) or \
                      self.find_column(df, ['revenue'])
        
        if oi_col and revenue_col:
            return self.safe_divide(
                pd.to_numeric(df[oi_col], errors='coerce'),
                pd.to_numeric(df[revenue_col], errors='coerce')
            ) * 100
        return pd.Series([np.nan] * len(df), index=df.index)


class QualityScoreCalculator(FinancialRatioCalculator):
    """
    綜合品質分數 (0-1)
    
    用於 FundamentalScoreAgentReward 的獎勵計算
    綜合考慮：獲利能力、償債能力、營運效率、成長性
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights: 各指標權重，預設為均衡權重
        """
        self.weights = weights or {
            'profitability': 0.30,   # 獲利能力
            'solvency': 0.25,        # 償債能力
            'liquidity': 0.20,       # 流動性
            'efficiency': 0.15,      # 營運效率
            'cash_flow': 0.10,       # 現金流
        }
    
    @property
    def name(self) -> str:
        return "Quality Score"
    
    @property
    def required_fields(self) -> List[List[str]]:
        return [['debt', 'ratio']]  # 至少需要負債比率
    
    def _normalize_score(self, series: pd.Series, 
                         ideal_min: float, ideal_max: float,
                         higher_is_better: bool = True) -> pd.Series:
        """
        將指標標準化到 0-1 範圍
        """
        # 轉換為數值並處理缺失值
        series = pd.to_numeric(series, errors='coerce').fillna(0)
        
        # 限制極端值
        lower_bound = ideal_min - abs(ideal_max - ideal_min)
        upper_bound = ideal_max + abs(ideal_max - ideal_min)
        series = series.clip(lower_bound, upper_bound)
        
        if higher_is_better:
            score = (series - ideal_min) / (ideal_max - ideal_min + 1e-8)
        else:
            score = (ideal_max - series) / (ideal_max - ideal_min + 1e-8)
        
        return score.clip(0, 1)
    
    def _find_ratio_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """尋找已計算的比率欄位"""
        for name in possible_names:
            if name in df.columns:
                return name
            # 也嘗試小寫匹配
            for col in df.columns:
                if name.lower().replace(' ', '') == col.lower().replace(' ', ''):
                    return col
        return None
    
    def calculate(self, df: pd.DataFrame, matched_columns: Dict[str, str]) -> pd.Series:
        """計算綜合品質分數"""
        
        scores = pd.DataFrame(index=df.index)
        components_used = []
        
        print(f"    可用的計算欄位: {[c for c in df.columns if 'Ratio' in c or 'Margin' in c or 'Return' in c]}")
        
        # ========== 1. 獲利能力分數 (Profitability) ==========
        profitability_scores = []
        
        # ROA
        roa_col = self._find_ratio_column(df, ['Return on Ratio', 'Return on Assets', 'ROA'])
        if roa_col:
            roa = pd.to_numeric(df[roa_col], errors='coerce')
            roa_score = self._normalize_score(roa, 0.02, 0.15, higher_is_better=True)
            profitability_scores.append(roa_score)
            print(f"    - 使用 {roa_col} 計算 ROA 分數")
        
        # ROE
        roe_col = self._find_ratio_column(df, ['Return on Equity Ratio', 'ROE'])
        if roe_col:
            roe = pd.to_numeric(df[roe_col], errors='coerce')
            roe_score = self._normalize_score(roe, 0.05, 0.25, higher_is_better=True)
            profitability_scores.append(roe_score)
            print(f"    - 使用 {roe_col} 計算 ROE 分數")
        
        # 淨利率
        npm_col = self._find_ratio_column(df, ['Net Profit Margin', 'NPM'])
        if npm_col:
            npm = pd.to_numeric(df[npm_col], errors='coerce') / 100  # 百分比轉小數
            npm_score = self._normalize_score(npm, 0.03, 0.20, higher_is_better=True)
            profitability_scores.append(npm_score)
            print(f"    - 使用 {npm_col} 計算淨利率分數")
        
        if profitability_scores:
            scores['profitability'] = pd.concat(profitability_scores, axis=1).mean(axis=1)
            components_used.append('profitability')
        else:
            scores['profitability'] = 0.5
        
        # ========== 2. 償債能力分數 (Solvency) ==========
        solvency_scores = []
        
        # 負債比率 (越低越好)
        debt_col = self._find_ratio_column(df, ['Debt Ratio'])
        if debt_col:
            debt_ratio = pd.to_numeric(df[debt_col], errors='coerce')
            debt_score = self._normalize_score(debt_ratio, 0.2, 0.7, higher_is_better=False)
            solvency_scores.append(debt_score)
            print(f"    - 使用 {debt_col} 計算負債分數")
        
        # 利息保障倍數
        icr_col = self._find_ratio_column(df, ['Interest Coverage Ratio', 'ICR'])
        if icr_col:
            icr = pd.to_numeric(df[icr_col], errors='coerce')
            # 過濾極端值
            icr = icr.clip(-50, 50)
            icr_score = self._normalize_score(icr, 2, 15, higher_is_better=True)
            solvency_scores.append(icr_score)
            print(f"    - 使用 {icr_col} 計算利息保障分數")
        
        if solvency_scores:
            scores['solvency'] = pd.concat(solvency_scores, axis=1).mean(axis=1)
            components_used.append('solvency')
        else:
            scores['solvency'] = 0.5
        
        # ========== 3. 流動性分數 (Liquidity) ==========
        liquidity_scores = []
        
        # 流動比率
        cr_col = self._find_ratio_column(df, ['Current Ratio'])
        if cr_col:
            cr = pd.to_numeric(df[cr_col], errors='coerce')
            cr_score = self._normalize_score(cr, 1.0, 2.5, higher_is_better=True)
            liquidity_scores.append(cr_score)
            print(f"    - 使用 {cr_col} 計算流動比率分數")
        
        # 速動比率
        atr_col = self._find_ratio_column(df, ['Acid Test Ratio', 'Quick Ratio'])
        if atr_col:
            atr = pd.to_numeric(df[atr_col], errors='coerce')
            atr_score = self._normalize_score(atr, 0.5, 1.5, higher_is_better=True)
            liquidity_scores.append(atr_score)
            print(f"    - 使用 {atr_col} 計算速動比率分數")
        
        if liquidity_scores:
            scores['liquidity'] = pd.concat(liquidity_scores, axis=1).mean(axis=1)
            components_used.append('liquidity')
        else:
            scores['liquidity'] = 0.5
        
        # ========== 4. 營運效率分數 (Efficiency) ==========
        efficiency_scores = []
        
        # 資產周轉率
        at_col = self._find_ratio_column(df, ['Asset Turnover Ratio'])
        if at_col:
            at = pd.to_numeric(df[at_col], errors='coerce')
            at_score = self._normalize_score(at, 0.3, 1.5, higher_is_better=True)
            efficiency_scores.append(at_score)
            print(f"    - 使用 {at_col} 計算資產周轉率分數")
        
        # 存貨周轉率
        it_col = self._find_ratio_column(df, ['Inventory Turnover Ratio'])
        if it_col:
            it = pd.to_numeric(df[it_col], errors='coerce')
            it_score = self._normalize_score(it, 3, 12, higher_is_better=True)
            efficiency_scores.append(it_score)
            print(f"    - 使用 {it_col} 計算存貨周轉率分數")
        
        if efficiency_scores:
            scores['efficiency'] = pd.concat(efficiency_scores, axis=1).mean(axis=1)
            components_used.append('efficiency')
        else:
            scores['efficiency'] = 0.5
        
        # ========== 5. 現金流分數 (Cash Flow) ==========
        cash_flow_scores = []
        
        # 營業現金流量比率
        ocf_col = self._find_ratio_column(df, ['Operating Cash Flow Ratio'])
        if ocf_col:
            ocf = pd.to_numeric(df[ocf_col], errors='coerce')
            ocf_score = self._normalize_score(ocf, 0.1, 0.6, higher_is_better=True)
            cash_flow_scores.append(ocf_score)
            print(f"    - 使用 {ocf_col} 計算現金流分數")
        
        if cash_flow_scores:
            scores['cash_flow'] = pd.concat(cash_flow_scores, axis=1).mean(axis=1)
            components_used.append('cash_flow')
        else:
            scores['cash_flow'] = 0.5
        
        # ========== 計算加權總分 ==========
        quality_score = pd.Series(0.0, index=df.index)
        
        for component, weight in self.weights.items():
            if component in scores.columns:
                quality_score += scores[component] * weight
        
        # 確保分數在 0-1 範圍內
        quality_score = quality_score.clip(0, 1)
        
        print(f"    使用的組成部分: {components_used}")
        print(f"    各維度平均分數:")
        for comp in components_used:
            print(f"      - {comp}: {scores[comp].mean():.4f}")
        
        return quality_score


# ============ 指標計算器註冊表 ============

class FinancialRatioRegistry:
    """財務比率計算器註冊表"""
    
    _calculators: Dict[str, FinancialRatioCalculator] = {}
    
    @classmethod
    def register(cls, calculator: FinancialRatioCalculator):
        """註冊計算器"""
        cls._calculators[calculator.name] = calculator
    
    @classmethod
    def get(cls, name: str) -> Optional[FinancialRatioCalculator]:
        """獲取計算器"""
        return cls._calculators.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, FinancialRatioCalculator]:
        """獲取所有計算器"""
        return cls._calculators.copy()
    
    @classmethod
    def list_names(cls) -> List[str]:
        """列出所有指標名稱"""
        return list(cls._calculators.keys())


# 註冊所有計算器
def _register_default_calculators():
    calculators = [
        CurrentRatioCalculator(),
        AcidTestRatioCalculator(),
        OperatingCashFlowRatioCalculator(),
        DebtRatioCalculator(),
        DebtToEquityRatioCalculator(),
        InterestCoverageRatioCalculator(),
        AssetTurnoverRatioCalculator(),
        InventoryTurnoverRatioCalculator(),
        DaySalesInInventoryCalculator(),
        ReturnOnAssetsCalculator(),
        ReturnOnEquityCalculator(),
        GrossProfitMarginCalculator(),
        NetProfitMarginCalculator(),
        OperatingMarginCalculator(),
        QualityScoreCalculator(),  # 新增：品質分數計算器
    ]
    for calc in calculators:
        FinancialRatioRegistry.register(calc)
        print(f"  已註冊: {calc.name}")  # 除錯用

_register_default_calculators()


# ============ FundamentalData 類別 ============

class FundamentalData:

    def __init__(self, ticker, start_date, end_date, data_dir='data/fundamentals'):
        """
        Initializes the FundamentalData class.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            data_dir (str): Directory to save/load the fundamental data.
        """
        self.ticker = ticker
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.file_path = os.path.join(self.data_dir, f"{self.ticker}_fundamentals.csv")
        
        # 初始化設定變數
        self.config = None
        self.api_key = None
        self.indicator_list = []
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # 計算器註冊表
        self.ratio_registry = FinancialRatioRegistry

    def load_api_key(self, config_path='configs/defaults.yaml'):
        """
        Loads the API key from a YAML configuration file.
        """
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                self.api_key = self.config.get('data', {}).get('fundamental', {}).get('api_key', None)
                self.indicator_list = self.config.get('data', {}).get('fundamental', {}).get('indicator_list', [])
        except Exception as e:
            print(f"Error loading API key: {e}")
        return self.api_key
    
    def download_all_fundamental_data(self, force: bool = False):
        """
        Downloads fundamental data for all tickers in the config file.
        
        Args:
            force (bool): 如果為 True，即使檔案已存在也強制重新下載
        """
        ticker_list = self.config.get('data', {}).get('ticker_list', [])
        
        if not ticker_list:
            print("No ticker list found in config file.")
            return
        
        original_ticker = self.ticker
        print(f"Found {len(ticker_list)} tickers to download: {ticker_list}")
        
        # ========== 新增：統計資訊 ==========
        skipped_count = 0
        downloaded_count = 0
        failed_count = 0
        # ====================================
        
        for ticker in ticker_list:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}...")
            print(f"{'='*50}")
            
            ticker_data = FundamentalData(ticker, self.start_date.strftime('%Y-%m-%d'), 
                                        self.end_date.strftime('%Y-%m-%d'), self.data_dir)
            ticker_data.config = self.config
            ticker_data.api_key = self.api_key
            ticker_data.indicator_list = self.indicator_list
            
            json_file_path = os.path.join(ticker_data.data_dir, f"{ticker}_fundamentals.json")
            
            # ========== 改進的檢查邏輯 ==========
            if os.path.exists(json_file_path) and not force:
                # 額外檢查檔案是否有效（不是空的或損壞的）
                try:
                    import json
                    with open(json_file_path, 'r') as f:
                        existing_data = json.load(f)
                    
                    # 檢查是否有實際資料
                    has_data = any(
                        report_type in existing_data and 
                        (existing_data[report_type].get('quarterlyReports') or 
                         existing_data[report_type].get('annualReports'))
                        for report_type in ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']
                    )
                    
                    if has_data:
                        print(f"✓ Valid data for {ticker} already exists. Skipping download.")
                        skipped_count += 1
                        continue
                    else:
                        print(f"⚠ Existing file for {ticker} is empty or invalid. Re-downloading...")
                        
                except (json.JSONDecodeError, Exception) as e:
                    print(f"⚠ Existing file for {ticker} is corrupted: {e}. Re-downloading...")
            # ====================================
            
            try:
                ticker_data.download_fundamental_data(force=force)
                downloaded_count += 1
                time.sleep(30)  # Alpha Vantage 免費版每分鐘最多 5 次請求
            except Exception as e:
                print(f"✗ Failed to download {ticker}: {e}")
                failed_count += 1
        
        self.ticker = original_ticker
        
        # ========== 新增：下載摘要 ==========
        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"Total tickers: {len(ticker_list)}")
        print(f"  ✓ Skipped (already exists): {skipped_count}")
        print(f"  ↓ Downloaded: {downloaded_count}")
        print(f"  ✗ Failed: {failed_count}")
        print(f"{'='*60}")
        # ====================================
    
    def download_fundamental_data(self, force: bool = False):
        """
        Downloads fundamental data for the specified ticker using the Alpha Vantage API.
        
        Args:
            force (bool): 如果為 True，即使檔案已存在也強制重新下載
        """
        if not self.api_key:
            print("API key not loaded. Please call load_api_key() first.")
            return
    
        if not self.indicator_list:
            print("No indicators specified in config file.")
            return
        
        # ========== 新增：檢查是否已存在 ==========
        json_file_path = os.path.join(self.data_dir, f"{self.ticker}_fundamentals.json")
        
        if os.path.exists(json_file_path) and not force:
            print(f"✓ Data for {self.ticker} already exists at {json_file_path}")
            print(f"  Use --force or force=True to re-download.")
            return
        
        if force and os.path.exists(json_file_path):
            print(f"⚠ Force mode: Re-downloading data for {self.ticker}...")
        # =============================================
        
        all_data = {}
        
        for list_item in self.indicator_list:
            print(f"Downloading {list_item} data for {self.ticker}...")
            url = f"https://www.alphavantage.co/query?function={list_item.upper()}&symbol={self.ticker}&apikey={self.api_key}"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    
                    if "Error Message" in data:
                        print(f"API Error for {list_item}: {data['Error Message']}")
                        continue
                    elif "Note" in data:
                        print(f"API Rate limit for {list_item}: {data['Note']}")
                        continue
                    
                    all_data[list_item] = data
                else:
                    print(f"Error fetching {list_item} data from Alpha Vantage: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {list_item}: {e}")
        
        if all_data:
            import json
            with open(json_file_path, 'w') as f:
                json.dump(all_data, f, indent=2)
            print(f"Raw fundamental data saved to {json_file_path}")
        else:
            print("No data downloaded.")

    def load_fundamental_data(self):
        """
        Loads the fundamental data from a JSON file.
        """
        json_file_path = os.path.join(self.data_dir, f"{self.ticker}_fundamentals.json")
    
        if os.path.exists(json_file_path):
            try:
                import json
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                print(f"Loaded fundamental data for {self.ticker}")
                return data
            except Exception as e:
                print(f"Error loading fundamental data: {e}")
                return None
        else:
            print(f"No fundamental data found for {self.ticker}. Please download it first.")
            return None

    def preprocess_fundamental_data(self):
        """
        Preprocesses the fundamental data for analysis.
        """
        data = self.load_fundamental_data()
        if data is not None:
            income_quarterly = pd.DataFrame(data.get('INCOME_STATEMENT', {}).get('quarterlyReports', []))
            balance_quarterly = pd.DataFrame(data.get('BALANCE_SHEET', {}).get('quarterlyReports', []))
            cashflow_quarterly = pd.DataFrame(data.get('CASH_FLOW', {}).get('quarterlyReports', []))
            
            # 選擇需要的欄位
            income_cols = ['fiscalDateEnding', 'netIncome', 'ebit']
            balance_cols = ['fiscalDateEnding', 'totalCurrentAssets', 'totalCurrentLiabilities', 
                          'totalLiabilities', 'totalAssets', 'commonStock', 'retainedEarnings', 
                          'cashAndCashEquivalentsAtCarryingValue', 'cashAndShortTermInvestments', 
                          'currentNetReceivables', 'inventory']
            cashflow_cols = ['fiscalDateEnding', 'operatingCashflow']
            
            # 安全選擇欄位
            income_df = income_quarterly[[c for c in income_cols if c in income_quarterly.columns]].copy()
            balance_df = balance_quarterly[[c for c in balance_cols if c in balance_quarterly.columns]].copy()
            cashflow_df = cashflow_quarterly[[c for c in cashflow_cols if c in cashflow_quarterly.columns]].copy()

            # 轉換日期並過濾
            for df in [income_df, balance_df, cashflow_df]:
                if 'fiscalDateEnding' in df.columns:
                    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            
            mask_income = (income_df['fiscalDateEnding'] >= self.start_date) & (income_df['fiscalDateEnding'] <= self.end_date)
            mask_balance = (balance_df['fiscalDateEnding'] >= self.start_date) & (balance_df['fiscalDateEnding'] <= self.end_date)
            mask_cashflow = (cashflow_df['fiscalDateEnding'] >= self.start_date) & (cashflow_df['fiscalDateEnding'] <= self.end_date)
            
            income_df = income_df.loc[mask_income].copy()
            balance_df = balance_df.loc[mask_balance].copy()
            cashflow_df = cashflow_df.loc[mask_cashflow].copy()
            
            merged_df = income_df.merge(balance_df, on='fiscalDateEnding').merge(cashflow_df, on='fiscalDateEnding')
            merged_df.set_index('fiscalDateEnding', inplace=True)
            
            return merged_df
        else:
            print("No data to preprocess.")
            return None
        
    def calculate_financial_ratios_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        從 DataFrame 計算財務比率（使用策略模式）
        
        Args:
            df: 包含財務數據的 DataFrame
        
        Returns:
            包含計算出的財務比率的 DataFrame
        """
        df_ratios = df.copy()
        
        print("\n" + "="*60)
        print("計算財務比率")
        print("="*60)
        print(f"可用欄位數量: {len(df.columns)}")
        print(f"可用欄位: {[col for col in df.columns if col not in ['fiscalDateEnding', 'ticker']][:10]}...")
        
        calculated_count = 0
        failed_count = 0
        
        # 取得所有計算器
        all_calculators = self.ratio_registry.get_all()
        
        # ========== 分離 QualityScoreCalculator，確保最後執行 ==========
        quality_calculator = all_calculators.pop('Quality Score', None)
        # ===============================================================
        
        # 先計算其他比率
        for name, calculator in all_calculators.items():
            try:
                # 特殊處理：Day Sales in Inventory 需要先計算 Inventory Turnover
                if name == "Day Sales in Inventory Ratio" and "Inventory Turnover Ratio" not in df_ratios.columns:
                    continue
                
                result = calculator.calculate(df_ratios, {})
                
                # 檢查結果是否有效
                valid_count = result.notna().sum()
                total_count = len(result)
                
                if valid_count > 0:
                    df_ratios[name] = result
                    print(f"  ✓ {name}: {valid_count}/{total_count} 有效值")
                    calculated_count += 1
                else:
                    print(f"  ✗ {name}: 無法計算（缺少必要欄位）")
                    failed_count += 1
                    
            except Exception as e:
                print(f"  ✗ {name}: 計算錯誤 - {str(e)}")
                failed_count += 1
        
        # ========== 最後計算 Quality Score ==========
        if quality_calculator:
            try:
                print("\n  --- 計算綜合品質分數 ---")
                result = quality_calculator.calculate(df_ratios, {})
                valid_count = result.notna().sum()
                total_count = len(result)
                
                if valid_count > 0:
                    df_ratios['Quality Score'] = result
                    print(f"  ✓ Quality Score: {valid_count}/{total_count} 有效值")
                    print(f"    範圍: {result.min():.4f} - {result.max():.4f}")
                    print(f"    平均: {result.mean():.4f}")
                    calculated_count += 1
                else:
                    print(f"  ✗ Quality Score: 無法計算")
                    failed_count += 1
            except Exception as e:
                print(f"  ✗ Quality Score: 計算錯誤 - {str(e)}")
                import traceback
                traceback.print_exc()
                failed_count += 1
        # =============================================
        
        print("="*60)
        print(f"計算完成: {calculated_count} 成功, {failed_count} 失敗")
        print("="*60 + "\n")
        
        return df_ratios

    def save_simple_fundamental_data(self, df):
        """
        簡化版的保存方法
        """
        if df is None or df.empty:
            print("No data to save.")
            return
        
        try:
            print(f"Saving data for {self.ticker}...")
            print(f"Data shape: {df.shape}")
            
            df_to_save = df.copy()
            df_to_save['ticker'] = self.ticker
            
            if df_to_save.index.name == 'fiscalDateEnding':
                df_to_save.reset_index(inplace=True)
            
            df_to_save.to_csv(self.file_path, index=False, encoding='utf-8-sig')
            print(f"Saved fundamental data for {self.ticker} to {self.file_path}")
            print(f"Saved {len(df_to_save)} records")
            
        except Exception as e:
            print(f"Error saving data for {self.ticker}: {e}")

    def process_all_fundamental_data(self):
        """
        處理所有股票的基本面資料
        """
        ticker_list = self.config.get('data', {}).get('ticker_list', [])
        
        if not ticker_list:
            print("No ticker list found in config file.")
            return
        
        print(f"Processing fundamental data for {len(ticker_list)} tickers...")
        
        for ticker in ticker_list:
            print(f"\n{'='*40}")
            print(f"Processing {ticker}...")
            print(f"{'='*40}")
            
            try:
                ticker_data = FundamentalData(ticker, self.start_date.strftime('%Y-%m-%d'), 
                                            self.end_date.strftime('%Y-%m-%d'), self.data_dir)
                ticker_data.config = self.config
                ticker_data.api_key = self.api_key
                ticker_data.indicator_list = self.indicator_list
                
                raw_data = ticker_data.load_fundamental_data()
                
                if raw_data is not None:
                    has_real_data = False
                    for report_type in ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']:
                        if report_type in raw_data:
                            if ('annualReports' in raw_data[report_type] and 
                                len(raw_data[report_type]['annualReports']) > 0):
                                has_real_data = True
                                break
                            if ('quarterlyReports' in raw_data[report_type] and 
                                len(raw_data[report_type]['quarterlyReports']) > 0):
                                has_real_data = True
                                break
                    
                    if has_real_data:
                        processed_data = ticker_data.preprocess_fundamental_data()
                        
                        if processed_data is not None and not processed_data.empty:
                            ticker_data.save_simple_fundamental_data(processed_data)
                        else:
                            print(f"No valid processed data for {ticker}")
                    else:
                        print(f"No real financial data available for {ticker}")
                else:
                    print(f"No raw data file found for {ticker}")
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        print(f"\nCompleted processing all fundamental data.")
    
    def json_to_csv(self):
        """
        Converts the downloaded JSON fundamental data to CSV format based on config features.
        """
        data = self.load_fundamental_data()
        if data is None:
            print("No data to convert from JSON to CSV.")
            return None
            
        ticker = data.get('INCOME_STATEMENT', {}).get('symbol', self.ticker)
        print(f"Converting data for {ticker}...")
        
        # 獲取配置中的特徵列表
        target_features = self.config.get('data', {}).get('fundamental', {}).get('features', [])
        if not target_features:
            print("No features specified in config file.")
            return None
        
        # ========== 確保 Quality Score 在特徵列表中 ==========
        if 'Quality Score' not in target_features:
            target_features = target_features + ['Quality Score']
            print(f"自動加入 Quality Score 到特徵列表")
        # =====================================================
        
        print(f"Target features: {target_features}")
        
        # 合併季度報告數據
        income_quarterly = data.get('INCOME_STATEMENT', {}).get('quarterlyReports', [])
        balance_quarterly = data.get('BALANCE_SHEET', {}).get('quarterlyReports', [])
        cashflow_quarterly = data.get('CASH_FLOW', {}).get('quarterlyReports', [])
        
        print(f"Found {len(income_quarterly)} income, {len(balance_quarterly)} balance, {len(cashflow_quarterly)} cashflow quarterly reports")
        
        # 按日期合併
        quarterly_data_by_date = {}
        
        for report in income_quarterly:
            date = report.get('fiscalDateEnding')
            if date:
                if date not in quarterly_data_by_date:
                    quarterly_data_by_date[date] = {'fiscalDateEnding': date, 'ticker': ticker}
                for key, value in report.items():
                    if key != 'fiscalDateEnding':
                        quarterly_data_by_date[date][f"income_{key}"] = value
        
        for report in balance_quarterly:
            date = report.get('fiscalDateEnding')
            if date and date in quarterly_data_by_date:
                for key, value in report.items():
                    if key != 'fiscalDateEnding':
                        quarterly_data_by_date[date][f"balance_{key}"] = value
        
        for report in cashflow_quarterly:
            date = report.get('fiscalDateEnding')
            if date and date in quarterly_data_by_date:
                for key, value in report.items():
                    if key != 'fiscalDateEnding':
                        quarterly_data_by_date[date][f"cashflow_{key}"] = value
        
        if not quarterly_data_by_date:
            print("No quarterly data available.")
            return None
        
        df = pd.DataFrame(list(quarterly_data_by_date.values()))
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df = df.sort_values('fiscalDateEnding', ascending=False)
        
        # 數據清理 - 將 "None" 字串轉為 NaN
        for col in df.columns:
            if col not in ['fiscalDateEnding', 'ticker']:
                df[col] = df[col].replace('None', pd.NA)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        # 計算財務比率
        df_with_ratios = self.calculate_financial_ratios_from_df(df)
        
        # 移除全為空值的原始欄位（但保留計算出的比率）
        empty_columns = []
        ratio_columns = ['Current Ratio', 'Acid Test Ratio', 'Operating Cash Flow Ratio', 
                        'Debt Ratio', 'Debt to Equity Ratio', 'Interest Coverage Ratio',
                        'Asset Turnover Ratio', 'Inventory Turnover Ratio', 
                        'Day Sales in Inventory Ratio', 'Return on Ratio', 
                        'Return on Equity Ratio', 'Gross Profit Margin', 
                        'Net Profit Margin', 'Operating Margin', 'Quality Score']
        
        for col in df_with_ratios.columns:
            if col in ['fiscalDateEnding', 'ticker'] or col in ratio_columns:
                continue
            if df_with_ratios[col].isna().all() or (df_with_ratios[col] == 0).all():
                empty_columns.append(col)
        
        if empty_columns:
            print(f"\n⚠ 移除空白欄位: {empty_columns}")
            df_with_ratios = df_with_ratios.drop(columns=empty_columns)
        
        # 選擇配置中指定的特徵
        feature_columns = ['fiscalDateEnding', 'ticker']
        available_features = []
        
        for feature in target_features:
            # 直接匹配
            if feature in df_with_ratios.columns:
                if feature not in feature_columns:
                    feature_columns.append(feature)
                    available_features.append(feature)
                continue
            
            # 模糊匹配
            feature_normalized = feature.lower().replace(' ', '').replace('ratio', '')
            for col in df_with_ratios.columns:
                col_normalized = col.lower().replace(' ', '').replace('ratio', '')
                if feature_normalized in col_normalized or col_normalized in feature_normalized:
                    if col not in feature_columns:
                        feature_columns.append(col)
                        available_features.append(feature)
                    break
        
        # ========== 確保 Quality Score 一定被加入 ==========
        if 'Quality Score' in df_with_ratios.columns and 'Quality Score' not in feature_columns:
            feature_columns.append('Quality Score')
            available_features.append('Quality Score')
            print("✓ 已加入 Quality Score 欄位")
        # ===================================================
        
        existing_columns = [col for col in feature_columns if col in df_with_ratios.columns]
        
        print(f"\nSelected features: {available_features}")
        print(f"Final columns: {existing_columns}")
        
        if len(existing_columns) <= 2:
            print("Warning: No feature columns found, using all calculated ratios")
            result_df = df_with_ratios
        else:
            result_df = df_with_ratios[existing_columns].copy()
        
        # 移除結果中全為空值的欄位（但不移除 Quality Score）
        cols_to_drop = []
        for col in result_df.columns:
            if col in ['fiscalDateEnding', 'ticker', 'Quality Score']:
                continue
            if result_df[col].isna().all():
                cols_to_drop.append(col)
        
        if cols_to_drop:
            print(f"⚠ 最終移除空白欄位: {cols_to_drop}")
            result_df = result_df.drop(columns=cols_to_drop)
        
        # 保存
        result_df.to_csv(self.file_path, index=False, encoding='utf-8-sig')
        print(f"\nConverted JSON to CSV and saved to {self.file_path}")
        print(f"Final shape: {result_df.shape}")
        print(f"Final columns: {list(result_df.columns)}")
        
        print("\nSample data:")
        print(result_df.head())
        
        return result_df
        
    
    def json_to_csv_all(self):
        """
        將所有股票的 JSON 基本面資料轉換為 CSV 格式
        """
        ticker_list = self.config.get('data', {}).get('ticker_list', [])
        
        if not ticker_list:
            print("No ticker list found in config file.")
            return
        
        print(f"Converting JSON to CSV for {len(ticker_list)} tickers...")
        
        successful_conversions = []
        failed_conversions = []
        
        for ticker in ticker_list:
            print(f"\n{'='*40}")
            print(f"Converting {ticker}...")
            print(f"{'='*40}")
            
            try:
                ticker_data = FundamentalData(ticker, self.start_date.strftime('%Y-%m-%d'), 
                                            self.end_date.strftime('%Y-%m-%d'), self.data_dir)
                ticker_data.config = self.config
                ticker_data.api_key = self.api_key
                ticker_data.indicator_list = self.indicator_list
                
                result = ticker_data.json_to_csv()
                
                if result is not None and not result.empty:
                    successful_conversions.append(ticker)
                    print(f"✓ Successfully converted {ticker}")
                else:
                    failed_conversions.append(ticker)
                    print(f"✗ Failed to convert {ticker} - No valid data")
                    
            except Exception as e:
                failed_conversions.append(ticker)
                print(f"✗ Error converting {ticker}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print("CONVERSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(ticker_list)}, Success: {len(successful_conversions)}, Failed: {len(failed_conversions)}")
        
        if successful_conversions:
            print(f"\nSuccessfully converted: {', '.join(successful_conversions)}")
        
        if failed_conversions:
            print(f"\nFailed to convert: {', '.join(failed_conversions)}")

    def check_existing_data(self) -> Dict[str, dict]:
        """
        檢查所有股票的資料狀態
        
        Returns:
            Dict 包含每支股票的資料狀態
        """
        ticker_list = self.config.get('data', {}).get('ticker_list', [])
        
        if not ticker_list:
            print("No ticker list found in config file.")
            return {}
        
        status = {}
        
        print(f"\n{'='*60}")
        print("DATA STATUS CHECK")
        print(f"{'='*60}")
        
        for ticker in ticker_list:
            json_path = os.path.join(self.data_dir, f"{ticker}_fundamentals.json")
            csv_path = os.path.join(self.data_dir, f"{ticker}_fundamentals.csv")
            
            ticker_status = {
                'json_exists': os.path.exists(json_path),
                'csv_exists': os.path.exists(csv_path),
                'json_valid': False,
                'json_records': 0,
                'csv_records': 0,
            }
            
            # 檢查 JSON 檔案
            if ticker_status['json_exists']:
                try:
                    import json
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    # 計算記錄數
                    for report_type in ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']:
                        if report_type in data:
                            quarterly = len(data[report_type].get('quarterlyReports', []))
                            annual = len(data[report_type].get('annualReports', []))
                            ticker_status['json_records'] = max(
                                ticker_status['json_records'], 
                                quarterly + annual
                            )
                    
                    ticker_status['json_valid'] = ticker_status['json_records'] > 0
                    
                except Exception:
                    ticker_status['json_valid'] = False
            
            # 檢查 CSV 檔案
            if ticker_status['csv_exists']:
                try:
                    df = pd.read_csv(csv_path)
                    ticker_status['csv_records'] = len(df)
                except Exception:
                    pass
            
            status[ticker] = ticker_status
            
            # 顯示狀態
            json_icon = "✓" if ticker_status['json_valid'] else ("⚠" if ticker_status['json_exists'] else "✗")
            csv_icon = "✓" if ticker_status['csv_records'] > 0 else "✗"
            
            print(f"{ticker:6s} | JSON: {json_icon} ({ticker_status['json_records']:3d} records) | "
                  f"CSV: {csv_icon} ({ticker_status['csv_records']:3d} records)")
        
        # 摘要
        total = len(ticker_list)
        json_valid = sum(1 for s in status.values() if s['json_valid'])
        csv_valid = sum(1 for s in status.values() if s['csv_records'] > 0)
        
        print(f"{'='*60}")
        print(f"Summary: JSON valid: {json_valid}/{total} | CSV valid: {csv_valid}/{total}")
        print(f"{'='*60}\n")
        
        return status

    def download_missing_data(self):
        """
        只下載缺失或無效的資料
        """
        status = self.check_existing_data()
        
        missing_tickers = [
            ticker for ticker, s in status.items() 
            if not s['json_valid']
        ]
        
        if not missing_tickers:
            print("✓ All data is already downloaded and valid!")
            return
        
        print(f"\nNeed to download {len(missing_tickers)} tickers: {missing_tickers}")
        
        for i, ticker in enumerate(missing_tickers, 1):
            print(f"\n[{i}/{len(missing_tickers)}] Downloading {ticker}...")
            
            ticker_data = FundamentalData(
                ticker, 
                self.start_date.strftime('%Y-%m-%d'),
                self.end_date.strftime('%Y-%m-%d'), 
                self.data_dir
            )
            ticker_data.config = self.config
            ticker_data.api_key = self.api_key
            ticker_data.indicator_list = self.indicator_list
            
            try:
                ticker_data.download_fundamental_data(force=True)
                print(f"✓ Downloaded {ticker}")
            except Exception as e:
                print(f"✗ Failed to download {ticker}: {e}")
            
            if i < len(missing_tickers):
                print("Waiting 30 seconds for API rate limit...")
                time.sleep(30)
        
        print("\n✓ Finished downloading missing data!")
    
    def align_to_daily_with_ma(self, daily_dates: pd.DatetimeIndex,
                               ma_window: int = 63,
                               method: str = 'ema') -> pd.DataFrame:
        """
        將季度基本面資料對齊到日線，並使用移動平均展平
        
        Args:
            daily_dates: 日線資料的日期索引 (pd.DatetimeIndex)
            ma_window: 移動平均窗口大小（預設 63 天，約一季）
            method: 移動平均方法
                - 'sma': 簡單移動平均 (Simple Moving Average)
                - 'ema': 指數移動平均 (Exponential Moving Average)
                - 'wma': 加權移動平均 (Weighted Moving Average)
                - 'interpolate': 線性內插後再平滑
        
        Returns:
            對齊並平滑後的日線基本面 DataFrame
        """
        # 先使用 forward fill 對齊到日線
        aligned_df = self.align_to_daily(daily_dates, method='ffill')
        
        if aligned_df.empty:
            return aligned_df
        
        # 取得數值欄位
        numeric_cols = aligned_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return aligned_df
        
        print(f"  使用 {method.upper()} (窗口={ma_window}) 平滑基本面資料...")
        
        # 設定日期為索引以便計算移動平均
        df_indexed = aligned_df.set_index('date')
        
        # 對每個數值欄位應用移動平均
        for col in numeric_cols:
            original_series = df_indexed[col].copy()
            
            if method == 'sma':
                # 簡單移動平均
                smoothed = original_series.rolling(
                    window=ma_window, 
                    min_periods=1,
                    center=False
                ).mean()
                
            elif method == 'ema':
                # 指數移動平均（對最近的數據給予更高權重）
                smoothed = original_series.ewm(
                    span=ma_window,
                    min_periods=1,
                    adjust=True
                ).mean()
                
            elif method == 'wma':
                # 加權移動平均
                weights = np.arange(1, ma_window + 1)
                smoothed = original_series.rolling(
                    window=ma_window,
                    min_periods=1
                ).apply(
                    lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(),
                    raw=True
                )
                
            elif method == 'interpolate':
                # 先線性內插季度數據，再用移動平均平滑
                # 找出原始季度報告的日期（數值變化的點）
                changes = original_series.diff().fillna(0) != 0
                quarterly_values = original_series[changes | (changes.index == changes.index[0])]
                
                # 線性內插
                interpolated = original_series.copy()
                interpolated.loc[:] = np.nan
                interpolated.loc[quarterly_values.index] = quarterly_values
                interpolated = interpolated.interpolate(method='linear')
                interpolated = interpolated.ffill().bfill()  # 處理邊界
                
                # 再用 EMA 平滑
                smoothed = interpolated.ewm(
                    span=ma_window // 2,  # 內插後用較短窗口
                    min_periods=1
                ).mean()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            df_indexed[col] = smoothed
        
        # 重設索引
        result = df_indexed.reset_index()
        
        return result
    
    def align_to_daily(self, daily_dates: pd.DatetimeIndex, 
                       method: str = 'ffill') -> pd.DataFrame:
        """
        將季度基本面資料對齊到日線資料
        
        使用 Forward Fill 方法：每個交易日使用最近一期已公布的基本面數據
        
        Args:
            daily_dates: 日線資料的日期索引 (pd.DatetimeIndex)
            method: 填充方法，'ffill' (向前填充) 或 'bfill' (向後填充)
        
        Returns:
            對齊到日線的基本面 DataFrame
        """
        # 讀取 CSV 檔案
        if not os.path.exists(self.file_path):
            print(f"Warning: {self.file_path} not found")
            return pd.DataFrame()
        
        df = pd.read_csv(self.file_path)
        
        # 確保日期欄位名稱統一
        date_col = None
        for col in ['fiscalDateEnding', 'date', 'Date']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            print(f"Warning: No date column found in {self.file_path}")
            return pd.DataFrame()
        
        # 轉換日期格式
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # 設定日期為索引
        df = df.set_index(date_col)
        
        # 移除非數值欄位（除了 ticker）
        ticker_col = df['ticker'].iloc[0] if 'ticker' in df.columns else self.ticker
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df[numeric_cols]
        
        # 創建日線日期的空 DataFrame
        daily_df = pd.DataFrame(index=daily_dates)
        daily_df.index.name = 'date'
        
        # 合併季度資料到日線
        # 使用 reindex 並向前填充 (ffill)
        aligned_df = df_numeric.reindex(daily_df.index.union(df_numeric.index))
        aligned_df = aligned_df.sort_index()
        
        if method == 'ffill':
            aligned_df = aligned_df.ffill()
        elif method == 'bfill':
            aligned_df = aligned_df.bfill()
        
        # 只保留日線日期
        aligned_df = aligned_df.loc[daily_dates]
        
        # 加回 ticker 欄位
        aligned_df['ticker'] = ticker_col
        
        # 重設索引
        aligned_df = aligned_df.reset_index()
        aligned_df = aligned_df.rename(columns={'index': 'date'})
        
        return aligned_df

    def align_to_daily_adaptive(self, daily_dates: pd.DatetimeIndex,
                                base_window: int = 63) -> pd.DataFrame:
        """
        自適應移動平均：根據季報間隔自動調整窗口大小
        
        在季報發布日附近使用較短窗口（快速反應），
        在季報之間使用較長窗口（保持穩定）
        
        Args:
            daily_dates: 日線日期索引
            base_window: 基礎窗口大小
        
        Returns:
            自適應平滑後的 DataFrame
        """
        # 讀取原始季度資料
        if not os.path.exists(self.file_path):
            print(f"Warning: {self.file_path} not found")
            return pd.DataFrame()
        
        quarterly_df = pd.read_csv(self.file_path)
        
        # 統一日期欄位
        date_col = 'fiscalDateEnding' if 'fiscalDateEnding' in quarterly_df.columns else 'date'
        quarterly_df[date_col] = pd.to_datetime(quarterly_df[date_col])
        quarterly_dates = set(quarterly_df[date_col])
        
        # 先對齊到日線
        aligned_df = self.align_to_daily(daily_dates, method='ffill')
        
        if aligned_df.empty:
            return aligned_df
        
        numeric_cols = aligned_df.select_dtypes(include=[np.number]).columns.tolist()
        df_indexed = aligned_df.set_index('date')
        
        print(f"  使用自適應移動平均平滑資料...")
        print(f"  季報日期數量: {len(quarterly_dates)}")
        
        for col in numeric_cols:
            original = df_indexed[col].copy()
            smoothed = original.copy()
            
            # 計算每個日期距離最近季報的天數
            for i, (date, value) in enumerate(original.items()):
                # 找最近的季報日期
                days_since_report = min(
                    abs((date - qd).days) 
                    for qd in quarterly_dates 
                    if qd <= date
                ) if any(qd <= date for qd in quarterly_dates) else base_window
                
                # 根據距離調整窗口：距離季報越近，窗口越短
                adaptive_window = max(5, min(base_window, days_since_report + 5))
                
                # 計算當前位置的 EMA
                start_idx = max(0, i - adaptive_window)
                window_data = original.iloc[start_idx:i+1]
                
                if len(window_data) > 0:
                    # 使用 EMA 權重
                    alpha = 2 / (len(window_data) + 1)
                    weights = [(1 - alpha) ** j for j in range(len(window_data))][::-1]
                    weights = np.array(weights) / sum(weights)
                    smoothed.iloc[i] = np.dot(window_data.values, weights)
            
            df_indexed[col] = smoothed
        
        return df_indexed.reset_index()
    
    @staticmethod
    def align_all_tickers_to_daily_with_ma(
        ticker_list: List[str],
        daily_dates: pd.DatetimeIndex,
        data_dir: str = 'data/fundamentals',
        ma_window: int = 63,
        method: str = 'ema'
    ) -> pd.DataFrame:
        """
        將所有股票的基本面資料對齊到日線並使用移動平均展平
        
        Args:
            ticker_list: 股票代碼列表
            daily_dates: 日線日期索引
            data_dir: 資料目錄
            ma_window: 移動平均窗口
            method: 移動平均方法 ('sma', 'ema', 'wma', 'interpolate')
        
        Returns:
            合併後的 DataFrame
        """
        all_aligned = []
        
        print(f"\n{'='*60}")
        print(f"對齊基本面資料到日線 (使用 {method.upper()} 移動平均)")
        print(f"{'='*60}")
        print(f"移動平均窗口: {ma_window} 天")
        print(f"日線日期範圍: {daily_dates.min()} ~ {daily_dates.max()}")
        print()
        
        for ticker in ticker_list:
            try:
                fund_data = FundamentalData(
                    ticker=ticker,
                    start_date=str(daily_dates.min().date()),
                    end_date=str(daily_dates.max().date()),
                    data_dir=data_dir
                )
                
                aligned = fund_data.align_to_daily_with_ma(
                    daily_dates, 
                    ma_window=ma_window,
                    method=method
                )
                
                if not aligned.empty:
                    all_aligned.append(aligned)
                    print(f"  ✓ {ticker}: {len(aligned)} 筆日線資料 (已平滑)")
                else:
                    print(f"  ⚠ {ticker}: 無有效資料")
                    
            except Exception as e:
                print(f"  ✗ {ticker}: 錯誤 - {e}")
        
        if all_aligned:
            result = pd.concat(all_aligned, ignore_index=True)
            print(f"\n{'='*60}")
            print(f"總計: {len(result)} 筆資料")
            print(f"{'='*60}\n")
            return result
        else:
            return pd.DataFrame()
    @staticmethod
    def _get_daily_dates_from_stock_data(ticker: str, stock_data_dir: str,
                                     start_date: str, end_date: str) -> Optional[pd.DatetimeIndex]:
        """
        從股票數據獲取日線日期
        
        Args:
            ticker: 股票代碼
            stock_data_dir: 股票資料目錄
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
        
        Returns:
            日線日期索引，如果失敗則返回 None
        """
        try:
            file_path = os.path.join(stock_data_dir, f"{ticker}_{start_date}_{end_date}.csv")
            
            if not os.path.exists(file_path):
                print(f"Error: Stock data file not found: {file_path}")
                return None
            
            df = pd.read_csv(file_path)
            
            # 尋找日期欄位
            date_col = None
            for col in ['date', 'Date', 'DATE', 'timestamp', 'Timestamp']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                print(f"Error: No date column found in {file_path}")
                return None
            
            df[date_col] = pd.to_datetime(df[date_col])
            
            # 過濾日期範圍
            df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
            
            if df.empty:
                print(f"Warning: No data found for {ticker} in date range {start_date} to {end_date}")
                return None
            
            dates = pd.DatetimeIndex(sorted(df[date_col].unique()))
            print(f"從 {ticker} 獲取 {len(dates)} 個交易日")
            print(f"日期範圍: {dates.min().date()} ~ {dates.max().date()}")
            
            return dates
            
        except Exception as e:
            print(f"Error loading stock data for {ticker}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Download and process fundamental financial data')
    
    parser.add_argument('--ticker', '-t', default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start-date', '-s', default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', '-e', default='2023-03-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', '-c', default='configs/defaults.yaml', help='Config file path')
    parser.add_argument('--data-dir', '-d', default='data/fundamentals', help='Fundamental data directory (input)')
    parser.add_argument('--output-dir', '-o', default=None, help='Output directory for aligned data (default: same as data-dir)')
    parser.add_argument('--action', '-a',
                       choices=['download', 'process', 'both', 'download-all', 'process-all',
                               'json-to-csv', 'json-to-csv-all', 'check', 'download-missing',
                               'align-daily', 'align-daily-all',
                               'align-daily-ma', 'align-daily-ma-all'],
                       default='both', help='Action to perform')
    parser.add_argument('--all-tickers', action='store_true', help='Process all tickers')
    parser.add_argument('--force', '-f', action='store_true', help='Force re-download even if data exists')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--ma-window', type=int, default=63,
                       help='Moving average window size (default: 63, about one quarter)')
    parser.add_argument('--ma-method', choices=['sma', 'ema', 'wma', 'interpolate'],
                       default='ema',
                       help='Moving average method (default: ema)')
    parser.add_argument('--stock-data-dir', default='data/stock_data',
                       help='Stock data directory for daily alignment')
    
    args = parser.parse_args()
    
    # 設定輸出目錄，如果沒有指定則使用輸入目錄
    output_dir = args.output_dir if args.output_dir else args.data_dir
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    fundamental_data = FundamentalData(args.ticker, args.start_date, args.end_date, args.data_dir)
    api_key = fundamental_data.load_api_key(args.config)
    
    if not api_key:
        print("API key not found. Please check your config file.")
        return

    # ========== 檢查與下載缺失資料 ==========
    if args.action == 'check':
        fundamental_data.check_existing_data()
        return
    
    if args.action == 'download-missing':
        fundamental_data.download_missing_data()
        return
    
    # ========== JSON 轉 CSV ==========
    if args.action == 'json-to-csv':
        fundamental_data.json_to_csv()
        return
    
    if args.action == 'json-to-csv-all':
        fundamental_data.json_to_csv_all()
        return

    # ========== 下載資料 ==========
    if args.action == 'download':
        fundamental_data.download_fundamental_data(force=args.force)
        return
    
    if args.action == 'download-all':
        fundamental_data.download_all_fundamental_data(force=args.force)
        return

    # ========== 處理資料 ==========
    if args.action == 'process':
        data = fundamental_data.load_fundamental_data()
        if data is not None:
            processed_data = fundamental_data.preprocess_fundamental_data()
            if processed_data is not None:
                fundamental_data.save_simple_fundamental_data(processed_data)
        return
    
    if args.action == 'process-all':
        fundamental_data.process_all_fundamental_data()
        return

    # ========== 對齊到日線（不使用移動平均）==========
    if args.action == 'align-daily':
        daily_dates = FundamentalData._get_daily_dates_from_stock_data(
            args.ticker, args.stock_data_dir, args.start_date, args.end_date
        )
        if daily_dates is not None:
            result = fundamental_data.align_to_daily(daily_dates, method='ffill')
            if not result.empty:
                output_path = os.path.join(
                    output_dir,
                    f"{args.ticker}_fundamentals_daily.csv"
                )
                result.to_csv(output_path, index=False)
                print(f"已儲存至: {output_path}")
                print(f"資料形狀: {result.shape}")
        return
    
    if args.action == 'align-daily-all':
        ticker_list = fundamental_data.config.get('data', {}).get('ticker_list', [])
        if not ticker_list:
            print("No ticker list found in config.")
            return
        
        daily_dates = FundamentalData._get_daily_dates_from_stock_data(
            ticker_list[0], args.stock_data_dir, args.start_date, args.end_date
        )
        
        if daily_dates is not None:
            success_count = 0
            fail_count = 0
            
            for ticker in ticker_list:
                try:
                    fund_data = FundamentalData(
                        ticker=ticker,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        data_dir=args.data_dir  # 輸入目錄
                    )
                    aligned = fund_data.align_to_daily(daily_dates, method='ffill')
                    if not aligned.empty:
                        # 每支股票獨立儲存
                        output_path = os.path.join(
                            output_dir,
                            f"{ticker}_fundamentals_daily.csv"
                        )
                        aligned.to_csv(output_path, index=False)
                        print(f"  ✓ {ticker}: {len(aligned)} 筆日線資料 -> {output_path}")
                        success_count += 1
                    else:
                        print(f"  ⚠ {ticker}: 無有效資料")
                        fail_count += 1
                except Exception as e:
                    print(f"  ✗ {ticker}: {e}")
                    fail_count += 1
            
            print(f"\n{'='*60}")
            print(f"完成! 成功: {success_count}, 失敗: {fail_count}")
            print(f"檔案儲存於: {output_dir}")
            print(f"{'='*60}")
        return

    # ========== 對齊到日線（使用移動平均）==========
    if args.action == 'align-daily-ma':
        daily_dates = FundamentalData._get_daily_dates_from_stock_data(
            args.ticker, args.stock_data_dir, args.start_date, args.end_date
        )
        if daily_dates is not None:
            result = fundamental_data.align_to_daily_with_ma(
                daily_dates,
                ma_window=args.ma_window,
                method=args.ma_method
            )
            if not result.empty:
                output_path = os.path.join(
                    output_dir,
                    f"{args.ticker}_fundamentals_daily.csv"
                )
                result.to_csv(output_path, index=False)
                print(f"已儲存至: {output_path}")
                print(f"資料形狀: {result.shape}")
        return
    
    if args.action == 'align-daily-ma-all':
        ticker_list = fundamental_data.config.get('data', {}).get('ticker_list', [])
        if not ticker_list:
            print("No ticker list found in config.")
            return
        
        daily_dates = FundamentalData._get_daily_dates_from_stock_data(
            ticker_list[0], args.stock_data_dir, args.start_date, args.end_date
        )
        
        if daily_dates is not None:
            print(f"\n{'='*60}")
            print(f"對齊基本面資料到日線 (使用 {args.ma_method.upper()} 移動平均)")
            print(f"{'='*60}")
            print(f"移動平均窗口: {args.ma_window} 天")
            print(f"日線日期範圍: {daily_dates.min()} ~ {daily_dates.max()}")
            print(f"輸出目錄: {output_dir}")
            print()
            
            success_count = 0
            fail_count = 0
            
            for ticker in ticker_list:
                try:
                    fund_data = FundamentalData(
                        ticker=ticker,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        data_dir=args.data_dir  # 輸入目錄
                    )
                    
                    aligned = fund_data.align_to_daily_with_ma(
                        daily_dates,
                        ma_window=args.ma_window,
                        method=args.ma_method
                    )
                    
                    if not aligned.empty:
                        # 每支股票獨立儲存
                        output_path = os.path.join(
                            output_dir,
                            f"{ticker}_fundamentals_daily.csv"
                        )
                        aligned.to_csv(output_path, index=False)
                        print(f"  ✓ {ticker}: {len(aligned)} 筆日線資料 -> {output_path}")
                        success_count += 1
                    else:
                        print(f"  ⚠ {ticker}: 無有效資料")
                        fail_count += 1
                        
                except Exception as e:
                    print(f"  ✗ {ticker}: {e}")
                    fail_count += 1
            
            print(f"\n{'='*60}")
            print(f"完成! 成功: {success_count}, 失敗: {fail_count}")
            print(f"檔案儲存於: {output_dir}")
            print(f"{'='*60}")
        return

    # ========== 預設動作：下載 + 處理 ==========
    if args.action == 'both':
        if args.all_tickers:
            fundamental_data.download_all_fundamental_data(force=args.force)
            fundamental_data.process_all_fundamental_data()
        else:
            fundamental_data.download_fundamental_data(force=args.force)
            data = fundamental_data.load_fundamental_data()
            if data is not None:
                processed_data = fundamental_data.preprocess_fundamental_data()
                if processed_data is not None:
                    fundamental_data.save_simple_fundamental_data(processed_data)


if __name__ == "__main__":
    main()