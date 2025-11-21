"""
Data models for fundamental financial data.
"""

import argparse
import os
import requests
import yaml
import pandas as pd
import time

class FundamentalData:

    def __init__(self, ticker, start_date, end_date, data_dir='data/fundamentals'):
        """
        Initializes the FundamentalData class.
        
        Args:
            ticker (str): Stock ticker symbol.
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

    def load_api_key(self, config_path='configs/defaults.yaml'):
        """
        Loads the API key from a YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            str: The API key if found, None otherwise.
        """
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)  # 正確載入 YAML 檔案
                self.api_key = self.config.get('data', {}).get('fundamental', {}).get('api_key', None)
                self.indicator_list = self.config.get('data', {}).get('fundamental', {}).get('indicator_list', [])
        except Exception as e:
            print(f"Error loading API key: {e}")
        return self.api_key
    
    def download_all_fundamental_data(self):
        """
        Downloads fundamental data for all tickers in the config file.
        """
        ticker_list = self.config.get('data', {}).get('ticker_list', [])
        
        if not ticker_list:
            print("No ticker list found in config file.")
            return
        
        original_ticker = self.ticker  # 保存原始 ticker
        
        print(f"Found {len(ticker_list)} tickers to download: {ticker_list}")
        
        for ticker in ticker_list:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}...")
            print(f"{'='*50}")
            
            # 為每個股票創建獨立的實例
            ticker_data = FundamentalData(ticker, self.start_date.strftime('%Y-%m-%d'), 
                                        self.end_date.strftime('%Y-%m-%d'), self.data_dir)
            
            # 載入配置
            ticker_data.config = self.config
            ticker_data.api_key = self.api_key
            ticker_data.indicator_list = self.indicator_list
            
            # 檢查檔案是否已存在
            json_file_path = os.path.join(ticker_data.data_dir, f"{ticker}_fundamentals.json")
            
            if os.path.exists(json_file_path):
                print(f"Data for {ticker} already exists. Skipping download.")
                continue
            
            # 下載數據
            ticker_data.download_fundamental_data()
            
            # 添加延遲避免 API 限制
            
            time.sleep(30)  # Alpha Vantage 免費版每分鐘最多 5 次請求
        
        # 恢復原始 ticker
        self.ticker = original_ticker
        print(f"\nCompleted downloading data for all tickers.")
    
    def download_fundamental_data(self):
        """
        Downloads fundamental data for the specified ticker using the Alpha Vantage API.
        """
        
        if not self.api_key:
            print("API key not loaded. Please call load_api_key() first.")
            return
    
        if not self.indicator_list:
            print("No indicators specified in config file.")
            return
        
        all_data = {}
        
        for list_item in self.indicator_list:
            print(f"Downloading {list_item} data for {self.ticker}...")
            url = f"https://www.alphavantage.co/query?function={list_item.upper()}&symbol={self.ticker}&apikey={self.api_key}"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    
                    # 檢查 API 回應是否有錯誤
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
        
        # 將所有數據保存到 CSV 檔案
        if all_data:
            # 這裡需要根據實際的 API 回應格式來處理數據
            # 暫時保存原始 JSON 數據
            import json
            json_file_path = os.path.join(self.data_dir, f"{self.ticker}_fundamentals.json")
            with open(json_file_path, 'w') as f:
                json.dump(all_data, f, indent=2)
            print(f"Raw fundamental data saved to {json_file_path}")
        else:
            print("No data downloaded.")

    def load_fundamental_data(self):
        """
        Loads the fundamental data from a JSON file.

        Returns:
            pd.DataFrame: DataFrame containing the fundamental data.
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
        
        Returns:
            pd.DataFrame: Preprocessed fundamental data.
        """
        data = self.load_fundamental_data()
        if data is not None:
            # 根據實際需求進行預處理
            # 例如，將數據轉換為 DataFrame，處理缺失值等

            # 獲取季報和年報數據
            income_annual = pd.DataFrame(data.get('INCOME_STATEMENT', {}).get('annualReports', []))
            income_quarterly = pd.DataFrame(data.get('INCOME_STATEMENT', {}).get('quarterlyReports', []))
            balance_annual = pd.DataFrame(data.get('BALANCE_SHEET', {}).get('annualReports', []))
            balance_quarterly = pd.DataFrame(data.get('BALANCE_SHEET', {}).get('quarterlyReports', []))
            cashflow_annual = pd.DataFrame(data.get('CASH_FLOW', {}).get('annualReports', []))
            cashflow_quarterly = pd.DataFrame(data.get('CASH_FLOW', {}).get('quarterlyReports', []))

            
            # 將數據合併為一個 DataFrame
            income_df = income_quarterly[['fiscalDateEnding', 'netIncome', 'ebit']].copy()
            balance_df = balance_quarterly[['fiscalDateEnding', 'totalCurrentAssets', 'totalCurrentLiabilities', 'totalLiabilities', 'totalAssets', 'commonStock', 'retainedEarnings', 'cashAndCashEquivalentsAtCarryingValue', 'cashAndShortTermInvestments', 'currentNetReceivables', 'inventory']].copy()
            cashflow_df = cashflow_quarterly[['fiscalDateEnding', 'operatingCashflow']].copy()

            # 刪除時間外的數據
            income_df['fiscalDateEnding'] = pd.to_datetime(income_df['fiscalDateEnding'])
            balance_df['fiscalDateEnding'] = pd.to_datetime(balance_df['fiscalDateEnding'])
            cashflow_df['fiscalDateEnding'] = pd.to_datetime(cashflow_df['fiscalDateEnding'])
            mask_income = (income_df['fiscalDateEnding'] >= self.start_date) & (income_df['fiscalDateEnding'] <= self.end_date)
            mask_balance = (balance_df['fiscalDateEnding'] >= self.start_date) & (balance_df['fiscalDateEnding'] <= self.end_date)
            mask_cashflow = (cashflow_df['fiscalDateEnding'] >= self.start_date) & (cashflow_df['fiscalDateEnding'] <= self.end_date)
            income_df = income_df.loc[mask_income].copy()
            balance_df = balance_df.loc[mask_balance].copy()
            cashflow_df = cashflow_df.loc[mask_cashflow].copy()
            merged_df = income_df.merge(balance_df, on='fiscalDateEnding').merge(cashflow_df, on='fiscalDateEnding')
            merged_df.set_index('fiscalDateEnding', inplace=True)
            # merged_df = income_df.merge(balance_df, on='fiscalDateEnding', how='inner').merge(cashflow_df, on='fiscalDateEnding', how='inner')
            # numeric_columns = [col for col in merged_df.columns if col != 'fiscalDateEnding']
            # for col in numeric_columns:
            #     merged_df[col] = merged_df[col].replace('None', pd.NA)
            #     merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
            # print("Preprocessed fundamental data:")
            # print(merged_df.head())
            return merged_df
        else:
            print("No data to preprocess.")
            return None
        
    def calculate_financial_ratios_from_df(self, df):
        """
        從 DataFrame 計算財務比率
        
        Args:
            df (pd.DataFrame): 包含財務數據的 DataFrame
        
        Returns:
            pd.DataFrame: 包含計算出的財務比率的 DataFrame
        """
        df_ratios = df.copy()
        
        print("Available columns in DataFrame:")
        print([col for col in df.columns if col not in ['fiscalDateEnding', 'ticker']])
        
        try:
            # 計算各種財務比率
            df_ratios['Current Ratio'] = self._calculate_current_ratio(df)
            df_ratios['Acid Test Ratio'] = self._calculate_acid_test_ratio(df)
            df_ratios['Operating Cash Flow Ratio'] = self._calculate_operating_cash_flow_ratio(df)
            df_ratios['Debt Ratio'] = self._calculate_debt_ratio(df)
            df_ratios['Debt to Equity Ratio'] = self._calculate_debt_to_equity_ratio(df)
            df_ratios['Interest Coverage Ratio'] = self._calculate_interest_coverage_ratio(df)
            df_ratios['Asset Turnover Ratio'] = self._calculate_asset_turnover_ratio(df)
            df_ratios['Inventory Turnover Ratio'] = self._calculate_inventory_turnover_ratio(df)
            df_ratios['Day Sales in Inventory Ratio'] = self._calculate_day_sales_in_inventory_ratio(df_ratios)
            df_ratios['Return on Ratio'] = self._calculate_return_on_assets_ratio(df)
            df_ratios['Return on Equity Ratio'] = self._calculate_return_on_equity_ratio(df)
            
            # 計算額外的有用比率
            df_ratios['Gross Profit Margin'] = self._calculate_gross_profit_margin(df)
            df_ratios['Net Profit Margin'] = self._calculate_net_profit_margin(df)
            df_ratios['Operating Margin'] = self._calculate_operating_margin(df)
            
            calculated_ratios = [col for col in df_ratios.columns if col not in df.columns]
            print(f"\nSuccessfully calculated {len(calculated_ratios)} financial ratios:")
            for ratio in calculated_ratios:
                print(f"  - {ratio}")
            
        except Exception as e:
            print(f"Error calculating financial ratios: {e}")
            import traceback
            traceback.print_exc()
        
        return df_ratios

    def _calculate_current_ratio(self, df):
        """
        計算流動比率 = 總流動資產 / 總流動負債
        """
        try:
            if 'balance_totalCurrentAssets' in df.columns and 'balance_totalCurrentLiabilities' in df.columns:
                ratio = (df['balance_totalCurrentAssets'] / df['balance_totalCurrentLiabilities'])
                print("✓ Calculated Current Ratio")
                return ratio
            else:
                print("✗ Cannot calculate Current Ratio - missing columns")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Current Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_acid_test_ratio(self, df):
        """
        計算速動比率 = (現金 + 短期投資 + 應收帳款) / 總流動負債
        """
        try:
            cash_col = self._find_column(df, ['cash', 'carrying'])
            short_inv_col = self._find_column(df, ['shortterm', 'short'])
            receivables_col = self._find_column(df, ['receivable', 'net'])
            
            if cash_col and 'balance_totalCurrentLiabilities' in df.columns:
                cash = df[cash_col].fillna(0)
                short_inv = df[short_inv_col].fillna(0) if short_inv_col else pd.Series([0] * len(df))
                receivables = df[receivables_col].fillna(0) if receivables_col else pd.Series([0] * len(df))
                
                quick_assets = cash + short_inv + receivables
                ratio = quick_assets / df['balance_totalCurrentLiabilities']
                print("✓ Calculated Acid Test Ratio")
                return ratio
            else:
                print("✗ Cannot calculate Acid Test Ratio - missing columns")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Acid Test Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_operating_cash_flow_ratio(self, df):
        """
        計算營業現金流量比率 = 營業現金流量 / 總流動負債
        """
        try:
            if ('cashflow_operatingCashflow' in df.columns and 
                'balance_totalCurrentLiabilities' in df.columns):
                ratio = (df['cashflow_operatingCashflow'] / df['balance_totalCurrentLiabilities'])
                print("✓ Calculated Operating Cash Flow Ratio")
                return ratio
            else:
                print("✗ Cannot calculate Operating Cash Flow Ratio - missing columns")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Operating Cash Flow Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_debt_ratio(self, df):
        """
        計算負債比率 = 總負債 / 總資產
        """
        try:
            if 'balance_totalLiabilities' in df.columns and 'balance_totalAssets' in df.columns:
                ratio = (df['balance_totalLiabilities'] / df['balance_totalAssets'])
                print("✓ Calculated Debt Ratio")
                return ratio
            else:
                print("✗ Cannot calculate Debt Ratio - missing columns")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Debt Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_debt_to_equity_ratio(self, df):
        """
        計算負債股權比率 = 總負債 / 總股東權益
        """
        try:
            if ('balance_totalLiabilities' in df.columns and 
                'balance_totalShareholderEquity' in df.columns):
                ratio = (df['balance_totalLiabilities'] / df['balance_totalShareholderEquity'])
                print("✓ Calculated Debt to Equity Ratio")
                return ratio
            else:
                print("✗ Cannot calculate Debt to Equity Ratio - missing columns")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Debt to Equity Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_interest_coverage_ratio(self, df):
        """
        計算利息保障倍數 = EBIT / 利息費用
        """
        try:
            ebit_col = self._find_column(df, ['ebit'], exclude=['ebitda'])
            interest_col = self._find_column(df, ['interest'], include=['expense', 'debt'])
            
            if ebit_col and interest_col:
                # 處理利息費用為0或None的情況
                interest_expense = df[interest_col].replace(0, pd.NA)
                ratio = df[ebit_col] / interest_expense
                print("✓ Calculated Interest Coverage Ratio")
                return ratio
            else:
                print(f"✗ Cannot calculate Interest Coverage Ratio - EBIT col: {ebit_col}, Interest col: {interest_col}")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Interest Coverage Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_asset_turnover_ratio(self, df):
        """
        計算資產周轉率 = 總營收 / 總資產
        """
        try:
            revenue_col = self._find_column(df, ['revenue', 'total'])
            
            if revenue_col and 'balance_totalAssets' in df.columns:
                ratio = df[revenue_col] / df['balance_totalAssets']
                print("✓ Calculated Asset Turnover Ratio")
                return ratio
            else:
                print(f"✗ Cannot calculate Asset Turnover Ratio - Revenue col: {revenue_col}")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Asset Turnover Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_inventory_turnover_ratio(self, df):
        """
        計算存貨周轉率 = 銷貨成本 / 存貨
        """
        try:
            cost_col = self._find_column(df, ['cost'], include=['revenue', 'goods'])
            inventory_col = self._find_column(df, ['inventory'])
            
            if cost_col and inventory_col:
                # 處理庫存為0的情況
                inventory = df[inventory_col].replace(0, pd.NA)
                ratio = df[cost_col] / inventory
                print("✓ Calculated Inventory Turnover Ratio")
                return ratio
            else:
                print(f"✗ Cannot calculate Inventory Turnover Ratio - Cost col: {cost_col}, Inventory col: {inventory_col}")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Inventory Turnover Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_day_sales_in_inventory_ratio(self, df):
        """
        計算存貨銷售天數 = 365 / 存貨周轉率
        """
        try:
            if 'Inventory Turnover Ratio' in df.columns:
                ratio = 365 / df['Inventory Turnover Ratio']
                print("✓ Calculated Day Sales in Inventory Ratio")
                return ratio
            else:
                print("✗ Cannot calculate Day Sales in Inventory Ratio - Inventory Turnover Ratio not available")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Day Sales in Inventory Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_return_on_assets_ratio(self, df):
        """
        計算資產報酬率 = 淨利 / 總資產
        """
        try:
            net_income_col = self._find_column(df, ['netincome'])
            
            if net_income_col and 'balance_totalAssets' in df.columns:
                ratio = df[net_income_col] / df['balance_totalAssets']
                print("✓ Calculated Return on Assets Ratio")
                return ratio
            else:
                print(f"✗ Cannot calculate Return on Assets Ratio - Net Income col: {net_income_col}")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Return on Assets Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_return_on_equity_ratio(self, df):
        """
        計算股東權益報酬率 = 淨利 / 總股東權益
        """
        try:
            net_income_col = self._find_column(df, ['netincome'])
            
            if (net_income_col and 'balance_totalShareholderEquity' in df.columns):
                ratio = df[net_income_col] / df['balance_totalShareholderEquity']
                print("✓ Calculated Return on Equity Ratio")
                return ratio
            else:
                print(f"✗ Cannot calculate Return on Equity Ratio - Net Income col: {net_income_col}")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Return on Equity Ratio: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_gross_profit_margin(self, df):
        """
        計算毛利率 = 毛利 / 總營收 * 100
        """
        try:
            gross_profit_col = self._find_column(df, ['gross', 'profit'])
            revenue_col = self._find_column(df, ['revenue', 'total'])
            
            if gross_profit_col and revenue_col:
                ratio = (df[gross_profit_col] / df[revenue_col]) * 100
                print("✓ Calculated Gross Profit Margin")
                return ratio
            else:
                print("✗ Cannot calculate Gross Profit Margin - missing columns")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Gross Profit Margin: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_net_profit_margin(self, df):
        """
        計算淨利率 = 淨利 / 總營收 * 100
        """
        try:
            net_income_col = self._find_column(df, ['netincome'])
            revenue_col = self._find_column(df, ['revenue', 'total'])
            
            if net_income_col and revenue_col:
                ratio = (df[net_income_col] / df[revenue_col]) * 100
                print("✓ Calculated Net Profit Margin")
                return ratio
            else:
                print("✗ Cannot calculate Net Profit Margin - missing columns")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Net Profit Margin: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _calculate_operating_margin(self, df):
        """
        計算營業利潤率 = 營業收入 / 總營收 * 100
        """
        try:
            operating_income_col = self._find_column(df, ['operating', 'income'])
            revenue_col = self._find_column(df, ['revenue', 'total'])
            
            if operating_income_col and revenue_col:
                ratio = (df[operating_income_col] / df[revenue_col]) * 100
                print("✓ Calculated Operating Margin")
                return ratio
            else:
                print("✗ Cannot calculate Operating Margin - missing columns")
                return pd.Series([None] * len(df), index=df.index)
        except Exception as e:
            print(f"Error calculating Operating Margin: {e}")
            return pd.Series([None] * len(df), index=df.index)

    def _find_column(self, df, required_terms, include=None, exclude=None):
        """
        在 DataFrame 中尋找包含指定關鍵字的欄位
        
        Args:
            df: DataFrame
            required_terms: 必須包含的關鍵字列表
            include: 可選包含的關鍵字列表
            exclude: 必須排除的關鍵字列表
        
        Returns:
            str: 找到的欄位名稱，如果沒找到則返回 None
        """
        for col in df.columns:
            col_lower = col.lower().replace('_', '')
            
            # 檢查是否包含所有必需的關鍵字
            has_required = all(term.lower() in col_lower for term in required_terms)
            
            # 檢查排除的關鍵字
            has_excluded = False
            if exclude:
                has_excluded = any(term.lower() in col_lower for term in exclude)
            
            # 檢查可選的關鍵字
            has_included = True
            if include:
                has_included = any(term.lower() in col_lower for term in include)
            
            if has_required and not has_excluded and has_included:
                return col
        
        return None
    
    def save_simple_fundamental_data(self, df):
        """
        簡化版的保存方法，保存基本面資料到 CSV 檔案
        
        Args:
            df (pd.DataFrame): DataFrame containing fundamental data.
        """
        if df is None or df.empty:
            print("No data to save.")
            return
        
        try:
            print(f"Saving data for {self.ticker}...")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # 添加股票代碼
            df_to_save = df.copy()
            df_to_save['ticker'] = self.ticker
            
            # 重置索引，保留 fiscalDateEnding 作為欄位
            if df_to_save.index.name == 'fiscalDateEnding':
                df_to_save.reset_index(inplace=True)
            
            # 保存數據
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
                # 為每個股票創建獨立的實例
                ticker_data = FundamentalData(ticker, self.start_date.strftime('%Y-%m-%d'), 
                                            self.end_date.strftime('%Y-%m-%d'), self.data_dir)
                
                # 載入配置
                ticker_data.config = self.config
                ticker_data.api_key = self.api_key
                ticker_data.indicator_list = self.indicator_list
                
                # 載入和處理數據
                raw_data = ticker_data.load_fundamental_data()
                
                if raw_data is not None:
                    # 檢查是否有實際的財務數據
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
        if data is not None:
            ticker = data.get('INCOME_STATEMENT', {}).get('symbol', self.ticker)
            print(f"Converting data for {ticker}...")
            
            # 獲取配置中的特徵列表
            target_features = self.config.get('data', {}).get('fundamental', {}).get('features', [])
            if not target_features:
                print("No features specified in config file.")
                return None
            
            print(f"Target features: {target_features}")
            
            # 合併所有季度報告數據
            all_quarterly_data = []
            
            # 處理損益表季度數據
            income_quarterly = data.get('INCOME_STATEMENT', {}).get('quarterlyReports', [])
            balance_quarterly = data.get('BALANCE_SHEET', {}).get('quarterlyReports', [])
            cashflow_quarterly = data.get('CASH_FLOW', {}).get('quarterlyReports', [])
            
            print(f"Found {len(income_quarterly)} income quarterly reports")
            print(f"Found {len(balance_quarterly)} balance quarterly reports")
            print(f"Found {len(cashflow_quarterly)} cashflow quarterly reports")
            
            # 將三個報表的數據按日期合併
            quarterly_data_by_date = {}
            
            # 合併損益表數據
            for report in income_quarterly:
                date = report.get('fiscalDateEnding')
                if date:
                    if date not in quarterly_data_by_date:
                        quarterly_data_by_date[date] = {'fiscalDateEnding': date, 'ticker': ticker}
                    # 添加損益表欄位
                    for key, value in report.items():
                        if key != 'fiscalDateEnding':
                            quarterly_data_by_date[date][f"income_{key}"] = value
            
            # 合併資產負債表數據
            for report in balance_quarterly:
                date = report.get('fiscalDateEnding')
                if date and date in quarterly_data_by_date:
                    # 添加資產負債表欄位
                    for key, value in report.items():
                        if key != 'fiscalDateEnding':
                            quarterly_data_by_date[date][f"balance_{key}"] = value
            
            # 合併現金流量表數據
            for report in cashflow_quarterly:
                date = report.get('fiscalDateEnding')
                if date and date in quarterly_data_by_date:
                    # 添加現金流量表欄位
                    for key, value in report.items():
                        if key != 'fiscalDateEnding':
                            quarterly_data_by_date[date][f"cashflow_{key}"] = value
            
            # 轉換為 DataFrame
            if not quarterly_data_by_date:
                print("No quarterly data available.")
                return None
            
            df = pd.DataFrame(list(quarterly_data_by_date.values()))
            
            # 轉換日期格式並排序
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.sort_values('fiscalDateEnding', ascending=False)
            
            # 數據清理 - 將 'None' 字符串轉換為 NaN，然後轉換為數值
            numeric_columns = df.select_dtypes(include=['object']).columns
            for col in numeric_columns:
                if col not in ['fiscalDateEnding', 'ticker']:
                    df[col] = df[col].replace('None', pd.NA)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
            print(f"Date range: {df['fiscalDateEnding'].min()} to {df['fiscalDateEnding'].max()}")
            
            # 計算財務比率
            df_with_ratios = self.calculate_financial_ratios_from_df(df)
            
            # 選擇配置中指定的特徵
            available_features = []
            feature_columns = ['fiscalDateEnding', 'ticker']
            
            for feature in target_features:
                # 檢查每個特徵是否在計算出的比率中
                feature_found = False
                for col in df_with_ratios.columns:
                    # 使用更靈活的匹配方式
                    if (feature.lower().replace(' ', '').replace('ratio', '') in col.lower().replace(' ', '').replace('ratio', '') or
                        col.lower().replace(' ', '').replace('ratio', '') in feature.lower().replace(' ', '').replace('ratio', '')):
                        feature_columns.append(col)
                        available_features.append(feature)
                        feature_found = True
                        break
                
                if not feature_found:
                    print(f"Warning: Feature '{feature}' not found in calculated ratios")
            
            # 選擇最終的欄位
            final_columns = list(set(feature_columns))  # 去除重複
            final_columns.sort(key=lambda x: (x != 'fiscalDateEnding', x != 'ticker', x))
            
            # 確保所有欄位都存在
            existing_columns = [col for col in final_columns if col in df_with_ratios.columns]
            
            if len(existing_columns) <= 2:  # 只有 fiscalDateEnding 和 ticker
                print("Warning: No feature columns found, using all calculated ratios")
                result_df = df_with_ratios
            else:
                result_df = df_with_ratios[existing_columns].copy()
            
            print(f"Selected features: {available_features}")
            print(f"Final DataFrame columns: {list(result_df.columns)}")
            
            # 保存為 CSV
            result_df.to_csv(self.file_path, index=False, encoding='utf-8-sig')
            print(f"Converted JSON to CSV and saved to {self.file_path}")
            print(f"Final shape: {result_df.shape}")
            
            # 顯示樣本數據
            print("\nSample data:")
            print(result_df.head())
            
            return result_df
        else:
            print("No data to convert from JSON to CSV.")
            return None
    
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
                # 為每個股票創建獨立的實例
                ticker_data = FundamentalData(ticker, self.start_date.strftime('%Y-%m-%d'), 
                                            self.end_date.strftime('%Y-%m-%d'), self.data_dir)
                
                # 載入配置
                ticker_data.config = self.config
                ticker_data.api_key = self.api_key
                ticker_data.indicator_list = self.indicator_list
                
                # 執行 JSON 到 CSV 轉換
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
        
        # 總結報告
        print(f"\n{'='*60}")
        print("CONVERSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total tickers processed: {len(ticker_list)}")
        print(f"Successful conversions: {len(successful_conversions)}")
        print(f"Failed conversions: {len(failed_conversions)}")
        
        if successful_conversions:
            print(f"\nSuccessfully converted:")
            for ticker in successful_conversions:
                print(f"  ✓ {ticker}")
        
        if failed_conversions:
            print(f"\nFailed to convert:")
            for ticker in failed_conversions:
                print(f"  ✗ {ticker}")
        
        print(f"\nCompleted JSON to CSV conversion for all tickers.")
        

def main():
    parser = argparse.ArgumentParser(description='Download and process fundamental financial data')
    
    # 基本參數
    parser.add_argument('--ticker', '-t', 
                       default='AAPL',
                       help='Stock ticker symbol (default: AAPL)')
    
    parser.add_argument('--start-date', '-s',
                       default='2010-01-01',
                       help='Start date in YYYY-MM-DD format (default: 2010-01-01)')
    
    parser.add_argument('--end-date', '-e',
                       default='2023-03-01', 
                       help='End date in YYYY-MM-DD format (default: 2023-03-01)')
    
    parser.add_argument('--config', '-c',
                       default='configs/defaults.yaml',
                       help='Path to config file (default: configs/defaults.yaml)')
    
    parser.add_argument('--data-dir', '-d',
                       default='data/fundamentals',
                       help='Directory to save data (default: data/fundamentals)')
    
    # 操作選項
    parser.add_argument('--action', '-a',
                       choices=['download', 'process', 'both', 'download-all', 'process-all',
                               'json-to-csv', 'json-to-csv-all'],
                       default='both',
                       help='Action to perform (default: both)')
    
    # 批量處理選項
    parser.add_argument('--all-tickers', 
                       action='store_true',
                       help='Process all tickers in config file')
    
    # 其他選項
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # 初始化
    fundamental_data = FundamentalData(args.ticker, args.start_date, args.end_date, args.data_dir)
    
    # 載入配置
    api_key = fundamental_data.load_api_key(args.config)
    
    if not api_key:
        print("API key not found. Please check your config file.")
        return
    
    # 根據參數執行不同操作
    if args.action in ['json-to-csv', 'json-to-csv-all']:
        if args.action == 'json-to-csv':
            print(f"Converting JSON to CSV for {args.ticker}...")
            result = fundamental_data.json_to_csv()
            if result is not None:
                print("\nSample of converted data:")
                print(result.head())
        elif args.action == 'json-to-csv-all':
            print("Converting JSON to CSV for all tickers...")
            fundamental_data.json_to_csv_all()
        return

    if args.action == 'download' or args.action == 'both':
        if args.all_tickers:
            print("Downloading data for all tickers...")
            fundamental_data.download_all_fundamental_data()
        else:
            print(f"Downloading data for {args.ticker}...")
            fundamental_data.download_fundamental_data()
    
    if args.action == 'process' or args.action == 'both':
        if args.all_tickers:
            print("Processing data for all tickers...")
            fundamental_data.process_all_fundamental_data()
        else:
            print(f"Processing data for {args.ticker}...")
            data = fundamental_data.load_fundamental_data()
            if data is not None:
                processed_data = fundamental_data.preprocess_fundamental_data()
                if processed_data is not None:
                    fundamental_data.save_simple_fundamental_data(processed_data)
                else:
                    print("Failed to preprocess data")
            else:
                print("No data to process")
    
    if args.action == 'download-all':
        fundamental_data.download_all_fundamental_data()
    
    if args.action == 'process-all':
        fundamental_data.process_all_fundamental_data()


if __name__ == "__main__":
    main()