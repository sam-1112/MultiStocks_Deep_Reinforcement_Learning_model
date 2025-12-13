"""
數據分割工具：從完整的 CSV 數據中分割出訓練集和測試集
支持四種數據類型：股票數據、技術指標、基本面數據、對齊基本面數據
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import yaml

class ComprehensiveDataSplitter:
    """
    綜合數據分割工具
    同時分割股票數據、技術指標、基本面數據和對齊的基本面數據
    """
    
    def __init__(self, 
                 stock_data_dir: str = 'data/stock_data',
                 indicators_dir: str = 'data/indicators',
                 fundamentals_dir: str = 'data/fundamentals',
                 fundamentals_daily_dir: str = 'data/fundamentals/daily_aligned',
                 output_base_dir: str = 'data'):
        """
        初始化綜合數據分割器
        
        :param stock_data_dir: 股票數據目錄
        :param indicators_dir: 技術指標目錄
        :param fundamentals_dir: 基本面數據目錄
        :param fundamentals_daily_dir: 對齊基本面數據目錄
        :param output_base_dir: 輸出基礎目錄
        """
        self.stock_data_dir = stock_data_dir
        self.indicators_dir = indicators_dir
        self.fundamentals_dir = fundamentals_dir
        self.fundamentals_daily_dir = fundamentals_daily_dir
        self.output_base_dir = output_base_dir
        
        # 為四種數據類型創建輸出目錄
        self.stock_output_dir = os.path.join(output_base_dir, 'stock_data')
        self.indicators_output_dir = os.path.join(output_base_dir, 'indicators')
        self.fundamentals_output_dir = os.path.join(output_base_dir, 'fundamentals')
        self.fundamentals_daily_output_dir = os.path.join(output_base_dir, 'fundamentals', 'daily_aligned')
        
        for dir_path in [self.stock_output_dir, self.indicators_output_dir, 
                        self.fundamentals_output_dir, self.fundamentals_daily_output_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def _find_source_file(self, ticker: str, directory: str, suffix: str) -> Optional[str]:
        """
        在目錄中尋找特定股票的源文件
        
        :param ticker: 股票代碼
        :param directory: 搜索目錄
        :param suffix: 文件後綴（如 '_2010-01-01_2023-03-01.csv' 或 '_indicators.csv'）
        :return: 完整文件路徑或 None
        """
        # 優先尋找完整日期範圍的文件
        priority_patterns = [
            f"{ticker}_2010-01-01_2023-03-01.csv",
            f"{ticker}_2010-01-01_2023-03-01.csv",
        ]
        
        for pattern in priority_patterns:
            full_path = os.path.join(directory, pattern)
            if os.path.exists(full_path):
                return full_path
        
        # 其次，尋找匹配後綴的文件
        for file in os.listdir(directory):
            if file.startswith(ticker) and file.endswith(suffix):
                return os.path.join(directory, file)
        
        return None
    
    def split_stock_data(self, 
                        ticker: str, 
                        train_start: str, 
                        train_end: str,
                        test_start: str,
                        test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分割股票 OHLCV 數據"""
        stock_file = self._find_source_file(ticker, self.stock_data_dir, '_2010-01-01_2023-03-01.csv')
        
        if not stock_file:
            raise FileNotFoundError(f"找不到股票 {ticker} 的原始數據文件")
        
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        test_start_dt = pd.to_datetime(test_start)
        test_end_dt = pd.to_datetime(test_end)
        
        data = pd.read_csv(stock_file)
        data['date'] = pd.to_datetime(data['date'])
        
        train_data = data[(data['date'] >= train_start_dt) & (data['date'] <= train_end_dt)].copy()
        test_data = data[(data['date'] >= test_start_dt) & (data['date'] <= test_end_dt)].copy()
        
        return train_data, test_data
    
    def split_indicators_data(self, 
                             ticker: str, 
                             train_start: str, 
                             train_end: str,
                             test_start: str,
                             test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分割技術指標數據"""
        indicators_file = self._find_source_file(ticker, self.indicators_dir, '_indicators.csv')
        
        if not indicators_file:
            raise FileNotFoundError(f"找不到股票 {ticker} 的技術指標文件")
        
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        test_start_dt = pd.to_datetime(test_start)
        test_end_dt = pd.to_datetime(test_end)
        
        data = pd.read_csv(indicators_file)
        data['date'] = pd.to_datetime(data['date'])
        
        train_data = data[(data['date'] >= train_start_dt) & (data['date'] <= train_end_dt)].copy()
        test_data = data[(data['date'] >= test_start_dt) & (data['date'] <= test_end_dt)].copy()
        
        return train_data, test_data
    
    def split_fundamentals_data(self, 
                               ticker: str, 
                               train_start: str, 
                               train_end: str,
                               test_start: str,
                               test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分割基本面數據（按季度）"""
        fundamentals_file = self._find_source_file(ticker, self.fundamentals_dir, '_fundamentals.csv')
        
        if not fundamentals_file:
            raise FileNotFoundError(f"找不到股票 {ticker} 的基本面數據文件")
        
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        test_start_dt = pd.to_datetime(test_start)
        test_end_dt = pd.to_datetime(test_end)
        
        data = pd.read_csv(fundamentals_file)
        
        # 基本面數據使用 fiscalDateEnding 或 date 作為日期列
        date_col = None
        for col in ['fiscalDateEnding', 'date', 'Date']:
            if col in data.columns:
                date_col = col
                break
        
        if not date_col:
            return data.copy(), data.copy()
        
        data[date_col] = pd.to_datetime(data[date_col])
        
        train_data = data[(data[date_col] >= train_start_dt) & (data[date_col] <= train_end_dt)].copy()
        test_data = data[(data[date_col] >= test_start_dt) & (data[date_col] <= test_end_dt)].copy()
        
        return train_data, test_data
    
    def split_fundamentals_daily_data(self, 
                                     ticker: str, 
                                     train_start: str, 
                                     train_end: str,
                                     test_start: str,
                                     test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分割對齊的日度基本面數據
        （已與交易日對齊的基本面數據）
        """
        # 尋找對齊的基本面數據文件
        daily_file = os.path.join(self.fundamentals_daily_dir, f"{ticker}_fundamentals_daily.csv")
        
        if not os.path.exists(daily_file):
            raise FileNotFoundError(f"找不到股票 {ticker} 的對齊基本面數據文件: {daily_file}")
        
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        test_start_dt = pd.to_datetime(test_start)
        test_end_dt = pd.to_datetime(test_end)
        
        # 載入數據
        data = pd.read_csv(daily_file)
        data['date'] = pd.to_datetime(data['date'])
        
        # 分割訓練集和測試集
        train_data = data[(data['date'] >= train_start_dt) & (data['date'] <= train_end_dt)].copy()
        test_data = data[(data['date'] >= test_start_dt) & (data['date'] <= test_end_dt)].copy()
        
        return train_data, test_data
    
    def split_ticker(self, 
                    ticker: str, 
                    train_start: str, 
                    train_end: str,
                    test_start: str,
                    test_end: str) -> Dict[str, Dict]:
        """
        為單個股票分割四種數據類型
        
        :return: 包含分割結果和文件路徑的字典
        """
        results = {'ticker': ticker, 'data_types': {}}
        
        # 1. 分割股票數據
        try:
            train_stock, test_stock = self.split_stock_data(
                ticker, train_start, train_end, test_start, test_end
            )
            
            train_file = os.path.join(self.stock_output_dir, f"{ticker}_{train_start}_{train_end}.csv")
            test_file = os.path.join(self.stock_output_dir, f"{ticker}_{test_start}_{test_end}.csv")
            
            train_stock.to_csv(train_file, index=False)
            test_stock.to_csv(test_file, index=False)
            
            results['data_types']['stock'] = {
                'status': 'success',
                'train': {'file': train_file, 'rows': len(train_stock)},
                'test': {'file': test_file, 'rows': len(test_stock)}
            }
        except Exception as e:
            results['data_types']['stock'] = {'status': 'error', 'message': str(e)}
        
        # 2. 分割技術指標數據
        try:
            train_ind, test_ind = self.split_indicators_data(
                ticker, train_start, train_end, test_start, test_end
            )
            
            train_file = os.path.join(self.indicators_output_dir, f"{ticker}_{train_start}_{train_end}.csv")
            test_file = os.path.join(self.indicators_output_dir, f"{ticker}_{test_start}_{test_end}.csv")
            
            train_ind.to_csv(train_file, index=False)
            test_ind.to_csv(test_file, index=False)
            
            results['data_types']['indicators'] = {
                'status': 'success',
                'train': {'file': train_file, 'rows': len(train_ind)},
                'test': {'file': test_file, 'rows': len(test_ind)}
            }
        except Exception as e:
            results['data_types']['indicators'] = {'status': 'error', 'message': str(e)}
        
        # 3. 分割基本面數據（季度數據）
        try:
            train_fund, test_fund = self.split_fundamentals_data(
                ticker, train_start, train_end, test_start, test_end
            )
            
            train_file = os.path.join(self.fundamentals_output_dir, f"{ticker}_{train_start}_{train_end}.csv")
            test_file = os.path.join(self.fundamentals_output_dir, f"{ticker}_{test_start}_{test_end}.csv")
            
            train_fund.to_csv(train_file, index=False)
            test_fund.to_csv(test_file, index=False)
            
            results['data_types']['fundamentals'] = {
                'status': 'success',
                'train': {'file': train_file, 'rows': len(train_fund)},
                'test': {'file': test_file, 'rows': len(test_fund)}
            }
        except Exception as e:
            results['data_types']['fundamentals'] = {'status': 'error', 'message': str(e)}
        
        # 4. 分割對齊的基本面數據（日度數據）
        try:
            train_daily, test_daily = self.split_fundamentals_daily_data(
                ticker, train_start, train_end, test_start, test_end
            )
            
            train_file = os.path.join(self.fundamentals_daily_output_dir, f"{ticker}_{train_start}_{train_end}.csv")
            test_file = os.path.join(self.fundamentals_daily_output_dir, f"{ticker}_{test_start}_{test_end}.csv")
            
            train_daily.to_csv(train_file, index=False)
            test_daily.to_csv(test_file, index=False)
            
            results['data_types']['fundamentals_daily'] = {
                'status': 'success',
                'train': {'file': train_file, 'rows': len(train_daily)},
                'test': {'file': test_file, 'rows': len(test_daily)}
            }
        except Exception as e:
            results['data_types']['fundamentals_daily'] = {'status': 'error', 'message': str(e)}
        
        return results
    
    def split_all_tickers(self, 
                         tickers: List[str], 
                         train_start: str, 
                         train_end: str,
                         test_start: str,
                         test_end: str) -> Dict:
        """
        分割所有股票的四種數據類型
        
        :param tickers: 股票代碼列表
        :param train_start: 訓練開始日期
        :param train_end: 訓練結束日期
        :param test_start: 測試開始日期
        :param test_end: 測試結束日期
        :return: 所有結果的字典
        """
        print(f"\n{'='*80}")
        print(f"🔄 開始分割綜合數據（含對齊基本面數據）")
        print(f"{'='*80}\n")
        
        print(f"訓練期間: {train_start} 至 {train_end}")
        print(f"測試期間: {test_start} 至 {test_end}")
        print(f"股票數量: {len(tickers)}")
        print(f"\n處理股票:\n")
        
        all_results = {}
        
        for idx, ticker in enumerate(tickers, 1):
            print(f"  [{idx:2d}/{len(tickers)}] {ticker:6s}", end=' | ')
            
            try:
                result = self.split_ticker(ticker, train_start, train_end, test_start, test_end)
                all_results[ticker] = result
                
                # 檢查結果
                successful_types = sum(
                    1 for dt in result['data_types'].values() 
                    if dt.get('status') == 'success'
                )
                print(f"✓ {successful_types}/4 數據類型成功")
                
            except Exception as e:
                print(f"✗ 錯誤: {e}")
                all_results[ticker] = {'error': str(e)}
        
        print(f"\n{'='*80}")
        print(f"✅ 分割完成！")
        print(f"{'='*80}\n")
        
        return all_results
    
    def split_from_config(self, config_path: str = 'configs/defaults.yaml') -> Dict:
        """
        從配置文件中讀取日期和股票列表，自動進行分割
        
        :param config_path: 配置文件路徑
        :return: 分割結果字典
        """
        print(f"\n📋 從配置文件載入: {config_path}\n")
        
        # 載入配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        data_cfg = config['data']
        
        # 提取配置信息
        train_start = data_cfg['train_date_start']
        train_end = data_cfg['train_date_end']
        test_start = data_cfg['test_date_start']
        test_end = data_cfg['test_date_end']
        tickers = data_cfg['ticker_list']
        
        print(f"配置信息:")
        print(f"  訓練期間: {train_start} 至 {train_end}")
        print(f"  測試期間: {test_start} 至 {test_end}")
        print(f"  股票數量: {len(tickers)}\n")
        
        # 執行分割
        return self.split_all_tickers(
            tickers, train_start, train_end, test_start, test_end
        )
    
    def print_summary(self, results: Dict):
        """
        打印分割結果詳細摘要
        
        :param results: 分割結果字典
        """
        print(f"\n{'='*80}")
        print(f"📊 分割結果詳細摘要")
        print(f"{'='*80}\n")
        
        data_types_summary = {
            'stock': {}, 
            'indicators': {}, 
            'fundamentals': {},
            'fundamentals_daily': {}
        }
        
        for ticker, result in results.items():
            if 'error' in result:
                print(f"❌ {ticker:6s} - 錯誤: {result['error']}")
                continue
            
            print(f"✅ {ticker}")
            
            for data_type, info in result['data_types'].items():
                if info.get('status') == 'success':
                    train_rows = info['train']['rows']
                    test_rows = info['test']['rows']
                    
                    display_name = {
                        'stock': '股票數據    ',
                        'indicators': '技術指標    ',
                        'fundamentals': '基本面數據  ',
                        'fundamentals_daily': '對齊基本面  '
                    }.get(data_type, data_type)
                    
                    print(f"    {display_name} | 訓練: {train_rows:5d} 行 | 測試: {test_rows:5d} 行")
                    
                    # 累計統計
                    if data_type not in data_types_summary:
                        data_types_summary[data_type] = {}
                    if 'total_train' not in data_types_summary[data_type]:
                        data_types_summary[data_type]['total_train'] = 0
                        data_types_summary[data_type]['total_test'] = 0
                    
                    data_types_summary[data_type]['total_train'] += train_rows
                    data_types_summary[data_type]['total_test'] += test_rows
                else:
                    print(f"    {data_type:15s} | ❌ {info.get('message', '未知錯誤')}")
        
        # 打印總結
        print(f"\n{'-'*80}")
        print(f"📈 數據類型總結:")
        
        data_type_names = {
            'stock': '股票數據',
            'indicators': '技術指標',
            'fundamentals': '基本面數據',
            'fundamentals_daily': '對齊基本面'
        }
        
        for data_type in ['stock', 'indicators', 'fundamentals', 'fundamentals_daily']:
            if data_types_summary[data_type]:
                summary = data_types_summary[data_type]
                name = data_type_names.get(data_type, data_type)
                print(f"  {name:12s} | 訓練總行數: {summary['total_train']:8d} | 測試總行數: {summary['total_test']:8d}")
        
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # 從配置文件自動分割所有股票的四種數據
    splitter = ComprehensiveDataSplitter()
    results = splitter.split_from_config('configs/defaults.yaml')
    splitter.print_summary(results)