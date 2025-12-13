#!/usr/bin/env python3
"""檢查基本面數據中的 NaN 值"""

import pandas as pd
import numpy as np
import os
import yaml

def check_nan_in_fundamentals():
    # 載入配置
    with open('./configs/defaults.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    tickers = config['data']['ticker_list']
    data_dir = './data/fundamentals/daily_aligned'
    
    print(f"{'='*70}")
    print(f"檢查基本面數據 NaN 值")
    print(f"{'='*70}\n")
    
    total_files = 0
    files_with_nan = 0
    total_nan = 0
    
    nan_summary = []
    
    for ticker in tickers:
        possible_files = [
            f"{data_dir}/{ticker}_fundamentals_daily.csv",
            f"{data_dir}/{ticker}_fundamentals.csv",
            f"{data_dir}/{ticker}.csv",
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                total_files += 1
                df = pd.read_csv(file_path)
                
                nan_count = df.isna().sum().sum()
                inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
                
                if nan_count > 0 or inf_count > 0:
                    files_with_nan += 1
                    total_nan += nan_count
                    
                    # 找出有 NaN 的欄位
                    nan_columns = df.isna().sum()
                    nan_cols = nan_columns[nan_columns > 0]
                    
                    nan_summary.append({
                        'ticker': ticker,
                        'file': file_path,
                        'nan_count': nan_count,
                        'inf_count': inf_count,
                        'rows': len(df),
                        'nan_columns': dict(nan_cols)
                    })
                    
                    print(f"❌ {ticker}: {nan_count} NaN, {inf_count} Inf")
                    print(f"   檔案: {file_path}")
                    print(f"   總行數: {len(df)}")
                    print(f"   有 NaN 的欄位:")
                    for col, count in nan_cols.items():
                        pct = count / len(df) * 100
                        print(f"     - {col}: {count} ({pct:.1f}%)")
                    print()
                else:
                    print(f"✅ {ticker}: 無 NaN")
                
                break  # 找到檔案就跳出
        else:
            print(f"⚠️  {ticker}: 檔案不存在")
    
    print(f"\n{'='*70}")
    print(f"總結")
    print(f"{'='*70}")
    print(f"總檔案數: {total_files}")
    print(f"有 NaN 的檔案數: {files_with_nan}")
    print(f"總 NaN 數量: {total_nan}")
    
    if nan_summary:
        print(f"\n按 NaN 數量排序（最多的前 5 個）:")
        sorted_summary = sorted(nan_summary, key=lambda x: x['nan_count'], reverse=True)
        for item in sorted_summary[:5]:
            print(f"  {item['ticker']}: {item['nan_count']} NaN")

if __name__ == '__main__':
    check_nan_in_fundamentals()