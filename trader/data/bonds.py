import yfinance as yf
import pandas as pd
import yaml

class BondDataFetcher:

    # 需要縮放（除以 10 才是真實殖利率 %）
    SCALE_MAP = {
        "^TNX": 0.1,
        "^TYX": 0.1,
    }

    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        # Use absolute path relative to this file's location
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        file_path = os.path.join(project_root, "configs", "defaults.yaml")
        
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {file_path}")
        except UnicodeDecodeError:
            print(f"UTF-8 decode failed for {file_path}, trying with encoding detection...")
            with open(file_path, "r", encoding="utf-8-sig") as file:
                config = yaml.safe_load(file)
        return config


    def fetch_treasury_history(self):
        start = self.config.get("data", {}).get("date_start")
        end = self.config.get("data", {}).get("date_end")
        features = self.config.get("data", {}).get("bond", {}).get("features")
        all_data = {}
         
        if features is None:
            raise ValueError("'bond.features' not found in config.")

        for label, ticker in features.items():
            print(f"Downloading: {ticker} ({label})")

            df = yf.download(ticker, start=start, end=end, progress=False)

            if df.empty:
                print(f"⚠ No data for {ticker}, skipping.")
                continue

            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Get closing price
            series = df["Close"].copy()
            
            # Scale if needed
            if ticker in self.SCALE_MAP:
                series = series * self.SCALE_MAP[ticker]
            
            series.name = label
            all_data[label] = series
            print(f"  Added {label}: {len(series)} rows")

        if not all_data:
            print("ERROR: No bond data downloaded!")
            return pd.DataFrame()
            
        # Concatenate using dict (preserves series names as columns)
        result = pd.concat(all_data, axis=1)
        
        # Reorder columns
        column_order = ["3M", "1Y", "2Y", "5Y", "10Y", "30Y"]
        result = result[[c for c in column_order if c in result.columns]]
        
        return result

    def save_to_csv(self, df: pd.DataFrame, file_path: str):
        if df.empty:
            print("ERROR: DataFrame is empty!")
            return
        
        print(f"Saving DataFrame with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
        df.to_csv(file_path)
        print(f"✓ Saved bond data to {file_path}")

if __name__ == "__main__":
    fetcher = BondDataFetcher()
    bond_data = fetcher.fetch_treasury_history()
    fetcher.save_to_csv(bond_data, f"data/bonds/bond_yields_{fetcher.config.get("data", {}).get("date_start")}_{fetcher.config.get("data", {}).get("date_end")}.csv")


