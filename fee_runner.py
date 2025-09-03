from datetime import datetime
import json
import os
import glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def calculate_price_from_sqrt_price(sqrt_price_str):
    """
    Calculate price from sqrt_price (stored as string).
    Price = (sqrt_price / 2^96)^2
    """
    sqrt_price = int(sqrt_price_str)
    # sqrt_price is stored as Q64.96 fixed point
    # Convert to actual price
    price = (sqrt_price / (2**96)) ** 2
    return price

def read_pool_snapshots(pool_address,snapshot_dir):
    """
    Read all snapshot files for a given pool address and extract key metrics.
    
    Args:
        pool_address (str): The pool address to analyze
        
    Returns:
        pd.DataFrame: DataFrame with timestamp, volume24h, tvl, and price
    """
    # Path to the snapshots directory
    snapshots_dir = f"{snapshot_dir}/{pool_address}/snapshots"
    
    if not os.path.exists(snapshots_dir):
        raise FileNotFoundError(f"Snapshots directory not found: {snapshots_dir}")
    
    # Get all JSON files in the snapshots directory
    snapshot_files = glob.glob(f"{snapshots_dir}/*.json")
    
    if not snapshot_files:
        raise FileNotFoundError(f"No snapshot files found in {snapshots_dir}")
    
    print(f"Found {len(snapshot_files)} snapshot files")
    
    data = []
    
    for file_path in sorted(snapshot_files):
        try:
            with open(file_path, 'r') as f:
                snapshot = json.load(f)
            
            # Extract timestamp
            timestamp_str = snapshot.get('snapped_at')
            if timestamp_str:
                # Parse the ISO timestamp
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # Fallback to filename as timestamp if no snapped_at field
                filename = os.path.basename(file_path)
                block_num = filename.replace('.json', '')
                timestamp = datetime.now()  # Placeholder, could be improved
            
            # Extract metrics
            volume_24h = snapshot.get('volume24h_usd', 0)
            tvl = snapshot.get('tvl_usd', 0)
            sqrt_price = snapshot.get('sqrt_price', '0')
            
            # Calculate price from sqrt_price
            price = calculate_price_from_sqrt_price(sqrt_price) * 10e11
            
            data.append({
                'timestamp': timestamp,
                'volume_24h_usd': volume_24h,
                'tvl_usd': tvl,
                'price': price,
                'block_number': snapshot.get('block_number', 0),
                'current_tick': snapshot.get('current_tick', 0)
            })
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    # Create DataFrame and sort by timestamp
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

 
def read_fee_config(config_file: str) -> dict:
    # read the config file
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, 'r') as f:
        return json.load(f)

def tvl_filter(df: pd.DataFrame, tvl_window: int, tvl_sigma: float) -> pd.Series:
    """
    Filter the df based on the tvl with a hampel filter
    """
    tvl = df['tvl_usd']
    rolling_median = tvl.rolling(window=tvl_window, center=True, min_periods=1).median()
    diff = (tvl - rolling_median).abs()
    mad = diff.rolling(window=tvl_window, center=True, min_periods=1).median()
    threshold = tvl_sigma * 1.4826 * mad
    mask_outlier = diff > threshold
    filtered = tvl.copy()
    filtered[mask_outlier] = rolling_median[mask_outlier]
    return filtered

def compute_volatility(volatility_gamma: float, df: pd.DataFrame) -> pd.Series:
    """
    Compute EWMA volatility of the price given gamma
    """
    price = df['price']

    # compute the returns
    returns = np.log(price / price.shift(1))

    # compute volatility and multiply by 10000 for easier visualization
    # check docs on complex formula
    return 10000*np.sqrt(returns.pow(2).ewm(alpha=1-volatility_gamma,adjust=False).mean())
    
def vol_to_tvl_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Compute the ratio of volume to tvl
    """
    return df['volume_24h_usd'] / df['tvl_filtered']

# def check_lag(df: pd.DataFrame) -> float:
#     volatility = df['volatility']
#     price = df['price']
#     returns = (np.log(price / price.shift(1)))*10000
#     lags = range(0,1000)
#     correlations = []
    
#     for lag in lags:
#         if lag < 0:
#             corr = returns.corr(volatility.shift(-lag))  # returns forward 
#         else:
#             corr = returns.corr(volatility.shift(lag))   # returns behind
#         correlations.append(corr)

#     ccf_df = pd.DataFrame({'lag': lags, 'correlation': correlations})

#     # Find the lag with maximum correlation
#     best_lag = ccf_df.loc[ccf_df['correlation'].idxmax()]
#     print(f"best lag: {best_lag['lag']} days, correlation: {best_lag['correlation']:.4f}")
#     return list(lags), correlations

if __name__ == "__main__":
    config = read_fee_config("./configs/fee_config.json")
    df = read_pool_snapshots(config["pool_address"],config["snapshot_dir"])
    df['volatility'] = compute_volatility(config["volatility_gamma"], df)
    df['tvl_filtered'] = tvl_filter(df, config["tvl_window"], config["tvl_sigma"])
    df['vol_to_tvl_ratio'] = vol_to_tvl_ratio(df)
    # lags, correlations = check_lag(df)

    # debug print
    print(df.head())
    print(config)
    print(df.columns)

    # plot price, tvl and volume as subplots
    fig = make_subplots(rows=5, cols=1, subplot_titles=("Price", "TVL", "Volume", "Volatility", "Volume to TVL (Filtered) Ratio"))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tvl_usd'], name='TVL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['volume_24h_usd'], name='Volume'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['volatility'], name='Volatility'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vol_to_tvl_ratio'], name='Volume to TVL (Filtered) Ratio'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tvl_filtered'], name='TVL Filtered'), row=2, col=1)
    fig.show()

