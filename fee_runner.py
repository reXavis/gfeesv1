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

def compute_volatility(volatility_gamma: float, second_volatility_gamma: float, df: pd.DataFrame) -> pd.Series:
    """
    Compute EWMA volatility of the price given gamma
    """
    price = df['price']

    # compute the returns
    returns = np.log(price / price.shift(1))

    # check docs on complex formuzla
    volatility =100000*np.sqrt(returns.pow(2).ewm(alpha=volatility_gamma,adjust=False).mean())
    volatility2 = np.sqrt(volatility.pow(2).ewm(alpha=second_volatility_gamma,adjust=False).mean())
    return volatility2

def vol_to_tvl_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Compute the ratio of volume to tvl
    """
    return df['volume_24h_usd'] / df['tvl_filtered']

def read_data(data_dir: str) -> pd.DataFrame:
    """
    Read the data from the data directory and create separate dataframes for each pool
    """
    # Get all CSV files from gliquid directory
    gliquid_dir = os.path.join(data_dir, "gliquid")
    if not os.path.exists(gliquid_dir):
        raise FileNotFoundError(f"Directory not found: {gliquid_dir}")
        
    # Dictionary to store data for each pool
    pool_data = {}
    
    for filename in os.listdir(gliquid_dir):
        if filename.endswith('.csv'):
            # Extract timestamp from filename
            timestamp = int(filename.split('.')[0])
            
            # Read the CSV file
            file_path = os.path.join(gliquid_dir, filename)
            df_temp = pd.read_csv(file_path)
            
            # Process each pool in the CSV
            for _, row in df_temp.iterrows():
                pool_address = row['pool_address']
                
                if pool_address not in pool_data:
                    pool_data[pool_address] = []
                    
                pool_data[pool_address].append({
                    'timestamp': timestamp,
                    'tvl_usd': row['tvl'],
                    'volume_24h_usd': row['volume24h'],
                    'price': row['price']
                })
    
    # Create separate DataFrames for each pool and store in dictionary
    pool_dfs = {}
    for pool_address, data in pool_data.items():
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        pool_dfs[pool_address] = df

    return pool_dfs

def read_delta_price(data_dir: str) -> pd.Series:
    """
    Read the delta price from the data directory
    """
    delta_price_dir = os.path.join(data_dir, "dex")
    if not os.path.exists(delta_price_dir):
        raise FileNotFoundError(f"Directory not found: {delta_price_dir}")

    delta_price_files = glob.glob(f"{delta_price_dir}/*.csv")
    if not delta_price_files:
        raise FileNotFoundError(f"No delta price files found in {delta_price_dir}")

    # Ensure files are processed in timestamp order
    delta_price_files = sorted(
        delta_price_files,
        key=lambda p: int(os.path.basename(p).split('.')[0])
    )

    values_by_timestamp = {}
    for file in delta_price_files:
        timestamp = int(os.path.basename(file).split('.')[0])
        with open(file, 'r') as f:
            first_line = f.readline().strip()
        value = float(first_line)
        values_by_timestamp[timestamp] = value

    delta_price_series = pd.Series(values_by_timestamp).sort_index()
    return delta_price_series
 
def compute_delta_delta_price(df: pd.DataFrame, delta_price_series: pd.Series) -> pd.Series:
    
    """
    Compute the delta delta price by aligning timestamps
    """
    # Create a series indexed by timestamp from the dataframe (ensure int index)
    price_by_ts = pd.Series(df['price'].values, index=df['timestamp'].astype(int))

    # Align delta_price_series to the dataframe timestamps using nearest neighbor
    aligned_delta = delta_price_series.sort_index().reindex(price_by_ts.index, method='nearest')

    # Return values aligned to the dataframe's index to avoid assignment NaNs
    delta_delta_values = price_by_ts.values - aligned_delta.values
    return pd.Series(delta_delta_values, index=df.index)

def smooth_delta_delta_price(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Smooth the delta delta price using a rolling window
    """
    return df['delta_delta_price'].ewm(alpha=1-config["delta_delta_price_gamma"],adjust=False).mean()

def read_ramses_calibration(csv_path: str) -> pd.DataFrame:
    """
    Read and clean Ramses calibration CSV with columns: timestamp, volume_24h, fees_24h
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Coerce to numeric and drop rows with missing essentials
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['volume_24h'] = pd.to_numeric(df['volume_24h'], errors='coerce')
    df['fees_24h'] = pd.to_numeric(df['fees_24h'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'volume_24h', 'fees_24h']).copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    return df

def test_ramses_correlation(ramses_df: pd.DataFrame) -> None:
    """
    Test the correlation between Ramses volume and fees
    """
    return np.corrcoef(ramses_df['volume_24h'], ramses_df['fees_24h'])[0, 1]

def compute_fee(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute the fee based on the volatility, volume to tvl ratio, and delta delta price
    """
    # Compute sigmoid terms for each component
    y1 = np.log(1 + np.exp((df['volatility'] - config['fee_parameters']['volatility_params']['volatility_a2']) /
                        config['fee_parameters']['volatility_params']['volatility_a1']))
    
    y2 = np.log(1 + np.exp((df['vol_to_tvl_ratio'] - config['fee_parameters']['vol_tvlf_params']['vol_tvlf_b2']) /
                          config['fee_parameters']['vol_tvlf_params']['vol_tvlf_b1']))
    
    y3 = np.log(1 + np.exp((np.abs(df['delta_delta_price']) - config['fee_parameters']['delta_delta_price_params']['delta_delta_price_a2']) /
                          config['fee_parameters']['delta_delta_price_params']['delta_delta_price_a1']))

    # Get fee bounds
    min_fee = config['fee_parameters']['min_fee']
    max_fee = config['fee_parameters']['max_fee']
    
    # Compute total fee and clip to bounds
    fee = np.clip(y1 + y2 + y3, min_fee, max_fee)
    
    # Return as pandas Series to maintain index alignment
    return pd.DataFrame({'fee': fee, 'fee_volatility': y1, 'fee_vol_to_tvl_ratio': y2, 'fee_delta_delta_price': y3}, index=df.index)

def build_distribution_plots(df: pd.DataFrame) -> None:
    """
    Build the kernel density distribution plots for the metrics
    """
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Volatility", "Volume to TVL (Filtered) Ratio", "Delta Delta Price"))
    fig.add_trace(go.Histogram(x=df['volatility'], name='Volatility', xbins=dict(size=1.25)), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['vol_to_tvl_ratio'], name='Volume to TVL (Filtered) Ratio', xbins=dict(size=0.005)), row=2, col=1)
    fig.add_trace(go.Histogram(x=df['delta_delta_price'], name='Delta Delta Price', xbins=dict(size=0.005)), row=3, col=1)

    fig.show()

def build_price_plots(df: pd.DataFrame, ts: pd.Series) -> None:
    """
    Build the price plots for the metrics across time
    """
    fig = make_subplots(rows=4, cols=1, subplot_titles=("Price", "Volatility", "Volume to TVL (Filtered) Ratio", "Delta Delta Price", "Delta Delta Price Smooth"))
    fig.add_trace(go.Scatter(x=ts, y=df['price'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=df['volatility'], name='Volatility'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=df["vol_to_tvl_ratio"], name='Volume to TVL (Filtered) Ratio'), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts, y=df['delta_delta_price'], name='Delta Delta Price'), row=4, col=1)
    fig.add_trace(go.Scatter(x=ts, y=df['delta_delta_price_smooth'], name='Delta Delta Price Smooth'), row=4, col=1)
    fig.show()

def build_fee_plots(df: pd.DataFrame, ts: pd.Series) -> None:  
    """
    Build the fee plots for the metrics
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=df['fee'], name='Fee'))
    fig.add_trace(go.Scatter(x=ts, y=df['fee_volatility'], name='Volatility Fee'))
    fig.add_trace(go.Scatter(x=ts, y=df['fee_vol_to_tvl_ratio'], name='Volume to TVL (Filtered) Ratio Fee'))
    fig.add_trace(go.Scatter(x=ts, y=df['fee_delta_delta_price'], name='Delta Delta Price Fee'))
    fig.show()

def build_fee_component_plots(ramses_df: pd.DataFrame, df: pd.DataFrame, ts: pd.Series) -> None:
    """
    Build the fee component plots as stacked area plots
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=df['fee_volatility'], name='Volatility Fee', mode='none', stackgroup='one', fillcolor='rgba(99,110,250,0.5)'))
    fig.add_trace(go.Scatter(x=ts, y=df['fee_vol_to_tvl_ratio'], name='Volume to TVL (Filtered) Ratio Fee', mode='none', stackgroup='one', fillcolor='rgba(239,85,59,0.5)'))
    fig.add_trace(go.Scatter(x=ts, y=df['fee_delta_delta_price'], name='Delta Delta Price Fee', mode='none', stackgroup='one', fillcolor='rgba(0,204,150,0.5)'))
    fig.add_trace(go.Scatter(x=ts, y=df['fee'], name='Total Fee', mode='lines', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=ts, y=ramses_df['fees_24h'], name='Ramses Fees', mode='lines', line=dict(width=2, color='black')))
    fig.update_layout(title="Fee Component Breakdown Over Time")
    fig.show()


if __name__ == "__main__":
    config = read_fee_config("./configs/fee_config.json")
    ramses_df = read_ramses_calibration(os.path.join(config["data_dir"], "calibration", "ramses.csv"))
    pool_dfs = read_data(config["data_dir"])
    delta_price_series = read_delta_price(config["data_dir"])
    for pool_address, df in pool_dfs.items():
        ts = pd.to_datetime(df['timestamp'], unit='s')
        df['volatility'] = compute_volatility(config["first_volatility_gamma"], config["second_volatility_gamma"], df)
        df['tvl_filtered'] = tvl_filter(df, config["tvl_window"], config["tvl_sigma"])
        df['vol_to_tvl_ratio'] = vol_to_tvl_ratio(df)
        df['delta_delta_price'] = compute_delta_delta_price(df, delta_price_series)
        df['delta_delta_price_smooth'] = smooth_delta_delta_price(df, config)
        build_distribution_plots(df)
        build_price_plots(df, ts)
        fee_df = compute_fee(df, config)
        print(fee_df.tail())
        build_fee_plots(fee_df, ts)
        build_fee_component_plots(ramses_df, fee_df, ts)
