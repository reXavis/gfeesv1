from datetime import datetime
import json
import os
import glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def read_fee_config(config_file: str) -> dict:
    # read the config file
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, 'r') as f:
        return json.load(f)

def metric_filter(metric: pd.Series, metric_window: int, metric_sigma: float) -> pd.Series:
    """
    Filter a metric with a hampel filter
    """
    rolling_median = metric.rolling(window=metric_window, center=True, min_periods=1).median()
    diff = (metric - rolling_median).abs()
    mad = diff.rolling(window=metric_window, center=True, min_periods=1).median()
    threshold = metric_sigma * 1.4826 * mad
    mask_outlier = diff > threshold
    filtered = metric.copy()
    filtered[mask_outlier] = rolling_median[mask_outlier]
    return filtered

def compute_volatility(volatility_gamma: float, second_volatility_gamma: float, price: pd.Series) -> pd.Series:
    """
    Compute EWMA volatility of the price given gamma
    """
    # compute the returns
    returns = np.log(price / price.shift(1))

    # check docs on complex formuzla
    volatility =100000*np.sqrt(returns.pow(2).ewm(alpha=volatility_gamma,adjust=False).mean())
    volatility2 = np.sqrt(volatility.pow(2).ewm(alpha=second_volatility_gamma,adjust=False).mean())
    return volatility2

def vol_to_tvl_ratio(volume: pd.Series, tvl: pd.Series) -> pd.Series:
    """
    Compute the ratio of volume to tvl
    """
    return volume / tvl

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
    price_by_ts = pd.Series(df['price_filtered'].values, index=df['timestamp'].astype(int))

    # Align delta_price_series to the dataframe timestamps using nearest neighbor
    aligned_delta = delta_price_series.sort_index().reindex(price_by_ts.index, method='nearest')

    # Return values aligned to the dataframe's index to avoid assignment NaNs
    delta_delta_values = price_by_ts.values - aligned_delta.values
    return pd.Series(delta_delta_values, index=df.index)

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
    return df

def compute_fee(volatility: pd.Series, vol_to_tvl_ratio: pd.Series, delta_delta_price: pd.Series, ts: pd.Series, config: dict) -> pd.DataFrame:
    """
    Compute the fee based on the volatility, volume to tvl ratio, and delta delta price.
    Only updates fee every proposal interval.
    """
    # Create empty DataFrames for storing fee components
    fee_df = pd.DataFrame(index=ts.index)
    
    # Get proposal interval from config
    proposal_interval = config["proposal_interval"]
    
    # Get first timestamp
    start_ts = ts.iloc[0]
    
    # Calculate fee components only at proposal intervals
    for i in range(0, len(ts), proposal_interval):
        # Get data for this interval
        interval_ts = ts.iloc[i:i+proposal_interval]
        
        # Calculate components for this interval
        y1 = (config["fee_parameters"]["volatility_params"]["volatility_b1"] 
        + config["fee_parameters"]["volatility_params"]["volatility_b5"]*volatility.iloc[i:i+proposal_interval].iloc[-1]
        + config["fee_parameters"]["volatility_params"]["volatility_b2"]/(1 + np.exp((volatility.iloc[i:i+proposal_interval].iloc[-1] - config["fee_parameters"]["volatility_params"]["volatility_b3"])/config["fee_parameters"]["volatility_params"]["volatility_b4"])))

        y2 = np.log(1 + np.exp((vol_to_tvl_ratio.iloc[i:i+proposal_interval].iloc[-1] - config['fee_parameters']['vol_tvlf_params']['vol_tvlf_b2']) /
                              config['fee_parameters']['vol_tvlf_params']['vol_tvlf_b1']))
        
        y3 = np.log(1 + np.exp((np.abs(delta_delta_price.iloc[i:i+proposal_interval].iloc[-1]) - config['fee_parameters']['delta_delta_price_params']['delta_delta_price_a2']) /
                              config['fee_parameters']['delta_delta_price_params']['delta_delta_price_a1']))

        # Get fee bounds
        min_fee = config['fee_parameters']['min_fee']
        max_fee = config['fee_parameters']['max_fee']
        
        # Compute total fee and clip to bounds
        fee = np.clip(y1 + y2 + y3, min_fee, max_fee)
        
        # Assign fee components to all rows in this interval
        fee_df.loc[interval_ts.index, 'fee'] = fee
        fee_df.loc[interval_ts.index, 'fee_volatility'] = y1
        fee_df.loc[interval_ts.index, 'fee_vol_to_tvl_ratio'] = y2  
        fee_df.loc[interval_ts.index, 'fee_delta_delta_price'] = y3
        fee_df.loc[interval_ts.index, 'timestamp'] = interval_ts

    return fee_df


# Plotting functions

def build_distribution_plots(volatility: pd.Series, vol_to_tvl_ratio: pd.Series, delta_delta_price: pd.Series) -> None:
    """
    Build the kernel density distribution plots for the metrics
    """
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Volatility", "Volume to TVL (Filtered) Ratio", "Delta Delta Price"))
    fig.add_trace(go.Histogram(x=volatility, name='Volatility', xbins=dict(size=1.25)), row=1, col=1)
    fig.add_trace(go.Histogram(x=vol_to_tvl_ratio, name='Volume to TVL (Filtered) Ratio', xbins=dict(size=0.005)), row=2, col=1)
    fig.add_trace(go.Histogram(x=delta_delta_price, name='Delta Delta Price', xbins=dict(size=0.005)), row=3, col=1)

    fig.show()

def build_price_plots(price: pd.Series, volatility: pd.Series, vol_to_tvl_ratio: pd.Series, delta_delta_price: pd.Series, ts: pd.Series) -> None:
    """
    Build the price plots for the metrics across time
    """
    fig = make_subplots(rows=4, cols=1, subplot_titles=("Price", "Volatility", "Volume to TVL (Filtered) Ratio", "Delta Delta Price", "Delta Delta Price Smooth"))
    fig.add_trace(go.Scatter(x=ts, y=price, name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=volatility, name='Volatility'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=vol_to_tvl_ratio, name='Volume to TVL (Filtered) Ratio'), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts, y=delta_delta_price, name='Delta Delta Price'), row=4, col=1)
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

def build_fee_component_plots(ramses_df: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Build the fee component plots as stacked area plots
    """
    # Convert timestamps to datetime for alignment
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['fee_volatility'], name='Volatility Fee', mode='none', stackgroup='one', fillcolor='rgba(99,110,250,0.5)'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['fee_vol_to_tvl_ratio'], name='Volume to TVL (Filtered) Ratio Fee', mode='none', stackgroup='one', fillcolor='rgba(239,85,59,0.5)'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['fee_delta_delta_price'], name='Delta Delta Price Fee', mode='none', stackgroup='one', fillcolor='rgba(0,204,150,0.5)'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['fee'], name='Total Fee', mode='lines', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=ramses_df['timestamp'], y=ramses_df['fees_24h'], name='Ramses Fees', mode='lines', line=dict(width=2, color='black')))
    fig.update_layout(title="Fee Component Breakdown Over Time")
    fig.show()

# Main function

if __name__ == "__main__":
    # Load config
    config = read_fee_config("./configs/fee_config.json")
    # Read Ramses, Gliquid and Dex Data
    ramses_df = read_ramses_calibration(os.path.join(config["data_dir"], "calibration", "ramses.csv"))
    pool_dfs = read_data(config["data_dir"])
    delta_price_series = read_delta_price(config["data_dir"])
    # For each pool we:
    for pool_address, df in pool_dfs.items():
        # Compute metrics
        df['volatility'] = compute_volatility(config["volatility_parameters"]["first_volatility_gamma"], config["volatility_parameters"]["second_volatility_gamma"], df['price'])
        df['tvl_filtered'] = metric_filter(df['tvl_usd'], config["tvl_filter_parameters"]["tvl_window"], config["tvl_filter_parameters"]["tvl_sigma"])
        df['price_filtered'] = metric_filter(df['price'], config["price_filter_parameters"]["price_window"], config["price_filter_parameters"]["price_sigma"])
        df['vol_to_tvl_ratio'] = vol_to_tvl_ratio(df['volume_24h_usd'], df['tvl_filtered'])
        df['delta_delta_price'] = compute_delta_delta_price(df, delta_price_series)
        # Build plots
        build_distribution_plots(df['volatility'], df['vol_to_tvl_ratio'], df['delta_delta_price'])
        build_price_plots(df['price_filtered'], df['volatility'], df['vol_to_tvl_ratio'], df['delta_delta_price'], df['timestamp'])
        # Compute fee
        fee_df = compute_fee(df['volatility'], df['vol_to_tvl_ratio'], df['delta_delta_price'], df['timestamp'], config)
        # Build fee plots
        build_fee_plots(fee_df, fee_df['timestamp'])
        build_fee_component_plots(ramses_df, fee_df)
