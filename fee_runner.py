import json
import os
import glob
import pandas as pd
import numpy as np
import time

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
 
def read_fee_config(config_file: str) -> dict:
    # read the config file
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, 'r') as f:
        return json.load(f)

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

def compute_volatility(volatility_gamma: float, second_volatility_gamma: float, prices: list) -> float:
    """
    Compute EWMA volatility of the price given gamma
    """

    prices = pd.Series(prices)
    # compute the returns
    returns = np.log(prices / prices.shift(1))

    # check docs on complex formula
    volatility =100000*np.sqrt(returns.pow(2).ewm(alpha=volatility_gamma,adjust=False).mean())
    volatility2 = np.sqrt(volatility.pow(2).ewm(alpha=second_volatility_gamma,adjust=False).mean())
    return volatility2.iloc[-1]

def vol_to_tvl_ratio(volume_24h: float, tvl: float) -> float:
    """
    Compute the ratio of volume to tvl
    """
    return volume_24h / tvl

def read_data(data_dir: str) -> dict:
    """
    Read the latest data values from the data directory and create a dictionary with the pool address as the key and the metrics as the values
    """
    data_dir = os.path.join(data_dir, "gliquid")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    data_files = glob.glob(f"{data_dir}/*.csv")
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    data_files = sorted(data_files, key=lambda p: int(os.path.basename(p).split('.')[0]))
    df = pd.read_csv(data_files[-1])
    return df.to_dict(orient='records')

def read_delta_price(data_dir: str) -> float:
    """
    Read the latest delta price value from the data directory
    """
    delta_price_dir = os.path.join(data_dir, "dex")
    if not os.path.exists(delta_price_dir):
        raise FileNotFoundError(f"Directory not found: {delta_price_dir}")

    delta_price_files = glob.glob(f"{delta_price_dir}/*.csv")
    if not delta_price_files:
        raise FileNotFoundError(f"No delta price files found in {delta_price_dir}")

    # Get latest file by timestamp
    latest_file = max(
        delta_price_files,
        key=lambda p: int(os.path.basename(p).split('.')[0])
    )

    # Read first line and convert to float
    with open(latest_file, 'r') as f:
        value = float(f.readline().strip())

    return value

def tvl_filter(tvls: list, tvl_window: int, tvl_sigma: float) -> float:
    """
    Apply Hampel filter to detect and correct outliers in the tvl
    """
    values = np.array(tvls[-tvl_window:])
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        scale = 0
    else:
        scale = 1.4826 * mad
    threshold = tvl_sigma * scale
    # If the current value deviates too much, replace it with the median of the window
    if abs(tvls[-1] - median) > threshold:
        return median
    else:
        return tvls[-1]

def compute_fee(volatility: float, vol_to_tvl_ratio: float, delta_delta_price: float, config: dict) -> float:
    """
    Compute the fee based on the volatility, volume to tvl ratio, and delta delta price
    """
    # Compute sigmoid terms for each component
    # y1 = np.log(1 + np.exp((df['volatility'] - config['fee_parameters']['volatility_params']['volatility_a2']) /
    #                    config['fee_parameters']['volatility_params']['volatility_a1']))
    
    y1 = (config["fee_parameters"]["volatility_params"]["volatility_b1"] 
    + config["fee_parameters"]["volatility_params"]["volatility_b5"]*volatility 
    + config["fee_parameters"]["volatility_params"]["volatility_b2"]/(1 + np.exp((volatility - config["fee_parameters"]["volatility_params"]["volatility_b3"])/config["fee_parameters"]["volatility_params"]["volatility_b4"])))

    y2 = np.log(1 + np.exp((vol_to_tvl_ratio - config['fee_parameters']['vol_tvlf_params']['vol_tvlf_b2']) /
                          config['fee_parameters']['vol_tvlf_params']['vol_tvlf_b1']))
    
    y3 = np.log(1 + np.exp((np.abs(delta_delta_price) - config['fee_parameters']['delta_delta_price_params']['delta_delta_price_a2']) /
                          config['fee_parameters']['delta_delta_price_params']['delta_delta_price_a1']))

    # Get fee bounds
    min_fee = config['fee_parameters']['min_fee']
    max_fee = config['fee_parameters']['max_fee']
    
    # Compute total fee and clip to bounds
    fee = np.clip(y1 + y2 + y3, min_fee, max_fee)
    
    # Return the fee
    return fee

if __name__ == "__main__":
    config = read_fee_config("./configs/fee_config.json")
    tvls=[]
    prices=[]
    while True:
        gliquid_pool_metrics = read_data(config["data_dir"])
        delta_price = read_delta_price(config["data_dir"])
        for pool_metric in gliquid_pool_metrics:
            pool_metric['vol_to_tvl_ratio'] = vol_to_tvl_ratio(pool_metric['volume24h'], pool_metric['tvl'])
            pool_metric['delta_delta_price'] = pool_metric['price'] - delta_price
            prices.append(pool_metric['price'])
            tvls.append(pool_metric['tvl'])
            if len(tvls) >= config["start_threshold"]*config["pull_interval"] and len(prices) >= config["start_threshold"]*config["pull_interval"]:
                pool_metric["filtered_tvl"] = tvl_filter(tvls, config["tvl_window"], config["tvl_sigma"])
                pool_metric['vol_to_tvl_ratio'] = vol_to_tvl_ratio(pool_metric['volume24h'], pool_metric['filtered_tvl'])
                pool_metric["volatility"] = compute_volatility(config["first_volatility_gamma"], config["second_volatility_gamma"], prices)
                pool_metric["fee"] = compute_fee(pool_metric["volatility"], pool_metric['vol_to_tvl_ratio'], pool_metric['delta_delta_price'], config)
        print(gliquid_pool_metrics)
        time.sleep(config["pull_interval"])
 
