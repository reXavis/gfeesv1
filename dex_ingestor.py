import os
import json
from requests import get
import pandas as pd
import time

def read_ingestor_config(config_file: str) -> dict:
    """Read and parse the ingestor configuration file.
    
    Args:
        config_file (str): Path to the config JSON file
        
    Returns:
        dict: Configuration parameters
        
    Raises:
        FileNotFoundError: If config file does not exist
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, 'r') as f:
        return json.load(f)

def get_pool_info(config: dict) -> dict:
    """Fetch pool information from the API.
    
    Args:
        config (dict): Configuration containing API URL and token addresses
        
    Returns:
        dict: Pool information response from API
    """
    url = f"{config['api_url']}/pools?tokenA={config['token0']}&tokenB={config['token1']}"
    response = get(url)
    return response.json()

def pool_info_to_csv(pool_info: dict, output_file: str):
    """Convert pool information to CSV format and save to file.
    
    Args:
        pool_info (dict): Pool information from API
        output_file (str): Path to save CSV file
    """
    pool_info_data = pool_info.get("data", [])
    df = pd.DataFrame(pool_info_data)
    df = df[["poolAddress", "fee", "protocol", "price", "pair"]]
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Load config and create output directory
    config = read_ingestor_config("ingestor_config.json")
    os.makedirs("data/liquidlabs", exist_ok=True)
    
    # Continuously poll API and save results
    counter = 0
    while True:
        pool_info = get_pool_info(config)
        pool_info_to_csv(pool_info, f"data/liquidlabs/pool_info_{counter}.csv")
        print(f"Pulled pool info {counter} times")
        time.sleep(config["pull_interval"])
        counter += 1