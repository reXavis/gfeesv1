import os
import json
from requests import get
import pandas as pd

def read_ingestor_config(config_file: str) -> dict:
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, 'r') as f:
        return json.load(f)

def get_pool_info(config: dict) -> dict:
    url = f"{config['api_url']}/pools?tokenA={config['token0']}&tokenB={config['token1']}"
    response = get(url)
    return response.json()

def pool_info_to_csv(pool_info: dict, output_file: str):
    pool_info_data = pool_info.get("data", [])
    df = pd.DataFrame(pool_info_data)
    # only keep poolAddress, fee, protocol, price, pair
    df = df[["poolAddress", "fee", "protocol", "price", "pair"]]
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    config = read_ingestor_config("ingestor_config.json")
    pool_info = get_pool_info(config)
    pool_info_to_csv(pool_info, "pool_info.csv")