import os
import json
import pandas as pd
from requests import post
import time

def fetch_pool_addresses(token0: str, token1: str, graph_url: str) -> list:
    """Fetch all pool addresses for a token pair from the graph.
    
    Args:
        token0 (str): Token0 address
        token1 (str): Token1 address
        graph_url (str): Graph URL
        
    Returns:
        list: List of pool addresses found, or empty list if no pools found
        
    Raises:
        Exception: If there's an error fetching from the subgraph
    """
    try:
        # Remove /query suffix if present and add it back
        if graph_url.endswith('/query'):
            url = graph_url
        else:
            url = f"{graph_url}/query" if not graph_url.endswith('/gn') else graph_url
            
        # Query for pools with the specified token pair
        # Note: tokens are objects with id field in the subgraph schema
        query = f"""
        {{
            pools(where: {{
                token0: "{token0.lower()}",
                token1: "{token1.lower()}"
            }}) {{
                id
                token0 {{
                    id
                }}
                token1 {{
                    id
                }}
            }}
        }}
        """
        
        response = post(url, json={"query": query})
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Check for GraphQL errors
        if "errors" in data:
            raise Exception(f"GraphQL errors: {data['errors']}")
            
        # Check if we have pools data
        if "data" not in data or "pools" not in data["data"]:
            raise Exception(f"Unexpected response format: {data}")
            
        pools = data["data"]["pools"]
        
        if not pools:
            return []
            
        # Return all pool addresses found
        pool_addresses = [pool["id"] for pool in pools]
        return pool_addresses
        
    except Exception as e:
        raise

def fetch_pool_volume24h(pool_address: str, graph_url: str) -> float:
    """Fetch 24-hour volume for a pool from the subgraph using hourly data.
     
    Args:
        pool_address (str): Pool address to fetch volume for
        graph_url (str): Graph URL
        
    Returns:
        float: 24-hour volume in USD, or 0.0 if not available or error
    """
    try:
        # Remove /query suffix if present and add it back
        if graph_url.endswith('/query'):
            url = graph_url
        else:
            url = f"{graph_url}/query" if not graph_url.endswith('/gn') else graph_url
            
        # Query for pool hourly data - mirror the reference implementation
        query = (
            "query($pool:String!){\n"
            "  poolHourDatas(first:1000, orderBy:periodStartUnix, orderDirection:desc, where:{ pool:$pool }){\n"
            "    periodStartUnix\n"
            "    volumeUSD\n"
            "  }\n"
            "}"
        )
        
        response = post(url, json={"query": query, "variables": {"pool": pool_address}})
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Check for GraphQL errors
        if "errors" in data:
            # Some subgraphs might not have poolHourDatas field, return 0
            if any("no field `poolHourDatas`" in str(error) for error in data["errors"]):
                return 0.0
            raise Exception(f"GraphQL errors: {data['errors']}")
            
        # Check if we have poolHourDatas data
        if "data" not in data or "poolHourDatas" not in data["data"]:
            raise Exception(f"Unexpected response format: {data}")
            
        items = data["data"]["poolHourDatas"]
        
        if not items:
            return 0.0
            
        # Calculate 24-hour volume from hourly data - mirror the reference implementation
        try:
            latest_hour = int(items[0].get("periodStartUnix") or 0)
        except Exception:
            latest_hour = 0
            
        if latest_hour <= 0:
            return 0.0
            
        cutoff = latest_hour - 24 * 3600
        total = 0.0
        
        for it in items:
            try:
                ts = int(it.get("periodStartUnix") or 0)
            except Exception:
                continue
            if ts < cutoff:
                break
            try:
                v = float(it.get("volumeUSD") or 0.0)
            except Exception:
                v = 0.0
            total += v
        return float(total)
        
    except Exception as e:
        return 0.0

def fetch_pool_price(pool_address: str, graph_url: str) -> float:
    """Fetch price for a pool from the subgraph.
    
    Args:
        pool_address (str): Pool address to fetch price for
        graph_url (str): Graph URL
        
    Returns:
        float: Token0 price in USD, or 0.0 if not available or error
    """
    try:
        # Remove /query suffix if present and add it back
        if graph_url.endswith('/query'):
            url = graph_url
        else:
            url = f"{graph_url}/query" if not graph_url.endswith('/gn') else graph_url
            
        # Query for pool price - only get sqrtPrice
        query = (
            "query($pool:String!){\n"
            "  pools(where: { id: $pool }){\n"
            "    id\n"
            "    sqrtPrice\n"
            "  }\n"
            "}"
        )
        
        response = post(url, json={"query": query, "variables": {"pool": pool_address}})
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Check for GraphQL errors
        if "errors" in data:
            raise Exception(f"GraphQL errors: {data['errors']}")
            
        # Check if we have pools data
        if "data" not in data or "pools" not in data["data"]:
            raise Exception(f"Unexpected response format: {data}")
            
        pools = data["data"]["pools"]
        
        if not pools:
            print(f"Pool {pool_address} not found for price query")
            return 0.0
            
        # Get price from the first pool found
        pool = pools[0]
        
        # Calculate price from sqrtPrice
        sqrt_price = pool.get("sqrtPrice", "0")
        if sqrt_price and float(sqrt_price) > 0:
            return (float(sqrt_price) / (2**96)) ** 2
            
        return 0.0
        
    except Exception as e:
        print(f"Error fetching price for pool {pool_address}: {str(e)}")
        return 0.0

def fetch_pool_tvl(pool_address: str, graph_url: str) -> float:
    """Fetch TVL for a pool from the subgraph.
    
    Args:
        pool_address (str): Pool address to fetch TVL for
        graph_url (str): Graph URL
        
    Returns:
        float: TVL in USD, or 0.0 if not available or error
    """
    try:
        # Remove /query suffix if present and add it back
        if graph_url.endswith('/query'):
            url = graph_url
        else:
            url = f"{graph_url}/query" if not graph_url.endswith('/gn') else graph_url
            
        # Query for pool TVL using pool address
        query = (
            "query($pool:String!){\n"
            "  pools(where: { id: $pool }){\n"
            "    id\n"
            "    totalValueLockedUSD\n"
            "  }\n"
            "}"
        )
        
        response = post(url, json={"query": query, "variables": {"pool": pool_address}})
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Check for GraphQL errors
        if "errors" in data:
            raise Exception(f"GraphQL errors: {data['errors']}")
            
        # Check if we have pools data
        if "data" not in data or "pools" not in data["data"]:
            raise Exception(f"Unexpected response format: {data}")
            
        pools = data["data"]["pools"]
        
        if not pools:
            print(f"Pool {pool_address} not found for TVL query")
            return 0.0
            
        # Get TVL from the first pool found
        pool = pools[0]
        val = pool.get("totalValueLockedUSD")
        return float(val) if val is not None else 0.0
        
    except Exception as e:
        print(f"Error fetching TVL for pool {pool_address}: {str(e)}")
        return 0.0

def read_gliquid_config(config_file: str) -> dict:
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    config = read_gliquid_config("./configs/gliquid_config.json")

    while True:
        pool_addresses = fetch_pool_addresses(config["token0"], config["token1"], config["subgraph_url"])
        volume24h = []
        price = []
        tvl = []

        for pool_address in pool_addresses:
            volume24h.append(fetch_pool_volume24h(pool_address, config["subgraph_url"]))
            price.append(fetch_pool_price(pool_address, config["subgraph_url"])*10e11)
            tvl.append(fetch_pool_tvl(pool_address, config["subgraph_url"]))

        df = pd.DataFrame(pool_addresses, columns=["pool_address"])
        df["volume24h"] = volume24h
        df["price"] = price
        df["tvl"] = tvl

        if not os.path.exists('data/gliquid'):
            os.makedirs('data/gliquid')
        df.to_csv(f"data/gliquid/{int(time.time())}.csv", index=False)

        time.sleep(config["pull_interval"])