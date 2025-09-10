# GFees 1.0 - Server
This branch covers the VPS setup of the dynamic fees system.


## Files

run.sh and stop.sh
- Bash scripts to manage both ingestor at once easily. It will automatically install and start the ingestor correctly and place them in the background.

gliquid_ingestor.py
- Python script that using the configuration (/configs) will retrieve the needed metrics for all GLiquid pools of the given token pair.

dex_ingestor.py
- Python script that using the configuration (/configs) will retrieve the metrics given other protocols's subgraphs of the given token pair.

By default, both ingestors run with a 60s (1 minute) update interval. Easily changeable in the ./configs

## How is the data saved?
Data will be saved:

- ./data/dex 
All data from the dex ingestor. As .csv files with the name of each one being the unix time it was taken and inside all the information

- ./data/gliquid
All data from the gliquid ingestor. As .csv files with the name of each one being the unix time it was taken and inside all the information

## What data is saved?

### GLiquid Ingestor

For each GLiquid pool with the given pair:
- Pool address
- TVL at that moment
- Last 24H Volume hourly aggregated
- Pool price

### Dex Ingestor

For each pool of each protocol with the given pair:
- Pool address
- Last 24H Volume hourly aggregated
- Pool price

The price at the top of the .csv is the "Delta Price" metric.
It's basically the average price of all dex pools ponderated by volume.
