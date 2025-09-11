# GFees 1.0 - Server
This branch covers the VPS setup of the dynamic fees system.


## Files

run.sh and stop.sh
- Bash scripts to manage both ingestor and fee runner at once easily. It will automatically install and start the scripts correctly and place them in the background.

gliquid_ingestor.py
- Python script that using the configuration (/configs) will retrieve the needed metrics for all GLiquid pools of the given token pair.

dex_ingestor.py
- Python script that using the configuration (/configs) will retrieve the metrics given other protocols's subgraphs of the given token pair.

fee_runner.py
- Python script that reads the data from the ingestors, computes the metrics, and prints the fee proposals.

By default, both ingestors run with a 60s (1 minute) update interval. Easily changeable in the ./configs
The fee runner has 3 time-related parameters:
- poll_interval: Update interval on raw data. Ideally, sinchronized with the two ingestors.
- start_threshold: "Warmup time". How many poll_intervals will wait before starting to compute metrics which need past data.
- proposal_interval: Fee proposal interval. How many update interval loops will wait between each fee proposal. Has to take into account tx gas costs.

An ideal timeframe could be:
- Ingestors and fee runners with a polling interval of 60s for fresh and updated data.
- A start_threshold which lands in about 2 hours. So it gets enough data to start with "warm" metrics.
- A proposal_interval of 15 minutes. A balance between gas costs and dynamic-enough fees.

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
