# FPL Data Collector

This script collects data from the Fantasy Premier League (FPL) API and saves it to a CSV file.

## Features

- Fetches comprehensive data for all players.
- Gathers information on fixtures, teams, and player positions.
- Aggregates player statistics for the current gameweek.
- Saves the data to a CSV file named `FPL_Data_GW_{current_gw}.csv` in a season-specific folder.

## How to Use

1. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the script:**

   ```bash
   python fpl_data_collector.py
   ```

The script will create a new folder for the current season (e.g., `FPL_Data_2023-24`) and save the CSV file inside it.
