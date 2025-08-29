import pandas as pd
import numpy as np
import os

def get_current_season_folder():
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("FPL_Data_"):
            if "Unknown" not in item:
                return item
    return None

# Get the current season folder
season_folder = get_current_season_folder()
if not season_folder:
    print("Error: Could not find the current season's data folder.")
    exit()

# Define file paths
input_csv_path = os.path.join(season_folder, 'merged_gws.csv')
output_csv_path = os.path.join(season_folder, 'top_50_players.csv')

# Check if the input file exists
if not os.path.exists(input_csv_path):
    print(f"Error: The file {input_csv_path} does not exist.")
    exit()

# Read the CSV file
df = pd.read_csv(input_csv_path)

# 1. Calculate player statistics
# Precompute home/away goals and assists for players
df['Home_Goals'] = df.apply(lambda row: row['Goals'] if row['Was home'] else 0, axis=1)
df['Away_Goals'] = df.apply(lambda row: row['Goals'] if not row['Was home'] else 0, axis=1)
df['Home_Assists'] = df.apply(lambda row: row['Assist'] if row['Was home'] else 0, axis=1)
df['Away_Assists'] = df.apply(lambda row: row['Assist'] if not row['Was home'] else 0, axis=1)

# Aggregate player stats
player_df = df.groupby('Web name').agg(
    Position=('Position', 'first'),
    Team=('Team', 'first'),
    Cost=('Cost', 'first'),
    Selected=('Selected', 'first'),
    Form=('Form', 'first'),
    Total_Minutes=('Minutes', 'sum'),
    Home_Goals=('Home_Goals', 'sum'),
    Away_Goals=('Away_Goals', 'sum'),
    Season_Goals=('Season Goals', 'max'),
    Home_Assists=('Home_Assists', 'sum'),
    Away_Assists=('Away_Assists', 'sum'),
    Season_Assists=('Season Assists', 'max'),
    Total_Saves=('Saves', 'sum'),
    Total_CS=('CS', 'sum'),
    Total_Points=('GW Points', 'sum'),
    Total_BPS=('Bps', 'sum'),
    Total_Bonus=('Bonus', 'sum'),
    Season_xGI=('xGI', 'first'),
    Season_xGC=('xGC', 'first')
).reset_index()

# 2. CORRECTED TEAM STATISTICS CALCULATION (with Match_ID deduplication)
# Create unique match identifier to avoid over-counting stats from multiple players in the same match
df['Match_ID'] = df.apply(lambda row:
                         f"{row['Team']}_{row['Opponent Team']}_{row['Team H Score']}_{row['Team A Score']}"
                         if row['Was home'] else
                         f"{row['Opponent Team']}_{row['Team']}_{row['Team A Score']}_{row['Team H Score']}", axis=1)

# Calculate team goals and GC for each match from team perspective
team_match_df = df.copy()
team_match_df['Team_Goals'] = team_match_df.apply(lambda row: row['Team H Score'] if row['Was home'] else row['Team A Score'], axis=1)
team_match_df['Team_GC'] = team_match_df.apply(lambda row: row['Team A Score'] if row['Was home'] else row['Team H Score'], axis=1)

# Deduplicate to get one row per team per match
team_match_stats = team_match_df.groupby(['Team', 'Match_ID']).agg(
    Team_Goals=('Team_Goals', 'first'),
    Team_GC=('Team_GC', 'first'),
    Was_Home=('Was home', 'first')
).reset_index()

# Aggregate total stats from the deduplicated frame
team_total = team_match_stats.groupby('Team').agg(
    Total_team_Goals=('Team_Goals', 'sum'),
    Total_team_GC=('Team_GC', 'sum')
).reset_index()

# Aggregate home stats from the deduplicated frame
team_home = team_match_stats[team_match_stats['Was_Home'] == True].groupby('Team').agg(
    Total_team_HGoals=('Team_Goals', 'sum'),
    Total_team_HGC=('Team_GC', 'sum')
).reset_index()

# Aggregate away stats from the deduplicated frame
team_away = team_match_stats[team_match_stats['Was_Home'] == False].groupby('Team').agg(
    Total_team_AGoals=('Team_Goals', 'sum'),
    Total_team_AGC=('Team_GC', 'sum')
).reset_index()

# Merge all team statistics together
team_stats = pd.merge(team_total, team_home, on='Team', how='left')
team_stats = pd.merge(team_stats, team_away, on='Team', how='left')

# Fill NaN values with 0 (for teams that might not have home or away games recorded)
team_stats = team_stats.fillna(0)


# 3. Merge team statistics with player data
final_df = pd.merge(player_df, team_stats, on='Team', how='left')

# 4. Calculate rate statistics with zero-division handling
final_df['Min/xGI'] = np.where(
    final_df['Season_xGI'] > 0,
    final_df['Total_Minutes'] / final_df['Season_xGI'],
    np.nan
)

final_df['Min/xGC'] = np.where(
    final_df['Season_xGC'] > 0,
    final_df['Total_Minutes'] / final_df['Season_xGC'],
    np.nan
)

final_df['BPS/90'] = np.where(
    final_df['Total_Minutes'] > 0,
    (final_df['Total_BPS'] * 90) / final_df['Total_Minutes'],
    0
)

final_df['Bonus/90'] = np.where(
    final_df['Total_Minutes'] > 0,
    (final_df['Total_Bonus'] * 90) / final_df['Total_Minutes'],
    0
)

final_df['Saves/90'] = np.where(
    final_df['Total_Minutes'] > 0,
    (final_df['Total_Saves'] * 90) / final_df['Total_Minutes'],
    0
)

# Reorder columns to match the requested format
column_order = [
    'Web name', 'Position', 'Team', 'Cost', 'Selected', 'Form',
    'Home_Goals', 'Away_Goals', 'Season_Goals',
    'Home_Assists', 'Away_Assists', 'Season_Assists',
    'Total_Saves', 'Total_CS', 'Total_Points',
    'Total_team_Goals', 'Total_team_HGoals', 'Total_team_AGoals',
    'Total_team_GC', 'Total_team_HGC', 'Total_team_AGC',
    'Min/xGI', 'Min/xGC', 'BPS/90', 'Bonus/90', 'Saves/90'
]

# Keep only existing columns (in case some aren't present in the data)
valid_columns = [col for col in column_order if col in final_df.columns]
remaining_columns = [col for col in final_df.columns if col not in column_order]

# Create the final dataframe with ordered columns
result_df = final_df[valid_columns + remaining_columns]

# 5. Select only the top 50 players with highest total points
top_50_players = result_df.sort_values(by='Total_Points', ascending=False).head(50)

# Save only the top 50 players to CSV
top_50_players.to_csv(output_csv_path, index=False)

print(f"Created CSV with top 50 players based on total points at {output_csv_path}")
print(f"Total players in output: {len(top_50_players)}")
print(f"Top player: {top_50_players.iloc[0]['Web name']} with {top_50_players.iloc[0]['Total_Points']} points")
print(f"Columns: {', '.join(top_50_players.columns)}")
