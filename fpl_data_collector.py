import os
import requests
import pandas as pd
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_bootstrap_data():
    """Fetches FPL bootstrap data from the API."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        logging.info("Bootstrap data fetched successfully.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching bootstrap data: {e}")
        raise

def fetch_all_player_histories(player_ids):
    player_histories = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_player_history, player_id): player_id
            for player_id in player_ids
        }
        for future in futures:
            player_id = futures[future]
            try:
                player_histories[player_id] = future.result()
            except Exception as e:
                logging.error(f"Error fetching history for player ID {player_id}: {e}")
    return player_histories

def fetch_player_history(player_id):
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    response = requests.get(url)
    if response.status_code == 200:
        logging.debug(f"Player history fetched for ID {player_id}.")
        return response.json()
    else:
        logging.error(f"Failed to fetch player history for ID {player_id}: {response.status_code}")
        raise Exception(f"Failed to fetch player history for ID {player_id}: {response.status_code}")

def fetch_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    if response.status_code == 200:
        logging.info("Fixtures fetched successfully.")
        return response.json()
    else:
        logging.error(f"Failed to fetch fixtures: {response.status_code}")
        raise Exception(f"Failed to fetch fixtures: {response.status_code}")

def map_team_ids_to_names(bootstrap_data):
    teams = bootstrap_data.get('teams', [])
    team_map = {team['id']: team['name'] for team in teams}
    logging.info("Team ID to name mapping created.")
    return team_map

def map_position_ids_to_names(bootstrap_data):
    elements = bootstrap_data.get('element_types', [])
    position_map = {element['id']: element['singular_name_short'] for element in elements}
    logging.info("Position ID to name mapping created.")
    return position_map

def find_next_opponent(team_id, fixtures, team_map):
    for fixture in fixtures:
        if fixture.get('event') and not fixture.get('finished'):
            if fixture['team_h'] == team_id:
                return team_map.get(fixture['team_a'], "N/A"), "Away"
            elif fixture['team_a'] == team_id:
                return team_map.get(fixture['team_h'], "N/A"), "Home"
    return "N/A", "N/A"

def aggregate_player_stats(gw_history):
    if not gw_history:
        return {
            'starts': 0,
            'was_home': False,
            'team_h_score': None,
            'team_a_score': None,
            'opponent_team': [],
            'influence': 0.0,
            'creativity': 0.0,
            'threat': 0.0,
            'ict_index': 0.0,
            'bps': 0,
            'gw_points': 0,
            'transfers_in': 0,
            'transfers_out': 0,
            'minutes': 0,
            'goals': 0,
            'assists': 0,
            'saves': 0,
            'gc': 0,
            'cs': 0,
            'ogs': 0,
            'pens_missed': 0,
            'pens_saved': 0,
            'yellow_cards': 0,
            'red_cards': 0
        }
    return {
        'starts': sum(gw['starts'] for gw in gw_history),
        'was_home': any(gw['was_home'] for gw in gw_history),
        'team_h_score': gw_history[0].get('team_h_score') if gw_history else None,
        'team_a_score': gw_history[0].get('team_a_score') if gw_history else None,
        'opponent_team': [gw.get('opponent_team', 'N/A') for gw in gw_history],
        'influence': sum(float(gw.get('influence', 0)) for gw in gw_history),
        'creativity': sum(float(gw.get('creativity', 0)) for gw in gw_history),
        'threat': sum(float(gw.get('threat', 0)) for gw in gw_history),
        'ict_index': sum(float(gw.get('ict_index', 0)) for gw in gw_history),
        'bps': sum(gw.get('bps', 0) for gw in gw_history),
        'gw_points': sum(gw.get('total_points', 0) for gw in gw_history),
        'transfers_in': sum(gw.get('transfers_in', 0) for gw in gw_history),
        'transfers_out': sum(gw.get('transfers_out', 0) for gw in gw_history),
        'minutes': sum(gw.get('minutes', 0) for gw in gw_history),
        'goals': sum(gw.get('goals_scored', 0) for gw in gw_history),
        'assists': sum(gw.get('assists', 0) for gw in gw_history),
        'saves': sum(gw.get('saves', 0) for gw in gw_history),
        'gc': sum(gw.get('goals_conceded', 0) for gw in gw_history),
        'cs': sum(gw.get('clean_sheets', 0) for gw in gw_history),
        'ogs': sum(gw.get('own_goals', 0) for gw in gw_history),
        'pens_missed': sum(gw.get('penalties_missed', 0) for gw in gw_history),
        'pens_saved': sum(gw.get('penalties_saved', 0) for gw in gw_history),
        'yellow_cards': sum(gw.get('yellow_cards', 0) for gw in gw_history),
        'red_cards': sum(gw.get('red_cards', 0) for gw in gw_history)
    }

def process_gameweek(current_gw, bootstrap_data, fixtures, season_folder, team_map, position_map):
    """Processes and saves data for a single gameweek."""
    logging.info(f"Starting data processing for GW {current_gw}.")

    output_file = os.path.join(season_folder, f"FPL_Data_GW_{current_gw}.csv")

    players = bootstrap_data.get('elements', [])
    player_ids = [player['id'] for player in players]
    player_histories = fetch_all_player_histories(player_ids)
    all_player_data = []

    for player in players:
        player_id = player['id']
        web_name = player.get('web_name', 'N/A')
        position = position_map.get(player.get('element_type'), 'N/A')
        team = team_map.get(player.get('team'), 'N/A')
        cost = player.get('now_cost', 0) / 10
        selected = player.get('selected_by_percent', 'N/A')
        form = player.get('form', 'N/A')
        status = player.get('status', 'N/A')

        player_history = player_histories.get(player_id, {})
        gw_history = [
            gw for gw in player_history.get('history', [])
            if gw.get('round') == current_gw
        ]
        aggregated_stats = aggregate_player_stats(gw_history)

        next_opponent, venue = find_next_opponent(
            player.get('team', 0),
            fixtures,
            team_map
        )

        row = {
            'Web name': web_name,
            'Position': position,
            'Team': team,
            'Cost': cost,
            'Selected': selected,
            'Form': form,
            'Status': status,
            'Minutes': aggregated_stats['minutes'],
            'Goals': aggregated_stats['goals'],
            'Assist': aggregated_stats['assists'],
            'Saves': aggregated_stats['saves'],
            'GC': aggregated_stats['gc'],
            'Season Goals': player.get('goals_scored', 0),
            'Season Assists': player.get('assists', 0),
            'xA': player.get('expected_assists', 0),
            'xG': player.get('expected_goals', 0),
            'xGI': player.get('expected_goal_involvements', 0),
            'xGC': player.get('expected_goals_conceded', 0),
            'CS': aggregated_stats['cs'],
            'OGs': aggregated_stats['ogs'],
            'Pens Missed': aggregated_stats['pens_missed'],
            'Pens Saved': aggregated_stats['pens_saved'],
            'Yellow cards': aggregated_stats['yellow_cards'],
            'Red cards': aggregated_stats['red_cards'],
            'Starts': aggregated_stats['starts'],
            'Was home': aggregated_stats['was_home'],
            'Team H Score': aggregated_stats['team_h_score'],
            'Team A Score': aggregated_stats['team_a_score'],
            'Opponent Team': ', '.join(
                [team_map.get(opponent, 'N/A')
                 for opponent in aggregated_stats['opponent_team']]
            ),
            'Influence': aggregated_stats['influence'],
            'Creativity': aggregated_stats['creativity'],
            'Threat': aggregated_stats['threat'],
            'ICT Index': aggregated_stats['ict_index'],
            'Bps': aggregated_stats['bps'],
            'GW Points': aggregated_stats['gw_points'],
            'Transfers In': aggregated_stats['transfers_in'],
            'Transfers Out': aggregated_stats['transfers_out'],
            'Next Fixture': f"{next_opponent} ({venue})"
                if next_opponent != "N/A" else "N/A",
        }
        all_player_data.append(row)

    df = pd.DataFrame(all_player_data)
    df.to_csv(output_file, index=False)
    logging.info(f"Data saved to {output_file}.")

def main():
    start_time = time.time()
    try:
        bootstrap_data = fetch_bootstrap_data()
        events = bootstrap_data.get('events', [])
        if not events:
            logging.error("No events found in bootstrap data. Exiting script.")
            return

        year = events[0]['deadline_time'].split('-')[0]
        season = f"{year}-{str(int(year) + 1)[-2:]}"
        season_folder = f"FPL_Data_{season}"
        os.makedirs(season_folder, exist_ok=True)
        
        fixtures = fetch_fixtures()
        team_map = map_team_ids_to_names(bootstrap_data)
        position_map = map_position_ids_to_names(bootstrap_data)
        
        now_utc = datetime.now(timezone.utc)
        
        processed_a_gameweek = False
        for event in events:
            gw_id = event['id']
            
            output_file = os.path.join(season_folder, f"FPL_Data_GW_{gw_id}.csv")
            
            if os.path.exists(output_file):
                logging.debug(f"Data for GW {gw_id} already exists. Overwriting.")

            deadline_str = event['deadline_time']
            deadline_time = datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))

            if now_utc > deadline_time:
                process_gameweek(gw_id, bootstrap_data, fixtures, season_folder, team_map, position_map)
                processed_a_gameweek = True

        if not processed_a_gameweek:
            logging.info("No new gameweeks to process at this time.")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
    finally:
        end_time = time.time()
        logging.info(f"Script execution completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
