import os
import requests
import pandas as pd
import logging
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_bootstrap_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    if response.status_code == 200:
        logging.info("Bootstrap data fetched successfully.")
        return response.json()
    else:
        logging.error(f"Failed to fetch bootstrap data: {response.status_code}")
        raise Exception(f"Failed to fetch bootstrap data: {response.status_code}")

def fetch_all_player_histories(player_ids):
    player_histories = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_player_history, player_id): player_id for player_id in player_ids
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

def get_current_gw(bootstrap_data):
    events = bootstrap_data['events']
    for event in events:
        if event['is_current']:
            logging.info(f"Current GW identified: {event['id']}.")
            return event['id']
    logging.warning("No current GW found.")
    return None

def map_team_ids_to_names(bootstrap_data):
    teams = bootstrap_data['teams']
    team_map = {team['id']: team['name'] for team in teams}
    logging.info("Team ID to name mapping created.")
    return team_map

def map_position_ids_to_names(bootstrap_data):
    elements = bootstrap_data['element_types']
    position_map = {element['id']: element['singular_name_short'] for element in elements}
    logging.info("Position ID to name mapping created.")
    return position_map

def find_next_opponent(team_id, fixtures, team_map):
    for fixture in fixtures:
        if fixture['event'] is not None and not fixture['finished']:
            if fixture['team_h'] == team_id:
                return team_map.get(fixture['team_a'], "N/A"), "Away"
            elif fixture['team_a'] == team_id:
                return team_map.get(fixture['team_h'], "N/A"), "Home"
    logging.warning(f"No upcoming fixtures found for team ID {team_id}.")
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
        'team_h_score': gw_history[0]['team_h_score'] if gw_history else None,
        'team_a_score': gw_history[0]['team_a_score'] if gw_history else None,
        'opponent_team': [gw['opponent_team'] for gw in gw_history],
        'influence': sum(float(gw['influence']) for gw in gw_history),
        'creativity': sum(float(gw['creativity']) for gw in gw_history),
        'threat': sum(float(gw['threat']) for gw in gw_history),
        'ict_index': sum(float(gw['ict_index']) for gw in gw_history),
        'bps': sum(gw['bps'] for gw in gw_history),
        'gw_points': sum(gw['total_points'] for gw in gw_history),
        'transfers_in': sum(gw['transfers_in'] for gw in gw_history),
        'transfers_out': sum(gw['transfers_out'] for gw in gw_history),
        'minutes': sum(gw['minutes'] for gw in gw_history),
        'goals': sum(gw['goals_scored'] for gw in gw_history),
        'assists': sum(gw['assists'] for gw in gw_history),
        'saves': sum(gw['saves'] for gw in gw_history),
        'gc': sum(gw['goals_conceded'] for gw in gw_history),
        'cs': sum(gw['clean_sheets'] for gw in gw_history),
        'ogs': sum(gw['own_goals'] for gw in gw_history),
        'pens_missed': sum(gw['penalties_missed'] for gw in gw_history),
        'pens_saved': sum(gw['penalties_saved'] for gw in gw_history),
        'yellow_cards': sum(gw['yellow_cards'] for gw in gw_history),
        'red_cards': sum(gw['red_cards'] for gw in gw_history)
    }

def main():
    start_time = time.time()
    try:
        # Fetch core data
        bootstrap_data = fetch_bootstrap_data()
        season = bootstrap_data.get('game_season', 'Unknown_Season').replace('/', '_')
        season_folder = f"FPL_Data_{season}"
        os.makedirs(season_folder, exist_ok=True)
        
        fixtures = fetch_fixtures()
        players = bootstrap_data['elements']
        team_map = map_team_ids_to_names(bootstrap_data)
        position_map = map_position_ids_to_names(bootstrap_data)
        current_gw = get_current_gw(bootstrap_data)
        
        if not current_gw:
            logging.error("No current GW found. Exiting script.")
            return
        
        # Create full file path
        output_file = f"{season_folder}/FPL_Data_GW_{current_gw}.csv"
        
        # Check if file already exists
        if os.path.exists(output_file):
            logging.info(f"Data for GW {current_gw} already exists. Skipping save.")
            return
        
        # Proceed with data collection
        player_ids = [player['id'] for player in players]
        player_histories = fetch_all_player_histories(player_ids)
        all_player_data = []
        
        for player in players:
            player_id = player['id']
            web_name = player['web_name']
            position = position_map[player['element_type']]
            team = team_map[player['team']]
            cost = player['now_cost'] / 10
            selected = player['selected_by_percent']
            form = player['form']
            status = player['status']
            
            # Get GW-specific stats
            player_history = player_histories.get(player_id, {})
            gw_history = [gw for gw in player_history.get('history', []) if gw['round'] == current_gw]
            aggregated_stats = aggregate_player_stats(gw_history)
            
            # Get next fixture
            next_opponent, venue = find_next_opponent(player['team'], fixtures, team_map)
            
            all_player_data.append({
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
                'Season Goals': player['goals_scored'],
                'Season Assists': player['assists'],
                'xA': player['expected_assists'],
                'xG': player['expected_goals'],
                'xGI': player['expected_goal_involvements'],
                'xGC': player['expected_goals_conceded'],
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
                'Opponent Team': ', '.join([team_map.get(opponent, "N/A") for opponent in aggregated_stats['opponent_team']]),
                'Influence': aggregated_stats['influence'],
                'Creativity': aggregated_stats['creativity'],
                'Threat': aggregated_stats['threat'],
                'ICT Index': aggregated_stats['ict_index'],
                'Bps': aggregated_stats['bps'],
                'GW Points': aggregated_stats['gw_points'],
                'Transfers In': aggregated_stats['transfers_in'],
                'Transfers Out': aggregated_stats['transfers_out'],
                'Next Fixture': f"{next_opponent} ({venue})" if next_opponent != "N/A" else "N/A",
            })
        
        # Save data
        df = pd.DataFrame(all_player_data)
        df.to_csv(output_file, index=False)
        logging.info(f"Data saved to {output_file}.")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        end_time = time.time()
        logging.info(f"Script execution completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
