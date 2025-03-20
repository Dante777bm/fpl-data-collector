import requests
import json

# URL for the FPL static data API
API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

def download_fpl_data():
    try:
        # Send GET request to the API
        response = requests.get(API_URL, timeout=10)
        
        # Raise an exception for HTTP errors (4xx, 5xx)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Save to a JSON file
        with open("fpl_data.json", "w") as file:
            json.dump(data, file, indent=4)
        
        print("FPL data downloaded and saved successfully as 'fpl_data.json'")
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error: {timeout_err}")
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")

if __name__ == "__main__":
    download_fpl_data()
