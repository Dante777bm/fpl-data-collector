import pandas as pd
import networkx as nx
import community as community_louvain
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_and_prepare_data(filepath):
    """Loads and prepares the FPL data from a CSV file."""
    df = pd.read_csv(filepath)

    # Fill missing values and select relevant features
    features = [
        'Cost', 'Selected', 'Form', 'Goals', 'Assist', 'xA', 'xG',
        'xGI', 'xGC', 'CS', 'Influence', 'Creativity', 'Threat', 'ICT Index', 'Bps'
    ]

    df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Normalize the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    return df, df_scaled, features

def build_network(df, df_scaled, similarity_threshold=0.8):
    """Builds a network of players based on performance similarity."""
    similarity_matrix = cosine_similarity(df_scaled)

    G = nx.Graph()
    for i in range(len(df)):
        G.add_node(i, name=df['Web name'][i], position=df['Position'][i], team=df['Team'][i])

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i, j] > similarity_threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    return G

def detect_communities(G):
    """Detects communities using the Louvain algorithm."""
    partition = community_louvain.best_partition(G)
    return partition

def main():
    """Main function to run the FPL analysis."""
    # Load and prepare the data
    filepath = 'FPL_Data_2025-26/FPL_Data_GW_1.csv'
    df, df_scaled, features = load_and_prepare_data(filepath)

    # Build the network
    G = build_network(df, df_scaled)

    # Detect communities
    partition = detect_communities(G)

    # Add community information to the DataFrame
    df['Community'] = df.index.map(partition)

    # Save the results
    output_path = 'FPL_Data_2025-26/FPL_Player_Communities.csv'
    df.to_csv(output_path, index=False)

    print(f"Player communities saved to {output_path}")

if __name__ == "__main__":
    main()
