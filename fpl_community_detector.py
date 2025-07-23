import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns

class FPLCommunityDetector:
    def __init__(self, fpl_data):
        self.data = fpl_data.copy()
        self.networks = {}
        self.communities = {}

    def create_performance_network(self, threshold=0.7):
        """Create network based on performance similarity"""
        # Select performance metrics
        performance_cols = ['Goals', 'Assist', 'xG', 'xA', 'ICT Index',
                          'Bps', 'GW Points', 'Form', 'Minutes']

        # Filter out players with insufficient data
        perf_data = self.data[performance_cols].fillna(0)

        # Normalize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(perf_data)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(normalized_data)

        # Create network
        G = nx.Graph()

        # Add nodes (players)
        for i, player in enumerate(self.data['Web name']):
            G.add_node(i, name=player, position=self.data.iloc[i]['Position'],
                      team=self.data.iloc[i]['Team'])

        # Add edges based on similarity threshold
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                if similarity_matrix[i][j] > threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])

        self.networks['performance'] = G
        return G

    def create_price_form_network(self, cost_threshold=1.0, form_threshold=1.0):
        """Create network based on cost and form similarity"""
        G = nx.Graph()

        # Add nodes
        for i, row in self.data.iterrows():
            G.add_node(i, name=row['Web name'], cost=row['Cost'],
                      form=row['Form'], position=row['Position'])

        # Connect players with similar cost and form
        for i in range(len(self.data)):
            for j in range(i+1, len(self.data)):
                cost_diff = abs(self.data.iloc[i]['Cost'] - self.data.iloc[j]['Cost'])
                form_diff = abs(self.data.iloc[i]['Form'] - self.data.iloc[j]['Form'])

                if cost_diff <= cost_threshold and form_diff <= form_threshold:
                    # Same position players get stronger connection
                    weight = 1.0
                    if self.data.iloc[i]['Position'] == self.data.iloc[j]['Position']:
                        weight = 1.5

                    G.add_edge(i, j, weight=weight)

        self.networks['price_form'] = G
        return G

    def create_transfer_network(self, correlation_threshold=0.5):
        """Create network based on transfer patterns"""
        # This would require historical transfer data
        # For now, we'll use current transfer in/out as proxy
        transfer_data = self.data[['Transfers In', 'Transfers Out']].fillna(0)

        # Calculate correlation between transfer patterns
        correlation_matrix = np.corrcoef(transfer_data.T)

        G = nx.Graph()

        # Add nodes
        for i, player in enumerate(self.data['Web name']):
            G.add_node(i, name=player, transfers_in=self.data.iloc[i]['Transfers In'],
                      transfers_out=self.data.iloc[i]['Transfers Out'])

        # Add edges based on transfer correlation
        for i in range(len(self.data)):
            for j in range(i+1, len(self.data)):
                # Calculate transfer similarity
                transfers_i = [self.data.iloc[i]['Transfers In'], self.data.iloc[i]['Transfers Out']]
                transfers_j = [self.data.iloc[j]['Transfers In'], self.data.iloc[j]['Transfers Out']]

                # Simple similarity based on transfer volumes
                similarity = 1 / (1 + abs(transfers_i[0] - transfers_j[0]) + abs(transfers_i[1] - transfers_j[1]))

                if similarity > correlation_threshold:
                    G.add_edge(i, j, weight=similarity)

        self.networks['transfer'] = G
        return G

    def detect_communities(self, network_type='performance'):
        """Apply Louvain algorithm to detect communities"""
        if network_type not in self.networks:
            print(f"Network type '{network_type}' not found. Available: {list(self.networks.keys())}")
            return None

        G = self.networks[network_type]

        # Apply Louvain algorithm
        partition = community_louvain.best_partition(G)

        # Calculate modularity
        modularity = community_louvain.modularity(partition, G)

        self.communities[network_type] = {
            'partition': partition,
            'modularity': modularity,
            'num_communities': len(set(partition.values()))
        }

        print(f"Found {len(set(partition.values()))} communities with modularity: {modularity:.3f}")

        return partition

    def analyze_communities(self, network_type='performance'):
        """Analyze the detected communities"""
        if network_type not in self.communities:
            print("No communities detected for this network type")
            return

        partition = self.communities[network_type]['partition']

        # Create community analysis
        community_analysis = []

        for community_id in set(partition.values()):
            # Get players in this community
            players_in_community = [i for i, comm in partition.items() if comm == community_id]

            # Get their data
            community_data = self.data.iloc[players_in_community]

            analysis = {
                'community_id': community_id,
                'size': len(players_in_community),
                'players': community_data['Web name'].tolist(),
                'positions': community_data['Position'].value_counts().to_dict(),
                'avg_cost': community_data['Cost'].mean(),
                'avg_points': community_data['GW Points'].mean(),
                'avg_form': community_data['Form'].mean(),
                'teams': community_data['Team'].value_counts().to_dict()
            }

            community_analysis.append(analysis)

        # Sort by community size
        community_analysis.sort(key=lambda x: x['size'], reverse=True)

        return community_analysis

    def visualize_communities(self, network_type='performance', figsize=(12, 8)):
        """Visualize the network with communities"""
        if network_type not in self.networks or network_type not in self.communities:
            print("Network or communities not found")
            return

        G = self.networks[network_type]
        partition = self.communities[network_type]['partition']

        plt.figure(figsize=figsize)

        # Create position layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes colored by community
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(partition.values()))))

        for i, (node, community) in enumerate(partition.items()):
            nx.draw_networkx_nodes(G, pos, nodelist=[node],
                                 node_color=[colors[community]],
                                 node_size=300, alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

        # Add labels for some nodes
        labels = {i: self.data.iloc[i]['Web name'][:8] for i in range(min(20, len(G.nodes())))}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.title(f'FPL Player Communities ({network_type})\n'
                 f'Modularity: {self.communities[network_type]["modularity"]:.3f}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
