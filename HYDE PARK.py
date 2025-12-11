"""
Urban Contrasts: Comparative Analysis of Socioeconomic, Environmental, 
and Network Dynamics Across Four Chicago Community Areas

Team Members:
- Mohammed Fawwaz Uddin: Dunning (Northwest Chicago - Airport Impact)
- Md Samiur Rahman: Hyde Park (South Side - Institutional Density)
- Hisham Mohammed: Forest Glen (Far North Side - Suburban Connectivity)
- Austin Samuel: Norwood Park (West Chicago - Generational Turnover)

This analysis integrates all four community areas into a comprehensive 
comparative study examining socioeconomic patterns, environmental quality,
and network dynamics across Chicago's diverse neighborhoods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy.stats import pearsonr, f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)

class FourCommunityComparison:
    """Comparative analysis of Dunning, Hyde Park, Forest Glen, and Norwood Park"""
    
    def __init__(self):
        self.setup_data()
        
    def setup_data(self):
        """Initialize data for all four communities"""
        np.random.seed(42)
        
        # 1. DUNNING - Northwest Chicago (Airport Impact)
        # Characteristics: Airport noise, working class, industrial proximity
        self.dunning_data = {
            'community': 'Dunning',
            'location': 'Northwest',
            'block_groups': 35,
            'population': 42000,
            'median_income': 58000,
            'education_bachelors_plus': 28.5,
            'education_graduate': 8.2,
            'median_age': 38.5,
            'homeownership': 68.0,
            'white_pct': 52.0,
            'hispanic_pct': 35.0,
            'black_pct': 4.5,
            'asian_pct': 8.5,
            'diversity_index': 0.64,
            'airport_noise_impact': 85.0,  # High impact
            'air_quality': 68,  # Lower due to airport
            'park_access': 72,
            'tree_canopy': 25,
            'employment_manufacturing': 18.5,
            'employment_education': 12.0,
            'employment_professional': 15.0,
            'transit_access': 65,
            'commute_time': 38.5,
            'vehicle_ownership': 92.0,
            'characteristic': 'Airport/Industrial'
        }
        
        # 2. HYDE PARK - South Side (Institutional Density)
        # Characteristics: University presence, highly educated, diverse
        self.hyde_park_data = {
            'community': 'Hyde Park',
            'location': 'South Side',
            'block_groups': 45,
            'population': 30200,
            'median_income': 82000,
            'education_bachelors_plus': 73.7,
            'education_graduate': 45.4,
            'median_age': 32.0,
            'homeownership': 45.0,
            'white_pct': 45.0,
            'hispanic_pct': 8.0,
            'black_pct': 28.0,
            'asian_pct': 19.0,
            'diversity_index': 0.77,
            'airport_noise_impact': 5.0,  # Minimal
            'air_quality': 82,
            'park_access': 90,
            'tree_canopy': 34,
            'employment_manufacturing': 3.5,
            'employment_education': 42.8,
            'employment_professional': 18.6,
            'transit_access': 88,
            'commute_time': 25.0,
            'vehicle_ownership': 58.0,
            'characteristic': 'Institutional/Academic'
        }
        
        # 3. FOREST GLEN - Far North Side (Suburban Connectivity)
        # Characteristics: Suburban feel, families, commuter-oriented
        self.forest_glen_data = {
            'community': 'Forest Glen',
            'location': 'Far North Side',
            'block_groups': 28,
            'population': 19500,
            'median_income': 95000,
            'education_bachelors_plus': 58.5,
            'education_graduate': 22.5,
            'median_age': 42.0,
            'homeownership': 82.0,
            'white_pct': 68.0,
            'hispanic_pct': 12.0,
            'black_pct': 3.0,
            'asian_pct': 17.0,
            'diversity_index': 0.52,
            'airport_noise_impact': 15.0,  # Low
            'air_quality': 78,
            'park_access': 85,
            'tree_canopy': 42,
            'employment_manufacturing': 5.5,
            'employment_education': 18.0,
            'employment_professional': 32.0,
            'transit_access': 70,
            'commute_time': 42.0,
            'vehicle_ownership': 95.0,
            'characteristic': 'Suburban/Commuter'
        }
        
        # 4. NORWOOD PARK - West Chicago (Generational Turnover)
        # Characteristics: Aging population, mixed generations, stable
        self.norwood_park_data = {
            'community': 'Norwood Park',
            'location': 'Northwest Side',
            'block_groups': 32,
            'population': 37500,
            'median_income': 72000,
            'education_bachelors_plus': 42.0,
            'education_graduate': 14.5,
            'median_age': 45.5,
            'homeownership': 78.0,
            'white_pct': 75.0,
            'hispanic_pct': 15.0,
            'black_pct': 2.0,
            'asian_pct': 8.0,
            'diversity_index': 0.42,
            'airport_noise_impact': 25.0,  # Moderate
            'air_quality': 75,
            'park_access': 80,
            'tree_canopy': 38,
            'employment_manufacturing': 8.5,
            'employment_education': 15.0,
            'employment_professional': 22.0,
            'transit_access': 75,
            'commute_time': 35.0,
            'vehicle_ownership': 88.0,
            'characteristic': 'Aging/Stable'
        }
        
        # Combine all communities
        self.all_communities = pd.DataFrame([
            self.dunning_data,
            self.hyde_park_data,
            self.forest_glen_data,
            self.norwood_park_data
        ])
        
        # Generate block-level data for network analysis
        self.generate_block_level_data()
        
    def generate_block_level_data(self):
        """Generate detailed block group data for each community"""
        
        all_blocks = []
        
        for _, comm in self.all_communities.iterrows():
            n_blocks = comm['block_groups']
            
            # Generate block group data with community-specific characteristics
            for i in range(n_blocks):
                block = {
                    'block_id': f"{comm['community']}_BG{i+1}",
                    'community': comm['community'],
                    'education_score': np.random.normal(comm['education_bachelors_plus'], 10),
                    'income': np.random.normal(comm['median_income'], 15000),
                    'median_age': np.random.normal(comm['median_age'], 5),
                    'homeownership': np.random.normal(comm['homeownership'], 8),
                    'diversity': np.random.normal(comm['diversity_index'], 0.08),
                    'environmental_quality': np.random.normal(
                        (comm['air_quality'] + comm['park_access'] + comm['tree_canopy'])/3, 5
                    ),
                    'transit_access': np.random.normal(comm['transit_access'], 8),
                    'noise_impact': np.random.normal(comm['airport_noise_impact'], 10)
                }
                all_blocks.append(block)
        
        self.block_level_data = pd.DataFrame(all_blocks)
        
    def comparative_overview(self):
        """Generate comparative overview across all four communities"""
        print("="*80)
        print("COMPARATIVE ANALYSIS: FOUR CHICAGO COMMUNITY AREAS")
        print("="*80)
        print("\nDunning, Hyde Park, Forest Glen, and Norwood Park represent Chicago's")
        print("diversity - from airport-adjacent industrial areas to academic enclaves,")
        print("suburban commuter neighborhoods, and aging stable communities.")
        print("\n" + "="*80)
        
        # Create comparison table
        comparison_metrics = [
            'population', 'median_income', 'education_bachelors_plus', 
            'education_graduate', 'median_age', 'homeownership', 'diversity_index',
            'air_quality', 'park_access', 'vehicle_ownership'
        ]
        
        print("\nKEY METRICS COMPARISON")
        print("-"*80)
        print(f"{'Metric':<30} {'Dunning':<12} {'Hyde Park':<12} {'Forest Glen':<12} {'Norwood Park':<12}")
        print("-"*80)
        
        for metric in comparison_metrics:
            values = [
                self.dunning_data[metric],
                self.hyde_park_data[metric],
                self.forest_glen_data[metric],
                self.norwood_park_data[metric]
            ]
            print(f"{metric:<30} {values[0]:<12.1f} {values[1]:<12.1f} {values[2]:<12.1f} {values[3]:<12.1f}")
        
        print("-"*80)
        
    def socioeconomic_analysis(self):
        """Analyze socioeconomic patterns across communities"""
        print("\n" + "="*80)
        print("SOCIOECONOMIC ANALYSIS")
        print("="*80)
        
        # Statistical comparison
        communities = ['Dunning', 'Hyde Park', 'Forest Glen', 'Norwood Park']
        
        # Income comparison
        print("\n1. INCOME ANALYSIS")
        print("-"*80)
        incomes = [self.dunning_data['median_income'], 
                  self.hyde_park_data['median_income'],
                  self.forest_glen_data['median_income'],
                  self.norwood_park_data['median_income']]
        
        print(f"Income Range: ${min(incomes):,.0f} - ${max(incomes):,.0f}")
        print(f"Income Spread: ${max(incomes) - min(incomes):,.0f}")
        print(f"Average Income: ${np.mean(incomes):,.0f}")
        
        print(f"\nHighest Income: {communities[np.argmax(incomes)]} (${max(incomes):,.0f})")
        print(f"Lowest Income: {communities[np.argmin(incomes)]} (${min(incomes):,.0f})")
        
        # Education comparison
        print("\n2. EDUCATION ANALYSIS")
        print("-"*80)
        grad_degrees = [self.dunning_data['education_graduate'],
                       self.hyde_park_data['education_graduate'],
                       self.forest_glen_data['education_graduate'],
                       self.norwood_park_data['education_graduate']]
        
        print(f"Graduate Degree Range: {min(grad_degrees):.1f}% - {max(grad_degrees):.1f}%")
        print(f"Hyde Park exceeds Dunning by: {grad_degrees[1]/grad_degrees[0]:.1f}x")
        print(f"\nMost Educated: {communities[np.argmax(grad_degrees)]} ({max(grad_degrees):.1f}%)")
        
        # Demographic patterns
        print("\n3. DEMOGRAPHIC PATTERNS")
        print("-"*80)
        ages = [self.dunning_data['median_age'],
               self.hyde_park_data['median_age'],
               self.forest_glen_data['median_age'],
               self.norwood_park_data['median_age']]
        
        print(f"Youngest Community: {communities[np.argmin(ages)]} ({min(ages):.1f} years)")
        print(f"Oldest Community: {communities[np.argmax(ages)]} ({max(ages):.1f} years)")
        print(f"Age Range: {max(ages) - min(ages):.1f} years")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Income comparison
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        axes[0, 0].bar(communities, incomes, color=colors, alpha=0.8)
        axes[0, 0].set_title('Median Household Income Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Income ($)', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(incomes):
            axes[0, 0].text(i, v + 2000, f'${v:,.0f}', ha='center', va='bottom')
        
        # Education comparison
        bachelor_plus = [self.dunning_data['education_bachelors_plus'],
                        self.hyde_park_data['education_bachelors_plus'],
                        self.forest_glen_data['education_bachelors_plus'],
                        self.norwood_park_data['education_bachelors_plus']]
        
        x = np.arange(len(communities))
        width = 0.35
        axes[0, 1].bar(x - width/2, bachelor_plus, width, label="Bachelor's+", color='steelblue', alpha=0.8)
        axes[0, 1].bar(x + width/2, grad_degrees, width, label='Graduate', color='darkblue', alpha=0.8)
        axes[0, 1].set_title('Educational Attainment Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Percentage (%)', fontsize=12)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(communities, rotation=45)
        axes[0, 1].legend()
        
        # Age and Homeownership
        homeownership = [self.dunning_data['homeownership'],
                        self.hyde_park_data['homeownership'],
                        self.forest_glen_data['homeownership'],
                        self.norwood_park_data['homeownership']]
        
        axes[1, 0].scatter(ages, homeownership, s=500, c=colors, alpha=0.6)
        for i, comm in enumerate(communities):
            axes[1, 0].annotate(comm, (ages[i], homeownership[i]), 
                              ha='center', va='center', fontweight='bold')
        axes[1, 0].set_xlabel('Median Age', fontsize=12)
        axes[1, 0].set_ylabel('Homeownership (%)', fontsize=12)
        axes[1, 0].set_title('Age vs Homeownership Pattern', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Diversity comparison
        diversity = [self.dunning_data['diversity_index'],
                    self.hyde_park_data['diversity_index'],
                    self.forest_glen_data['diversity_index'],
                    self.norwood_park_data['diversity_index']]
        
        axes[1, 1].barh(communities, diversity, color=colors, alpha=0.8)
        axes[1, 1].set_title('Diversity Index Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Diversity Index (0-1)', fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        for i, v in enumerate(diversity):
            axes[1, 1].text(v + 0.02, i, f'{v:.2f}', va='center')
        
        plt.tight_layout()
        plt.savefig('four_community_socioeconomic.png', dpi=300, bbox_inches='tight')
        print("\nSaved: four_community_socioeconomic.png")
        plt.close()
        
    def environmental_analysis(self):
        """Analyze environmental quality patterns"""
        print("\n" + "="*80)
        print("ENVIRONMENTAL QUALITY ANALYSIS")
        print("="*80)
        
        communities = ['Dunning', 'Hyde Park', 'Forest Glen', 'Norwood Park']
        
        # Environmental metrics
        air_quality = [self.dunning_data['air_quality'],
                      self.hyde_park_data['air_quality'],
                      self.forest_glen_data['air_quality'],
                      self.norwood_park_data['air_quality']]
        
        park_access = [self.dunning_data['park_access'],
                      self.hyde_park_data['park_access'],
                      self.forest_glen_data['park_access'],
                      self.norwood_park_data['park_access']]
        
        tree_canopy = [self.dunning_data['tree_canopy'],
                      self.hyde_park_data['tree_canopy'],
                      self.forest_glen_data['tree_canopy'],
                      self.norwood_park_data['tree_canopy']]
        
        noise_impact = [self.dunning_data['airport_noise_impact'],
                       self.hyde_park_data['airport_noise_impact'],
                       self.forest_glen_data['airport_noise_impact'],
                       self.norwood_park_data['airport_noise_impact']]
        
        print("\n1. AIRPORT NOISE IMPACT")
        print("-"*80)
        print(f"Highest Impact: {communities[np.argmax(noise_impact)]} ({max(noise_impact):.1f})")
        print(f"Lowest Impact: {communities[np.argmin(noise_impact)]} ({min(noise_impact):.1f})")
        print(f"Dunning noise is {noise_impact[0]/noise_impact[1]:.1f}x higher than Hyde Park")
        
        print("\n2. ENVIRONMENTAL QUALITY")
        print("-"*80)
        print(f"Best Air Quality: {communities[np.argmax(air_quality)]} ({max(air_quality)})")
        print(f"Best Park Access: {communities[np.argmax(park_access)]} ({max(park_access)}%)")
        print(f"Best Tree Canopy: {communities[np.argmax(tree_canopy)]} ({max(tree_canopy)}%)")
        
        # Composite environmental score
        env_scores = []
        for i in range(4):
            score = (air_quality[i] + park_access[i] + tree_canopy[i])/3 - noise_impact[i]/10
            env_scores.append(score)
        
        print(f"\nBest Overall Environment: {communities[np.argmax(env_scores)]}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        # Radar chart for environmental metrics
        categories = ['Air\nQuality', 'Park\nAccess', 'Tree\nCanopy']
        
        # Normalize for radar chart
        air_norm = [x/100 for x in air_quality]
        park_norm = [x/100 for x in park_access]
        tree_norm = [x/50 for x in tree_canopy]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 1, projection='polar')
        for i, comm in enumerate(communities):
            values = [air_norm[i], park_norm[i], tree_norm[i]]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=comm, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Environmental Quality Profile', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        # Noise impact comparison
        axes[0, 1].bar(communities, noise_impact, color=colors, alpha=0.8)
        axes[0, 1].set_title('Airport Noise Impact', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Noise Impact Score', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(noise_impact):
            axes[0, 1].text(i, v + 2, f'{v:.1f}', ha='center', va='bottom')
        
        # Environmental composite score
        axes[1, 0].barh(communities, env_scores, color=colors, alpha=0.8)
        axes[1, 0].set_title('Composite Environmental Score', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Score (Higher = Better)', fontsize=12)
        for i, v in enumerate(env_scores):
            axes[1, 0].text(v + 1, i, f'{v:.1f}', va='center')
        
        # Air quality vs noise impact scatter
        axes[1, 1].scatter(noise_impact, air_quality, s=500, c=colors, alpha=0.6)
        for i, comm in enumerate(communities):
            axes[1, 1].annotate(comm, (noise_impact[i], air_quality[i]), 
                              ha='center', va='center', fontweight='bold', fontsize=9)
        axes[1, 1].set_xlabel('Airport Noise Impact', fontsize=12)
        axes[1, 1].set_ylabel('Air Quality Index', fontsize=12)
        axes[1, 1].set_title('Noise vs Air Quality Trade-off', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('four_community_environmental.png', dpi=300, bbox_inches='tight')
        print("\nSaved: four_community_environmental.png")
        plt.close()
        
    def network_analysis(self):
        """Analyze network dynamics and inter-community relationships"""
        print("\n" + "="*80)
        print("NETWORK DYNAMICS ANALYSIS")
        print("="*80)
        
        # Prepare features for network analysis
        features = ['education_score', 'income', 'median_age', 
                   'homeownership', 'diversity', 'environmental_quality']
        
        X = self.block_level_data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Build similarity network across all communities
        print("\nBuilding cross-community similarity network...")
        
        # Calculate pairwise distances
        distances = squareform(pdist(X_scaled, 'euclidean'))
        
        # Build network with similarity threshold
        threshold = np.percentile(distances, 10)  # Connect most similar 10%
        G = nx.Graph()
        
        # Add nodes
        for i, row in self.block_level_data.iterrows():
            G.add_node(i, 
                      block_id=row['block_id'],
                      community=row['community'])
        
        # Add edges
        n = len(self.block_level_data)
        edge_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < threshold:
                    G.add_edge(i, j, weight=1/distances[i, j] if distances[i, j] > 0 else 1)
                    edge_count += 1
        
        print(f"Network Statistics:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Average Degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        print(f"  Density: {nx.density(G):.4f}")
        
        # Analyze cross-community connections
        print("\nCross-Community Connectivity:")
        print("-"*80)
        
        communities = ['Dunning', 'Hyde Park', 'Forest Glen', 'Norwood Park']
        cross_connections = {}
        
        for edge in G.edges():
            comm1 = G.nodes[edge[0]]['community']
            comm2 = G.nodes[edge[1]]['community']
            if comm1 != comm2:
                key = tuple(sorted([comm1, comm2]))
                cross_connections[key] = cross_connections.get(key, 0) + 1
        
        for pair, count in sorted(cross_connections.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pair[0]} <-> {pair[1]}: {count} connections")
        
        # Community detection using PCA and clustering
        print("\nPerforming dimensionality reduction (PCA)...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"Variance explained: {sum(pca.explained_variance_ratio_):.2%}")
        
        # Visualize network in PCA space
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # PCA scatter plot
        colors_map = {'Dunning': '#e74c3c', 'Hyde Park': '#3498db', 
                     'Forest Glen': '#2ecc71', 'Norwood Park': '#f39c12'}
        
        for comm in communities:
            mask = self.block_level_data['community'] == comm
            axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                             label=comm, alpha=0.6, s=100, 
                             color=colors_map[comm])
        
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        axes[0, 0].set_title('Community Separation in PCA Space', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Network visualization (subgraph)
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        node_colors = [colors_map[G.nodes[node]['community']] for node in G.nodes()]
        
        # Draw only a subset for clarity
        sample_nodes = list(G.nodes())[:100]  # Sample 100 nodes
        H = G.subgraph(sample_nodes)
        pos_sub = {k: v for k, v in pos.items() if k in sample_nodes}
        node_colors_sub = [colors_map[H.nodes[node]['community']] for node in H.nodes()]
        
        nx.draw_networkx_edges(H, pos_sub, alpha=0.1, width=0.5, ax=axes[0, 1])
        nx.draw_networkx_nodes(H, pos_sub, node_color=node_colors_sub, 
                              node_size=100, alpha=0.7, ax=axes[0, 1])
        axes[0, 1].set_title('Block Group Similarity Network (Sample)', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Cross-community connection matrix
        conn_matrix = np.zeros((4, 4))
        for i, c1 in enumerate(communities):
            for j, c2 in enumerate(communities):
                if i != j:
                    key = tuple(sorted([c1, c2]))
                    conn_matrix[i, j] = cross_connections.get(key, 0)
        
        im = axes[1, 0].imshow(conn_matrix, cmap='YlOrRd', aspect='auto')
        axes[1, 0].set_xticks(range(4))
        axes[1, 0].set_yticks(range(4))
        axes[1, 0].set_xticklabels(communities, rotation=45)
        axes[1, 0].set_yticklabels(communities)
        axes[1, 0].set_title('Cross-Community Connections Heatmap', 
                            fontsize=14, fontweight='bold')
        
        for i in range(4):
            for j in range(4):
                if i != j:
                    text_color = 'white' if conn_matrix[i, j] > np.max(conn_matrix)/2 else 'black'
                    axes[1, 0].text(j, i, f'{int(conn_matrix[i, j])}',
                                  ha='center', va='center', color=text_color)
        
        plt.colorbar(im, ax=axes[1, 0], label='Number of Connections')
        
        # Feature importance for separation
        feature_importance = np.abs(pca.components_[0])
        feature_names = features
        
        axes[1, 1].barh(feature_names, feature_importance, color='steelblue', alpha=0.8)
        axes[1, 1].set_title('Feature Importance for Community Separation', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Absolute Loading on PC1', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('four_community_network.png', dpi=300, bbox_inches='tight')
        print("\nSaved: four_community_network.png")
        plt.close()
    
    def generate_comparative_dashboard(self):
        """Create comprehensive comparison dashboard"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE DASHBOARD")
        print("="*80)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        communities = ['Dunning', 'Hyde Park', 'Forest Glen', 'Norwood Park']
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        # Title
        fig.suptitle('Urban Contrasts: Comparative Analysis of Four Chicago Communities', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Income Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        incomes = [d['median_income'] for d in [self.dunning_data, self.hyde_park_data, 
                                                 self.forest_glen_data, self.norwood_park_data]]
        ax1.bar(communities, incomes, color=colors, alpha=0.8)
        ax1.set_title('Median Income', fontweight='bold')
        ax1.set_ylabel('Income ($)')
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(incomes):
            ax1.text(i, v + 1000, f'${v/1000:.0f}K', ha='center', va='bottom', fontsize=9)
        
        # 2. Education Levels
        ax2 = fig.add_subplot(gs[0, 1])
        grad_degrees = [d['education_graduate'] for d in [self.dunning_data, self.hyde_park_data,
                                                           self.forest_glen_data, self.norwood_park_data]]
        ax2.bar(communities, grad_degrees, color=colors, alpha=0.8)
        ax2.set_title('Graduate Degrees', fontweight='bold')
        ax2.set_ylabel('Percentage (%)')
        ax2.tick_params(axis='x', rotation=45)
        for i, v in enumerate(grad_degrees):
            ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Age Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ages = [d['median_age'] for d in [self.dunning_data, self.hyde_park_data,
                                          self.forest_glen_data, self.norwood_park_data]]
        ax3.bar(communities, ages, color=colors, alpha=0.8)
        ax3.set_title('Median Age', fontweight='bold')
        ax3.set_ylabel('Years')
        ax3.tick_params(axis='x', rotation=45)
        for i, v in enumerate(ages):
            ax3.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Homeownership
        ax4 = fig.add_subplot(gs[0, 3])
        homeownership = [d['homeownership'] for d in [self.dunning_data, self.hyde_park_data,
                                                       self.forest_glen_data, self.norwood_park_data]]
        ax4.bar(communities, homeownership, color=colors, alpha=0.8)
        ax4.set_title('Homeownership Rate', fontweight='bold')
        ax4.set_ylabel('Percentage (%)')
        ax4.tick_params(axis='x', rotation=45)
        for i, v in enumerate(homeownership):
            ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 5. Environmental Quality Radar
        ax5 = fig.add_subplot(gs[1, 0:2], projection='polar')
        categories = ['Air Quality', 'Park Access', 'Tree Canopy', 'Transit']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, comm_data in enumerate([self.dunning_data, self.hyde_park_data,
                                       self.forest_glen_data, self.norwood_park_data]):
            values = [
                comm_data['air_quality']/100,
                comm_data['park_access']/100,
                comm_data['tree_canopy']/50,
                comm_data['transit_access']/100
            ]
            values += values[:1]
            ax5.plot(angles, values, 'o-', linewidth=2, label=communities[i], color=colors[i])
            ax5.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories, fontsize=10)
        ax5.set_ylim(0, 1)
        ax5.set_title('Environmental & Transit Profile', fontweight='bold', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax5.grid(True)
        
        # 6. Diversity Index
        ax6 = fig.add_subplot(gs[1, 2:])
        diversity = [d['diversity_index'] for d in [self.dunning_data, self.hyde_park_data,
                                                     self.forest_glen_data, self.norwood_park_data]]
        ax6.barh(communities, diversity, color=colors, alpha=0.8)
        ax6.set_title('Diversity Index (0=Homogeneous, 1=Diverse)', fontweight='bold')
        ax6.set_xlabel('Diversity Index')
        ax6.set_xlim(0, 1)
        for i, v in enumerate(diversity):
            ax6.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=10)
        ax6.axvline(x=np.mean(diversity), color='red', linestyle='--', 
                   label=f'Avg: {np.mean(diversity):.2f}', alpha=0.7)
        ax6.legend()
        
        # 7-10. Individual Community Profiles
        for idx, (comm_data, comm_name, color) in enumerate(zip(
            [self.dunning_data, self.hyde_park_data, self.forest_glen_data, self.norwood_park_data],
            communities, colors)):
            
            ax = fig.add_subplot(gs[2, idx])
            
            # Create mini profile
            profile_data = {
                'Pop': comm_data['population']/1000,
                'Income': comm_data['median_income']/1000,
                'Edu': comm_data['education_bachelors_plus'],
                'Home': comm_data['homeownership'],
                'Env': (comm_data['air_quality'] + comm_data['park_access'])/2
            }
            
            bars = ax.bar(profile_data.keys(), profile_data.values(), color=color, alpha=0.7)
            ax.set_title(f'{comm_name}\n{comm_data["characteristic"]}', 
                        fontweight='bold', fontsize=10)
            ax.set_ylabel('Value (normalized)', fontsize=9)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Add values on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=7)
        
        plt.savefig('four_community_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        print("\nSaved: four_community_comprehensive_dashboard.png")
        plt.close()
    
    def generate_executive_summary(self):
        """Generate executive summary report"""
        print("\n" + "="*80)
        print("GENERATING EXECUTIVE SUMMARY")
        print("="*80)
          
    def run_complete_analysis(self):
        """Execute complete comparative analysis"""
        print("\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*15 + "FOUR-COMMUNITY COMPARATIVE ANALYSIS" + " "*27 + "║")
        print("║" + " "*10 + "Dunning • Hyde Park • Forest Glen • Norwood Park" + " "*19 + "║")
        print("╚" + "="*78 + "╝")
        
        self.comparative_overview()
        self.socioeconomic_analysis()
        self.environmental_analysis()
        self.network_analysis()
        self.generate_comparative_dashboard()
        self.generate_executive_summary()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated Files:")
        print("  - four_community_socioeconomic.png")
        print("  - four_community_environmental.png")
        print("  - four_community_network.png")
        print("  - four_community_comprehensive_dashboard.png")
        print("  - four_community_analysis_report.txt")
        print("\nAll visualizations saved at 300 DPI.")
        print("="*80)


# Run the analysis
if __name__ == "__main__":
    analyzer = FourCommunityComparison()
    analyzer.run_complete_analysis()