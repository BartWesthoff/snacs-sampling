import networkx as nx
import numpy as np
from scipy.stats import ks_2samp

class BenchmarkTemporal:

    def __init__(self,Gt,St):
        self.Gt = Gt[:-1] # TODO overleggen, sampling 100% from nodes and comparing is a waste of time
        self.St = St[:-1]


    def b_flow_efficiency(self,G: nx.DiGraph) -> float:
        """
        Calculate the Flow Efficiency of a directed graph.
        
        Flow Efficiency is defined as the sum of the reciprocal shortest path lengths 
        between all pairs of nodes (i, j) in the graph, divided by the total possible 
        number of node pairs.
        
        Parameters:
            G (nx.DiGraph): A NetworkX directed graph.
        
        Returns:
            float: The flow efficiency of the graph.
        """
        # Number of nodes in the graph
        n = G.number_of_nodes()
        
        # Total possible pairs of nodes
        total_pairs = n * (n - 1)
        
        # Handle the case of an empty graph or a single node
        if total_pairs == 0:
            return 0.0
        
        # Calculate reciprocal of shortest path lengths
        efficiency_sum = 0
        for source in G.nodes():
            # Get shortest path lengths from the source node
            shortest_paths = nx.single_source_shortest_path_length(G, source)
            for target, length in shortest_paths.items():
                if source != target and length > 0:
                    efficiency_sum += 1 / length
        
        # Compute flow efficiency
        flow_efficiency = efficiency_sum / total_pairs
        return flow_efficiency
    
    def b_edge_density(self,G: nx.DiGraph) -> float:
        """
        Calculate the edge density of a directed graph.
        
        Edge density is defined as the ratio of the number of edges in the graph 
        to the total possible number of edges between all pairs of nodes.
        
        Parameters:
            G (nx.DiGraph): A NetworkX directed graph.
        
        Returns:
            float: The edge density of the graph.
        """
        # Number of nodes in the graph
        n = G.number_of_nodes()
        
        # Total possible pairs of nodes
        total_pairs = n * (n - 1)
        
        # Handle the case of an empty graph or a single node
        if total_pairs == 0:
            return 0.0
        
        # Calculate the edge density
        edge_density = G.number_of_edges() / total_pairs
        return edge_density
    

    def b_compute_weighted_hits_hubs(self,G: nx.DiGraph, weight='rating', max_iter=100, tol=1e-8):
        """
        Computes the HITS algorithm with weighted edges to calculate hub scores in a directed graph.

        Parameters:
            G (nx.DiGraph): A directed graph with weighted edges.
            weight (str): The edge attribute used as the weight.
            max_iter (int): Maximum number of iterations for convergence.
            tol (float): Convergence tolerance for changes in scores.

        Returns:
            dict: A dictionary where keys are nodes and values are their hub scores.
        """
        # Initialize hub and authority scores to 1
        hubs = {node: 1.0 for node in G}
        authorities = {node: 1.0 for node in G}
        
        for _ in range(max_iter):
            # Update authority scores based on weighted predecessors
            new_authorities = {
                node: sum(hubs[neighbor] * G[neighbor][node].get(weight, 1) for neighbor in G.predecessors(node))
                for node in G
            }
            
            # Update hub scores based on weighted successors
            new_hubs = {
                node: sum(new_authorities[neighbor] * G[node][neighbor].get(weight, 1) for neighbor in G.successors(node))
                for node in G
            }
            
            # Normalize the scores
            norm_authorities = sum(new_authorities.values())
            norm_hubs = sum(new_hubs.values())
            
            new_authorities = {node: score / norm_authorities for node, score in new_authorities.items()}
            new_hubs = {node: score / norm_hubs for node, score in new_hubs.items()}
            
            # Check for convergence
            if all(abs(new_hubs[node] - hubs[node]) < tol for node in G) and \
            all(abs(new_authorities[node] - authorities[node]) < tol for node in G):
                break
            
            hubs = new_hubs
            authorities = new_authorities
        # create dict with key the node and as value a dict with the hub score and authority score
        dict_all = {}
        for node in G:
            dict_all[node] = {'hub': hubs[node], 'authority': authorities[node]}
        return dict_all
    

    def T1(self):
        """
        Flow Efficiency: Compare the flow efficiency of the graphs.
        """
        efficiency_g = [self.b_flow_efficiency(g_t) for g_t in self.Gt]
        efficiency_s = [self.b_flow_efficiency(g_s) for g_s in self.St]
        statistic  = ks_2samp(efficiency_g, efficiency_s).statistic
        return statistic
    

    def T2(self):
        """
        Edge Density Over Time: Compare the edge density of the graph over time.
        """
        density_g = [self.b_edge_density(g_t) for g_t in self.Gt]
        density_s = [self.b_edge_density(g_s) for g_s in self.St]
        statistic = ks_2samp(density_g, density_s).statistic
        return statistic
    
    def T3(self):
        """
        Hub Scores: Compare the hub scores of the graphs.
        """
        hubs_authorities_g_dict = [benchmark.b_compute_weighted_hits_hubs(g_t) for g_t in Gt]
        hubs_authorities_s_dict = [benchmark.b_compute_weighted_hits_hubs(g_s) for g_s in St]
        hubs_g_dict_all = {}
        hubs_s_dict_all = {}
        for hubs in hubs_authorities_g_dict:
            for key, value in hubs.items():
                if key in hubs_g_dict_all:
                    hubs_g_dict_all[key].append(value)
                else:
                    hubs_g_dict_all[key] = [value]
        for hubs in hubs_authorities_s_dict:
            for key, value in hubs.items():
                if key in hubs_s_dict_all:
                    hubs_s_dict_all[key].append(value)
                else:
                    hubs_s_dict_all[key] = [value]

        # now instead of a list get the mean of the values
        hubs_g = {}
        for key,value in hubs_g_dict_all.items():
            hubs_g[key] = {'hub': np.mean([x['hub'] for x in value]), 'authority': np.mean([x['authority'] for x in value])}
        hubs_s = {}
        for key,value in hubs_s_dict_all.items():
            hubs_s[key] = {'hub': np.mean([x['hub'] for x in value]), 'authority': np.mean([x['authority'] for x in value])}
        statistic_hubs = ks_2samp([x['hub'] for x in hubs_g.values()], [x['hub'] for x in hubs_s.values()]).statistic
        statistic_authorities = ks_2samp([x['authority'] for x in hubs_g.values()], [x['authority'] for x in hubs_s.values()]).statistic
        return {'statistic_hubs':statistic_hubs, 'statistic_authorities':statistic_authorities}

    def T4(self):
        """
        Compare PageRank distributions between real and sampled nodes using KS statistic.
        # first get the pagerank for each node over time
        # then get the ks statistic and p-value for each node
        # at last get the mean of the ks statistic and p-value per node
        Returns:
            dict: KS statistic and p-value for each node.
        """
        # Compute PageRank for real and sampled graphs
        real_pagerank = [nx.pagerank(graph) for graph in self.Gt]
        sampled_pagerank = [nx.pagerank(graph) for graph in self.St]
  
        real_scores = {}    # over time so K times for each slice
        sampled_scores = {} # over time so K times for each slice
        
        for t in range(len(real_pagerank)):
            for node in real_pagerank[t]:
                if node in real_scores:
                    real_scores[node].append(real_pagerank[t][node])
                else:
                    real_scores[node] = [real_pagerank[t][node]]
            for node in sampled_pagerank[t]:
                if node in sampled_scores:
                    sampled_scores[node].append(sampled_pagerank[t][node])
                else:
                    sampled_scores[node] = [sampled_pagerank[t][node]]
        
        # get ks statistic and p-value for each node
        ks_statistic = {}
        for node in real_scores:
            if node in sampled_scores:
                ks_statistic[node]= ks_2samp(real_scores[node], sampled_scores[node]).statistic
        ks_statistic = np.mean(list(ks_statistic.values()))
        return ks_statistic

    def T5(self):
        """
        Betweenness Centrality: Compare the betweenness centrality of the graphs.
        """
        betweenness_g = [nx.betweenness_centrality(g_t,weight='rating',k=int(len(g_t.nodes())*0.05)) for g_t in self.Gt]
        betweenness_s = [nx.betweenness_centrality(g_s,weight='rating',k=int(len(g_s.nodes())*0.05)) for g_s in self.St]

        betweenness_g_dict_all = {}
        betweenness_s_dict_all = {}
        for betweenness in betweenness_g:
            for key, value in betweenness.items():
                if key in betweenness_g_dict_all:
                    betweenness_g_dict_all[key].append(value)
                else:
                    betweenness_g_dict_all[key] = [value]
        for betweenness in betweenness_s:
            for key, value in betweenness.items():
                if key in betweenness_s_dict_all:
                    betweenness_s_dict_all[key].append(value)
                else:
                    betweenness_s_dict_all[key] = [value]
        # now instead of a list get the mean of the values
        betweenness_g = {}
        for key,value in betweenness_g_dict_all.items():
            betweenness_g[key] = np.mean(value)
        betweenness_s = {}
        for key,value in betweenness_s_dict_all.items():
            betweenness_s[key] = np.mean(value)

        statistic = ks_2samp(list(betweenness_g.values()), list(betweenness_s.values())).statistic
        return statistic