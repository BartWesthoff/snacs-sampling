
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp
class BenchmarkStatic:

    def __init__(self,G,Gs):
        self.G = G
        self.Gs = Gs

    @staticmethod
    def base_statistics(G):
        number_of_nodes = len(G.nodes)
        number_of_edges = len(G.edges)
        lwcc = max(nx.weakly_connected_components(G), key=len)
        lscc = max(nx.strongly_connected_components(G), key=len)
        average_clustering_coefficient = nx.average_clustering(G,weight='rating')
        lscc_graph = G.subgraph(lscc)
        diameter = nx.diameter(lscc_graph)
        ninety_percentile = np.percentile(list(dict(G.degree()).values()),90)
        return {
            "number_of_nodes": number_of_nodes,
            "number_of_edges": number_of_edges,
            "lwcc": len(lwcc),
            "lscc": len(lscc),
            "average_clustering_coefficient": average_clustering_coefficient,
            "diameter": diameter,
            "ninety_percentile": ninety_percentile
        }


    def S1(self):
        #  In-degree distribution: for every degree d, we count
        # the number of nodes with in-degree d. Typically it follows
        # a power-law and some other heavy tailed distribution
        # compute using the kolmogorov-smirnov test
        in_degree = dict(self.G.in_degree()) # todo kijken of het beter kan
        in_degree = list(in_degree.values())
        in_degree2 = dict(self.Gs.in_degree())
        in_degree2 = list(in_degree2.values())
        return ks_2samp(in_degree,in_degree2).statistic
    
    def S2(self):
        # Out-degree distribution.
        out_degree = dict(self.G.out_degree())
        out_degree = list(out_degree.values())
        out_degree2 = dict(self.Gs.out_degree())
        out_degree2 = list(out_degree2.values())
        return ks_2samp(out_degree,out_degree2).statistic
    
    def S3(self):
        # S3: The distribution of sizes of weakly connected components (“wcc”): a set of nodes is weakly connected if for
        # any pair of nodes u and v there exists an undirected path
        # from u to v.
        wcc = list(nx.weakly_connected_components(self.G))
        wcc = [len(x) for x in wcc]
        wcc2 = list(nx.weakly_connected_components(self.Gs))
        wcc2 = [len(x) for x in wcc2]
        return ks_2samp(wcc,wcc2).statistic
    def S4(self):
        # • S4: The distribution of sizes of strongly connected
        # components (“scc”): a set of nodes is strongly connected, if
        # for any pair of nodes u and v, there exists a directed path
        # from u to v and from v to u.
        scc = list(nx.strongly_connected_components(self.G))
        scc = [len(x) for x in scc]
        scc2 = list(nx.strongly_connected_components(self.Gs))
        scc2 = [len(x) for x in scc2]
        return ks_2samp(scc,scc2).statistic
    
    def S5(self):
        # • S5: Hop-plot: the number P(h) of reachable pairs of
        # nodes at distance h or less; h is the number of hops [11].
        p_cum1 = self.compute_hopplot(self.G)
        p_cum2 = self.compute_hopplot(self.Gs)
        return ks_2samp(list(p_cum1.values()),list(p_cum2.values())).statistic

    def S6(self):
        # • S6: Hop-plot on the largest WCC.
        largest_wcc = max(nx.weakly_connected_components(self.G), key=len)
        largest_wcc2 = max(nx.weakly_connected_components(self.Gs), key=len)
        p_cum1 = self.compute_hopplot(self.G.subgraph(largest_wcc))
        p_cum2 = self.compute_hopplot(self.Gs.subgraph(largest_wcc2))
        return ks_2samp(list(p_cum1.values()),list(p_cum2.values())).statistic
    
    def S7(self):
        # • S7: The distribution of the first left singular vector of
        # the graph adjacency matrix versus the rank.
        A1 = nx.adjacency_matrix(self.G).toarray()
        A2 = nx.adjacency_matrix(self.Gs).toarray()
        #Perform Singular Value Decomposition (SVD)
        U1, _, _ = np.linalg.svd(A1)
        U2, _, _ = np.linalg.svd(A2)
        first_vector_1 = U1[:, 0]
        first_vector_2 = U2[:, 0]
        return ks_2samp(np.abs(first_vector_1), np.abs(first_vector_2)).statistic

    def S8(self):
         # • S8: The distribution of singular values of the graph
        # adjacency matrix versus the rank. Spectral properties of
        # graphs often follow a heavy-tailed distribution [3].
        # Step 2: Get the adjacency matrices for both graphs

        # TODO optimize with S7
        A1 = nx.adjacency_matrix(self.G).toarray()
        A2 = nx.adjacency_matrix(self.Gs).toarray()
        _, singular_values_1, _ = np.linalg.svd(A1)
        _, singular_values_2, _ = np.linalg.svd(A2)
        sorted_singular_values_1 = np.sort(singular_values_1)[::-1]
        sorted_singular_values_2 = np.sort(singular_values_2)[::-1]
        return ks_2samp(sorted_singular_values_1, sorted_singular_values_2).statistic
    def S9(self):
         # • S9: The distribution of the clustering coefficient Cd [16]
        # defined as follows. Let node v have k neighbors; then at most
        # k(k −1)/2 edges can exist between them. Let Cv denote the
        # fraction of these allowable edges that actually exist. Then
        # Cd is defined as the average Cv over all nodes v of degree d.
        avg_clustering1 = self.degree_clustering_distribution(self.G)
        avg_clustering2 = self.degree_clustering_distribution(self.Gs)
        clustering_G1 = [C for degree, clustering in avg_clustering1.items() for C in [clustering]]
        clustering_G2 = [C for degree, clustering in avg_clustering2.items() for C in [clustering]]
        return ks_2samp(clustering_G1, clustering_G2).statistic

    def compute_hopplot(self,graph):
        # Compute shortest paths
        all_lengths = [
            length for _, paths in nx.all_pairs_shortest_path_length(graph)
            for length in paths.values()
        ]

        # Count hops using NumPy
        unique_hops, counts = np.unique(all_lengths, return_counts=True)

        # Compute cumulative counts
        cumulative_counts = np.cumsum(counts) / (len(graph) * (len(graph) - 1))

        return dict(zip(unique_hops, cumulative_counts))
    
    def degree_clustering_distribution(self,G):
        # Get the clustering coefficients for all nodes in the graph
        clustering_coeffs = nx.clustering(G)
        
        # Group clustering coefficients by degree
        degree_clustering = {}
        
        for node, C_v in clustering_coeffs.items():
            degree = G.degree(node)
            
            if degree not in degree_clustering:
                degree_clustering[degree] = []
            
            degree_clustering[degree].append(C_v)
        
        # Calculate the average clustering coefficient for each degree
        avg_clustering = {degree: np.mean(clustering) for degree, clustering in degree_clustering.items()}
        
        return avg_clustering