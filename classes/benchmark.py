import networkx as nx
import numpy as np
from scipy.stats import ks_2samp
class BenchmarkStatic:


    def __init__(self, G):
        self.G = G
        self.Gs = None

    def base_statistics(self):
        number_of_nodes = len(self.G.nodes)
        number_of_edges = len(self.G.edges)
        self.largest_wcc = max(nx.weakly_connected_components(self.G), key=len)
        lscc = max(nx.strongly_connected_components(self.G), key=len)
        average_clustering_coefficient = nx.average_clustering(self.G,weight='rating')
        lscc_graph = self.G.subgraph(lscc)
        diameter = nx.diameter(lscc_graph)
        ninety_percentile = np.percentile(list(dict(self.G.degree()).values()),90)
        return {
            "number_of_nodes": number_of_nodes,
            "number_of_edges": number_of_edges,
            "lwcc": len(self.largest_wcc),
            "lscc": len(lscc),
            "average_clustering_coefficient": average_clustering_coefficient,
            "diameter": diameter,
            "ninety_percentile": ninety_percentile
        }
    def precompute(self):
        # S1
        in_degree = dict(self.G.in_degree())
        self.in_degree = list(in_degree.values())
        # S2
        out_degree = dict(self.G.out_degree())
        self.out_degree = list(out_degree.values())
        # S3
        wcc = list(nx.weakly_connected_components(self.G))
        self.wcc = [len(x) for x in wcc]
        # S4
        scc = list(nx.strongly_connected_components(self.G))
        self.scc = [len(x) for x in scc]
        # S5
        self.p_cum1 = self.compute_hopplot(self.G)
        # S6
        if not hasattr(self, 'largest_wcc'):
            print(f"Computing LWCC for G1")
            self.largest_wcc = max(nx.weakly_connected_components(self.G), key=len)
        self.p_cum1_sub = self.compute_hopplot(self.G.subgraph(self.largest_wcc))
        self.p_cum1_full = self.compute_hopplot(self.G)
        # S7 and S8
        A1 = nx.adjacency_matrix(self.G).toarray()
        U1, singular_values_1, _ = np.linalg.svd(A1)
        self.first_vector_1 = U1[:, 0]
        self.sorted_singular_values_1 = np.sort(singular_values_1)[::-1]
        # S9
        avg_clustering1 = self.degree_clustering_distribution(self.G)
        self.clustering_G1 = [C for _, clustering in avg_clustering1.items() for C in [clustering]]


    def S1(self):
        #  In-degree distribution
        in_degree2 = dict(self.Gs.in_degree())
        in_degree2 = list(in_degree2.values())
        if not hasattr(self, 'in_degree'):
            print(f"Computing in_degree for G1")
            in_degree = dict(self.G.in_degree())
            self.in_degree = list(in_degree.values())
        return ks_2samp(self.in_degree ,in_degree2).statistic
    
    def S2(self):
        # Out-degree distribution.
        out_degree2 = dict(self.Gs.out_degree())
        out_degree2 = list(out_degree2.values())
        if not hasattr(self, 'out_degree'):
            print(f"Computing out_degree for G1")
            out_degree = dict(self.G.out_degree())
            self.out_degree = list(out_degree.values())
        return ks_2samp(self.out_degree,out_degree2).statistic
    
    def S3(self):
        # S3: The distribution of sizes of weakly connected components (“wcc”): a set of nodes is weakly connected if for
        # any pair of nodes u and v there exists an undirected path
        # from u to v.
        wcc2 = list(nx.weakly_connected_components(self.Gs))
        wcc2 = [len(x) for x in wcc2]
        if not hasattr(self, 'wcc'):
            print(f"Computing wcc for G1")
            wcc = list(nx.weakly_connected_components(self.G))
            self.wcc = [len(x) for x in wcc]
        return ks_2samp(self.wcc,wcc2).statistic
    
    def S4(self):
        # • S4: The distribution of sizes of strongly connected
        # components (“scc”): a set of nodes is strongly connected, if
        # for any pair of nodes u and v, there exists a directed path
        # from u to v and from v to u.
        scc2 = list(nx.strongly_connected_components(self.Gs))
        scc2 = [len(x) for x in scc2]
        if not hasattr(self, 'scc'):
            print(f"Computing scc for G1")
            scc = list(nx.strongly_connected_components(self.G))
            self.scc = [len(x) for x in scc]
        return ks_2samp(self.scc,scc2).statistic
    
    def S5(self):
        # • S5: Hop-plot: the number P(h) of reachable pairs of
        # nodes at distance h or less; h is the number of hops [11].
        p_cum2_full = self.compute_hopplot(self.Gs)
        if not hasattr(self, 'p_cum1_full'):
            print(f"Computing hopplot for G1")
            self.p_cum1_full = self.compute_hopplot(self.G)
        return ks_2samp(list(self.p_cum1_full.values()),list(p_cum2_full.values())).statistic

    def S6(self):
        # • S6: Hop-plot on the largest WCC.
        largest_wcc2 = max(nx.weakly_connected_components(self.Gs), key=len)
        p_cum2_sub = self.compute_hopplot(self.Gs.subgraph(largest_wcc2))
        if not hasattr(self, 'p_cum1_sub'):
            print(f"Computing hopplot for G1")
            largest_wcc = max(nx.weakly_connected_components(self.G), key=len)
            self.p_cum1_sub = self.compute_hopplot(self.G.subgraph(largest_wcc))
        return ks_2samp(list(self.p_cum1_sub.values()),list(p_cum2_sub.values())).statistic
    
    def S7_S8(self):
        # • S7: The distribution of the first left singular vector of
        # the graph adjacency matrix versus the rank.
        # • S8: The distribution of singular values of the graph
        # adjacency matrix versus the rank. Spectral properties of
        # graphs often follow a heavy-tailed distribution [3].
        A2 = nx.adjacency_matrix(self.Gs).toarray()
        U2, singular_values_2, _ = np.linalg.svd(A2)
        first_vector_2 = U2[:, 0]
        sorted_singular_values_2 = np.sort(singular_values_2)[::-1]
        if not hasattr(self, 'sorted_singular_values_1'):
            print(f"Computing SVD for G1")
            A = nx.adjacency_matrix(self.G).toarray()
            U, singular_values, _ = np.linalg.svd(A)
            self.first_vector_1 = U[:, 0]
            self.sorted_singular_values_1 = np.sort(singular_values)[::-1]
        k_stat_svd = ks_2samp(self.sorted_singular_values_1, sorted_singular_values_2).statistic
        k_stat_first_vector = ks_2samp(self.first_vector_1, first_vector_2).statistic
        return k_stat_svd, k_stat_first_vector
    def S9(self):
         # • S9: The distribution of the clustering coefficient Cd [16]
        # defined as follows. Let node v have k neighbors; then at most
        # k(k −1)/2 edges can exist between them. Let Cv denote the
        # fraction of these allowable edges that actually exist. Then
        # Cd is defined as the average Cv over all nodes v of degree d.
        avg_clustering2 = self.degree_clustering_distribution(self.Gs)
        clustering_G2 = [C for _, clustering in avg_clustering2.items() for C in [clustering]]
        if not hasattr(self, 'clustering_G1'):
            print(f"Computing clustering for G1")
            avg_clustering = self.degree_clustering_distribution(self.G)
            self.clustering_G1 = [C for _, clustering in avg_clustering.items() for C in [clustering]]
        return ks_2samp(self.clustering_G1, clustering_G2).statistic

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
    