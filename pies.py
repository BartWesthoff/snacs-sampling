import random
import networkx as nx
def pies_sampling(G, phi):
    """
    Implements the PIES algorithm for sampling a graph.
    
    Args:
        edges (list of tuples): A list of edges (u, v) sorted by time.
        phi (float): Probability of sampling an edge.
    
    Returns:
        nx.Graph: A sampled graph Gs.
    """
    # NOTE: Differs from original implementation, as we are using a graph instead of a list of edges
    Gs = nx.DiGraph() 
    edges = list(G.edges)
    N = phi * len(G.nodes)
    while len(Gs.nodes) < N:
        for edge in edges:
            u, v = edge
            
            # Check if both nodes are already in the graph
            if u in Gs.nodes and v in Gs.nodes:
                Gs.add_edge(u, v) 
            else:
                # Sample the edge with probability phi
                if random.random() < phi: # line 6
                    Gs.add_edge(u, v)  
    print(f"Number of nodes in Gs: {len(Gs.nodes)}, number of edges in Gs: {len(Gs.edges)}")
    return Gs