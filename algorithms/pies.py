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
    edges = list(G.edges)
    Gs = nx.DiGraph() 
    target = G.number_of_nodes() * phi
    for index,edge in enumerate(edges):
        u, v = edge
        
        if index > 0 and index % 10 == 0 and Gs.number_of_nodes() >= target:
            break
        if u in Gs.nodes and v in Gs.nodes:
            Gs.add_edge(u, v) 
        else:
            if random.random() < phi: # line 6
                Gs.add_edge(u, v)  
    return Gs