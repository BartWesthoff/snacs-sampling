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
    target = len(G.edges) * phi
    for index,edge in enumerate(edges):
        u, v = edge
        
        # choice between error and accuracy
        if index > 0 and index % 10 == 0 and target < len(Gs.edges):
            break
        # Check if both nodes are already in the graph
        if u in Gs.nodes and v in Gs.nodes:
            Gs.add_edge(u, v) 
            # print('add edge',u,v)
        else:
            # Sample the edge with probability phi
            if random.random() < phi: # line 6
                Gs.add_edge(u, v)  
                # print('add edge',u,v)
    return Gs