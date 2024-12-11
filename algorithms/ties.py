import random
import networkx as nx

def ties_sampling(G, phi) -> nx.DiGraph:
    """
    Perform TIES sampling on a network.

    Parameters:
        G (networkx.Graph): Input graph.
        phi (float): Fraction of nodes to sample (0 < phi <= 1).

    Returns:
        networkx.Graph: Sampled subgraph.
    """
    Vs = set()
    Es = set()

    #Edge-based node sampling step 4-8
    edges = list(G.edges)
    while len(Vs) < phi * len(G.nodes):
        r = random.randint(0, len(edges) - 1)  # Uniformly random index
        u, v = edges[r]                      
        Vs.update([u, v])                    

    #Graph induction step 10-15
    for u, v in edges:
        if u in Vs and v in Vs:
            Es.add((u, v))

    #Create the sampled subgraph 16-end
    Gs = nx.DiGraph()
    Gs.add_nodes_from(Vs)
    Gs.add_edges_from(Es)

    return Gs