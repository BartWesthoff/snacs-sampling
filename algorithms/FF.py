import numpy as np
import networkx as nx
def forest_fire_model(graph, p_f=0.5, p_b=0.3, min_percent=0.1):
    """
    Simulates a forest fire spreading through a NetworkX graph with forward and backward burning probabilities
    until a minimum percentage of nodes are burned.

    Parameters:
    graph (nx.Graph or nx.DiGraph): The NetworkX graph to simulate on.
    p_f (float): The probability of the fire spreading to a neighboring node (forward burning).
    p_b (float): The probability of the fire spreading to the source of an incoming edge (backward burning).
    min_percent (float): Minimum percentage of nodes that must be burned before stopping (0-1).

    Returns:
    tuple:
        - visited (set): Set of nodes that were 'burned' during the simulation.
        - burned_down_edges (list): List of edges within the 'burned' area.
        - burned_subgraph (nx.Graph or nx.DiGraph): Subgraph induced by the 'burned' nodes.
    """
    total_nodes = len(graph.nodes)  
    required_burn_count = int(total_nodes * min_percent)
    print(f"Required burn count: {required_burn_count}, Total nodes: {total_nodes}")
    
    burned_nodes = set()  
    unvisited_nodes = set(graph.nodes)
    while len(burned_nodes) < required_burn_count:
        if not unvisited_nodes:
            break
        
        v = np.random.choice(list(unvisited_nodes)) 
        frontier = {v} 
        unvisited_nodes.remove(v)
        burned_nodes.add(v)

        while frontier and len(burned_nodes) < required_burn_count:
            current_node = frontier.pop()  
            if graph.is_directed():
                neighbors = list(graph.successors(current_node))
                predecessors = list(graph.predecessors(current_node))
            else:
                neighbors = list(graph.neighbors(current_node))
                predecessors = []
            forward_probs = np.random.rand(len(neighbors))
            backward_probs = np.random.rand(len(predecessors)) if graph.is_directed() else []

            for neighbor, prob in zip(neighbors, forward_probs):
                if neighbor not in burned_nodes and prob < p_f:
                    burned_nodes.add(neighbor)
                    frontier.add(neighbor)
            if graph.is_directed():
                for predecessor, prob in zip(predecessors, backward_probs):
                    if predecessor not in burned_nodes and prob < p_b:
                        burned_nodes.add(predecessor)
                        frontier.add(predecessor)

    burned_down_edges = [
        edge for edge in graph.edges 
        if edge[0] in burned_nodes and edge[1] in burned_nodes
    ]
    burned_subgraph = graph.edge_subgraph(burned_down_edges).copy()
    
    return burned_nodes, burned_down_edges, burned_subgraph