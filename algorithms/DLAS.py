import random
import networkx as nx

def dlas_algorithm(graph, num_iterations, sample_ratio):
    """
    DLAS algorithm for graph sampling with a NetworkX graph.

    Parameters:
        graph (networkx.Graph): The input graph.
        num_iterations (int): Number of iterations for the algorithm.
        sample_ratio (float): Percentage of nodes to sample from the graph.

    Returns:
        set: Sampled nodes.
    """
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    sample_size = int(num_nodes * sample_ratio)
    
    probabilities = {node: 1 / num_nodes for node in nodes} # probability vector

    sampled_nodes = set()
    current_node = random.choice(nodes)

    for _ in range(num_iterations):
        if len(sampled_nodes) >= sample_size:
            break

        neighbors = list(graph.neighbors(current_node))
        neighbors = [n for n in neighbors if n not in sampled_nodes] # remove visited nodes

        if not neighbors:
            unvisited_nodes = [node for node in nodes if node not in sampled_nodes]
            if not unvisited_nodes:
                break  # all nodes have been sampled
            current_node = random.choice(unvisited_nodes)
            continue

        probabilities_sum = sum(probabilities[n] for n in neighbors)
        normalized_probs = [probabilities[n] / probabilities_sum for n in neighbors]

        selected_node = random.choices(neighbors, weights=normalized_probs)[0]
        sampled_nodes.add(selected_node)
        probabilities[selected_node] = 0 # set probability to 0 for visited node
        total_prob = sum(probabilities.values())
        probabilities = {node: prob / total_prob for node, prob in probabilities.items()}
        current_node = selected_node
 
    Gs = nx.DiGraph()
    for node in sampled_nodes:
        for neighbor in graph.neighbors(node):
            if neighbor in sampled_nodes:
                Gs.add_edge(node, neighbor)
    return Gs