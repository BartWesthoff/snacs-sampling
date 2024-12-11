import random
def dlas(graph, num_iterations, sample_ratio):
    """
    DLAS algorithm for graph sampling.

    Parameters:
        graph (dict): The outgoing neighbors mapping (adjacency list).
        num_iterations (int): Number of iterations for the algorithm.
        sample_ratio (float): Percentage of nodes to sample from the graph.

    Returns:
        set: Sampled nodes.
    """
    nodes = list(graph.keys())
    num_nodes = len(nodes)
    sample_size = int(num_nodes * sample_ratio)
    
    # Initialize probability vectors for all nodes
    probabilities = {node: 1 / num_nodes for node in nodes}

    # Start from a randomly chosen node
    sampled_nodes = set()
    current_node = random.choice(nodes)

    for _ in range(num_iterations):
        # If we've sampled enough nodes, break the loop
        if len(sampled_nodes) >= sample_size:
            break

        # Choose the next action (neighbor node) based on probabilities
        neighbors = graph.get(current_node, [])
        
        # Exclude already sampled nodes from neighbors
        neighbors = [n for n in neighbors if n not in sampled_nodes]

        if not neighbors:
            # If no unvisited neighbors, pick a random unvisited node
            unvisited_nodes = [node for node in nodes if node not in sampled_nodes]
            if not unvisited_nodes:
                break  # All nodes have been sampled
            current_node = random.choice(unvisited_nodes)
            continue

        # Normalize probabilities for the neighbors
        probabilities_sum = sum(probabilities[n] for n in neighbors)
        normalized_probs = [probabilities[n] / probabilities_sum for n in neighbors]

        # Select the next node based on probabilities
        selected_node = random.choices(neighbors, weights=normalized_probs)[0]

        # Add the selected node to the sampled set
        sampled_nodes.add(selected_node)

        # Update the probabilities
        probabilities[selected_node] = 0  # Set probability to zero to avoid revisiting

        # Optionally, redistribute the probability mass to unvisited nodes
        total_prob = sum(probabilities.values())
        probabilities = {node: prob / total_prob for node, prob in probabilities.items()}

        # Move to the selected node
        current_node = selected_node

    return sampled_nodes