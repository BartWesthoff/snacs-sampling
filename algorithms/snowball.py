
def snowball_sampling(neighbors_dict, seed_node, depth=2):
    """
    Perform snowball sampling using a neighbors dictionary.

    Parameters:
        neighbors_dict (dict): A dictionary where keys are nodes, and values are sets of neighbors.
        seed_node (int/str): The starting node for sampling.
        depth (int): The number of layers to expand.

    Returns:
        set: A set of nodes included in the sampled subgraph.
    """
    sampled_nodes = set([seed_node])  # Start with the seed node
    current_layer = set([seed_node])

    for _ in range(depth):
        next_layer = set()
        for node in current_layer:
            next_layer.update(neighbors_dict.get(node, []))  # Add neighbors of current node
        next_layer -= sampled_nodes  # Avoid revisiting nodes
        sampled_nodes.update(next_layer)
        current_layer = next_layer  # Move to the next layer

    return sampled_nodes
