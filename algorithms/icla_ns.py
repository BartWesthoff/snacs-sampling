import random
from collections import deque

def icla_ns_directed(outgoing_neighbors, f, tau):
    """
    Implements the ICLA-NS algorithm for directed graphs.

    Parameters:
        outgoing_neighbors (dict): Node-to-outgoing-neighbors mapping.
        f (float): Fraction of nodes to sample.
        tau (float): Convergence threshold.

    Returns:
        sampled_nodes (set): The set of sampled nodes.
    """
    # Initialization
    num_nodes = len(outgoing_neighbors)
    num_sampled_nodes = int(f * num_nodes)
    sampled_nodes = set(random.sample(list(outgoing_neighbors.keys()), num_sampled_nodes))
    q = deque(sampled_nodes)  # Queue of sampled nodes

    # Automata mapping phase
    automata = {node: list(neighbors.union({node})) for node, neighbors in outgoing_neighbors.items()}
    state = {node: random.choice(automata[node]) for node in sampled_nodes}

    # Improvement phase
    while q:
        vi = q.popleft()
        automaton_vi = automata[vi]
        selected_action = state[vi]  # Use the current state as the action

        # Evaluate the selected action
        if selected_action in automaton_vi:  # Action must be valid (self or outgoing neighbor)
            if selected_action in sampled_nodes:
                # Reward: Keep the state
                pass
            else:
                # Penalize: Update state to another action
                state[vi] = random.choice(automaton_vi)
        else:
            # Penalize: Update state to another action
            state[vi] = random.choice(automaton_vi)

        # Add the selected action to the sampled set if valid
        if selected_action not in sampled_nodes:
            sampled_nodes.add(selected_action)
            q.append(selected_action)

        # Convergence check
        if len(sampled_nodes) / num_nodes >= tau:
            break

    return sampled_nodes