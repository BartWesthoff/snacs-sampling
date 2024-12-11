import random
import numpy as np
import pandas as pd

class Automaton:
    def __init__(self, node_id, neighbors):
        self.node_id = node_id
        self.neighbors = neighbors  # Set of neighbor nodes
        if neighbors:
            self.action_probs = {neighbor: 1.0 / len(neighbors) for neighbor in neighbors}
        else:
            self.action_probs = {}
        self.last_action = None
        self.activity_level = 'Passive'  # 'Passive', 'Active', 'Fire', 'Off'


def normalize_probs(prob_dict):
    total = sum(prob_dict.values())
    if total > 0:
        for k in prob_dict:
            prob_dict[k] /= total
    return prob_dict


def compute_uncharted_scores(S):
    """Compute uncharted scores based on visitation frequencies."""
    # Less visited nodes should have higher uncharted scores
    visits = np.array(list(S.values()))
    # Define an uncharted threshold as the median of visited counts
    threshold = np.median(visits) if len(visits) > 0 else 0
    # Uncharted score = max(0, threshold - visit_count), normalized later
    raw_scores = {node: max(0, threshold - S[node]) for node in all_nodes}
    total = sum(raw_scores.values()) if sum(raw_scores.values()) > 0 else 1
    uncharted_scores = {node: raw_scores[node] / total for node in all_nodes}
    return uncharted_scores


def due_sampling_directed(all_nodes, outgoing_neighbors, K=1000, sampling_ratio=0.5,
                          reward_strength=0.1, penalty_strength=0.1, update_interval=50):
    """
    Decentralized Uncharted Expansion (DUE) sampling for a directed graph.

    Parameters:
    - all_nodes: A set of all node IDs.
    - outgoing_neighbors: Dictionary mapping each node to its set of outgoing neighbors.
    - K: Maximum number of iterations.
    - sampling_ratio: Fraction of nodes to include in the final sampled network.
    - reward_strength: Magnitude of reward updates.
    - penalty_strength: Magnitude of penalty updates.
    - update_interval: Frequency (in iterations) at which uncharted scores are recalculated.

    Returns:
    - sampled_nodes: The list of sampled nodes.
    - Gs_edges: List of directed edges in the sampled subgraph.
    """
    # Initialize automata
    automata = {}
    for node in all_nodes:
        out_neighbors = outgoing_neighbors.get(node, set())
        automata[node] = Automaton(node_id=node, neighbors=out_neighbors)

    # Visited count
    S = {node: 0 for node in all_nodes}  
    # Automaton state sets
    passive_set = set(all_nodes)  # Passive
    active_set = set()  # Active
    fire_set = set()    # Fire
    off_set = set()     # Off

    # Select a starting node at random
    starting_node = random.choice(list(passive_set))
    automata[starting_node].activity_level = 'Active'
    active_set.add(starting_node)
    passive_set.remove(starting_node)

    uncharted_scores = compute_uncharted_scores(S)

    k = 1
    while k <= K:
        if not active_set:
            # If no active automata, select a new start
            candidates = passive_set if passive_set else all_nodes
            starting_node = random.choice(list(candidates))
            automata[starting_node].activity_level = 'Active'
            active_set.add(starting_node)
            passive_set.discard(starting_node)

        # Pick an active automaton to "fire"
        firing_node = random.choice(list(active_set))
        firing_automaton = automata[firing_node]
        active_set.remove(firing_node)
        fire_set = {firing_node}

        # Automaton chooses an action (outgoing neighbor)
        if firing_automaton.action_probs:
            neighbors = list(firing_automaton.action_probs.keys())
            probabilities = list(firing_automaton.action_probs.values())
            chosen_neighbor = np.random.choice(neighbors, p=probabilities)
            firing_automaton.last_action = chosen_neighbor

            # Update visitation
            S[firing_node] += 1
            S[chosen_neighbor] += 1

            # Activate neighbors of chosen_neighbor if they are passive
            for nbr in outgoing_neighbors.get(chosen_neighbor, set()):
                if automata[nbr].activity_level == 'Passive':
                    automata[nbr].activity_level = 'Active'
                    active_set.add(nbr)
                    passive_set.discard(nbr)
        else:
            chosen_neighbor = None

        # Determine reward or penalty:
        # If the chosen action leads to a node with a high uncharted score (less visited), reward
        # If it leads to a node with a low uncharted score (already well covered), penalize
        if chosen_neighbor is not None and firing_automaton.last_action is not None and firing_automaton.last_action in firing_automaton.action_probs:
            action = firing_automaton.last_action
            # Compare uncharted score of the reached node
            u_score = uncharted_scores.get(chosen_neighbor, 0)

            if u_score > 0.5:  
                # Node is considered uncharted, reward action
                old_prob = firing_automaton.action_probs[action]
                increase = reward_strength * (1 - old_prob)
                firing_automaton.action_probs[action] = old_prob + increase
                # Slightly decrease others
                for a in firing_automaton.action_probs:
                    if a != action:
                        firing_automaton.action_probs[a] *= (1 - reward_strength)
            else:
                # Node is well visited, penalize action
                old_prob = firing_automaton.action_probs[action]
                decrease = penalty_strength * old_prob
                firing_automaton.action_probs[action] = old_prob - decrease
                # Part of penalty: Slightly boost other actions
                other_actions = [a for a in firing_automaton.action_probs if a != action]
                if other_actions:
                    for a in other_actions:
                        # Increase each other action proportionally
                        firing_automaton.action_probs[a] += (penalty_strength * (1 / len(other_actions))) * (1 - firing_automaton.action_probs[a])

            # Normalize probabilities
            firing_automaton.action_probs = normalize_probs(firing_automaton.action_probs)

        # Move fired automaton to Off
        off_set.update(fire_set)
        fire_set = set()

        # Periodically update uncharted scores
        if k % update_interval == 0:
            uncharted_scores = compute_uncharted_scores(S)

        k += 1

    # After K iterations, sort nodes by visitation frequency
    sorted_nodes = sorted(S.items(), key=lambda item: item[1], reverse=True)
    num_sampled_nodes = int(len(all_nodes) * sampling_ratio)
    sampled_nodes = [node for node, count in sorted_nodes[:num_sampled_nodes]]

    # Construct sampled graph edges
    Gs_edges = []
    sampled_set = set(sampled_nodes)
    for node in sampled_nodes:
        for neighbor in outgoing_neighbors.get(node, set()):
            if neighbor in sampled_set:
                Gs_edges.append((node, neighbor))

    return sampled_nodes, Gs_edges