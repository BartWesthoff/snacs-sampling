import random
import numpy as np
import networkx as nx

class Automaton:
    def __init__(self, node_id, neighbors):
        self.node_id = node_id
        self.neighbors = neighbors  
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


def compute_uncharted_scores(S, all_nodes):
    """Compute uncharted scores based on visitation frequencies."""
    visits = np.array(list(S.values()))
    threshold = np.median(visits) if len(visits) > 0 else 0
    raw_scores = {node: max(0, threshold - S[node]) for node in all_nodes} # normalize later
    total = sum(raw_scores.values()) if sum(raw_scores.values()) > 0 else 1
    uncharted_scores = {node: raw_scores[node] / total for node in all_nodes}
    return uncharted_scores

def due_sampling_directed(graph, K=1000, sampling_ratio=0.5, 
                          reward_strength=0.1, penalty_strength=0.1, update_interval=50):
    """
    Decentralized Uncharted Expansion (DUE) sampling for a directed graph.

    Parameters:
    - graph: A NetworkX DiGraph.
    - K: Maximum number of iterations.
    - sampling_ratio: Fraction of nodes to include in the final sampled network.
    - reward_strength: Magnitude of reward updates.
    - penalty_strength: Magnitude of penalty updates.
    - update_interval: Frequency (in iterations) at which uncharted scores are recalculated.

    Returns:
    - sampled_nodes: The list of sampled nodes.
    - Gs_edges: List of directed edges in the sampled subgraph.
    """
    all_nodes = list(graph.nodes())
    outgoing_neighbors = {node: set(graph.successors(node)) for node in all_nodes}

    automata = {}
    for node in all_nodes:
        out_neighbors = outgoing_neighbors.get(node, set())
        automata[node] = Automaton(node_id=node, neighbors=out_neighbors)
    S = {node: 0 for node in all_nodes}  
    passive_set = set(all_nodes)  
    active_set = set()  
    fire_set = set()  
    off_set = set()     

    starting_node = random.choice(list(passive_set))
    automata[starting_node].activity_level = 'Active'
    active_set.add(starting_node)
    passive_set.remove(starting_node)

    uncharted_scores = compute_uncharted_scores(S, all_nodes)

    k = 1
    while k <= K:
        if not active_set:
            # if no active automata, select a new start
            candidates = passive_set if passive_set else all_nodes
            starting_node = random.choice(list(candidates))
            automata[starting_node].activity_level = 'Active'
            active_set.add(starting_node)
            passive_set.discard(starting_node)

        # pick an active automaton to "fire"
        firing_node = random.choice(list(active_set))
        firing_automaton = automata[firing_node]
        active_set.remove(firing_node)
        fire_set = {firing_node}

        # automaton chooses an action (outgoing neighbor)
        if firing_automaton.action_probs:
            neighbors = list(firing_automaton.action_probs.keys())
            probabilities = list(firing_automaton.action_probs.values())
            chosen_neighbor = np.random.choice(neighbors, p=probabilities)
            firing_automaton.last_action = chosen_neighbor
            S[firing_node] += 1
            S[chosen_neighbor] += 1

            for nbr in outgoing_neighbors.get(chosen_neighbor, set()):
                if automata[nbr].activity_level == 'Passive':
                    automata[nbr].activity_level = 'Active'
                    active_set.add(nbr)
                    passive_set.discard(nbr)
        else:
            chosen_neighbor = None

        # determine reward or penalty:
        if chosen_neighbor is not None and firing_automaton.last_action is not None and firing_automaton.last_action in firing_automaton.action_probs:
            action = firing_automaton.last_action
            u_score = uncharted_scores.get(chosen_neighbor, 0)

            if u_score > 0.5:  
                old_prob = firing_automaton.action_probs[action]
                increase = reward_strength * (1 - old_prob)
                firing_automaton.action_probs[action] = old_prob + increase
                for a in firing_automaton.action_probs:
                    if a != action:
                        firing_automaton.action_probs[a] *= (1 - reward_strength)
            else:
                old_prob = firing_automaton.action_probs[action]
                decrease = penalty_strength * old_prob
                firing_automaton.action_probs[action] = old_prob - decrease
                other_actions = [a for a in firing_automaton.action_probs if a != action]
                if other_actions:
                    for a in other_actions:
                        firing_automaton.action_probs[a] += (penalty_strength * (1 / len(other_actions))) * (1 - firing_automaton.action_probs[a])

            firing_automaton.action_probs = normalize_probs(firing_automaton.action_probs) # normalize probabilities

        off_set.update(fire_set)
        fire_set = set()

        if k % update_interval == 0:
            uncharted_scores = compute_uncharted_scores(S, all_nodes)
        k += 1

    sorted_nodes = sorted(S.items(), key=lambda item: item[1], reverse=True)
    num_sampled_nodes = int(len(all_nodes) * sampling_ratio)
    sampled_nodes = [node for node, _ in sorted_nodes[:num_sampled_nodes]]

    Gs_edges = []
    sampled_set = set(sampled_nodes)
    for node in sampled_nodes:
        for neighbor in outgoing_neighbors.get(node, set()):
            if neighbor in sampled_set:
                Gs_edges.append((node, neighbor))
    Gs = nx.DiGraph()
    Gs.add_edges_from(Gs_edges)
    return sampled_nodes, Gs_edges,Gs