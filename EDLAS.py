import random
import numpy as np
class Automaton:
    def __init__(self, node_id, neighbors):
        self.node_id = node_id
        self.neighbors = neighbors  # Set of neighbor nodes
        if neighbors:
            self.action_probs = {neighbor: 1 / len(neighbors) for neighbor in neighbors}
        else:
            self.action_probs = {}
        self.last_action = None
        self.activity_level = 'Passive'  # 'Passive', 'Active', 'Fire', 'Off'


def edlas_sampling_directed(all_nodes, outgoing_neighbors, K=1000, sampling_ratio=0.5):
    """
    Performs eDLA-based sampling on a directed network.

    Parameters:
    - all_nodes: Set of all node IDs in the network.
    - incoming_neighbors: Dictionary mapping each node to its set of incoming neighbors.
    - outgoing_neighbors: Dictionary mapping each node to its set of outgoing neighbors.
    - K: Maximum number of iterations.
    - sampling_ratio: Fraction of nodes to include in the sampled network.

    Returns:
    - sampled_nodes: List of nodes in the sampled network.
    - Gs_edges: List of directed edges in the sampled network.
    """
    # Initialize automata
    automata = {}
    for node in all_nodes:
        out_neighbors = outgoing_neighbors.get(node, set())
        automata[node] = Automaton(node_id=node, neighbors=out_neighbors)
    
    # Initialize variables
    S = {node: 0 for node in all_nodes}  # Visited counts
    Pa = set(all_nodes)  # Passive automata
    Ac = set()  # Active automata
    Of = set()  # Off automata
    Fi = set()  # Fire automata
    S_prev_sum = 0
    k = 1
    
    # Select a starting node vs at random
    vs = random.choice(list(Pa))
    automata[vs].activity_level = 'Active'
    Ac.add(vs)
    Pa.remove(vs)
    
    while k < K:
        if not Ac:
            # If Active automata set is empty, select a new starting node
            if Pa:
                vs = random.choice(list(Pa))
            else:
                vs = random.choice(list(all_nodes))
            automata[vs].activity_level = 'Active'
            Ac.add(vs)
            Pa.discard(vs)
    
        # Select an automaton As from Ac
        As_node = random.choice(list(Ac))
        As = automata[As_node]
    
        # Set Fi to As and remove from Ac
        Fi = {As_node}
        Ac.remove(As_node)
    
        # Automaton As chooses an action
        if As.action_probs:
            neighbors = list(As.action_probs.keys())
            probabilities = list(As.action_probs.values())
            vm = np.random.choice(neighbors, p=probabilities)
            As.last_action = vm
    
            # Update visited counts
            S[As_node] += 1
            S[vm] += 1
    
            # Update Ac and Pa
            N_vm = [n for n in outgoing_neighbors.get(vm, set()) if automata[n].activity_level == 'Passive']
            for neighbor in N_vm:
                automata[neighbor].activity_level = 'Active'
                Ac.add(neighbor)
                Pa.discard(neighbor)
        else:
            vm = None
    
        # Check conditions to reward or penalize
        if len(Of) >= len(all_nodes) or not Ac:
            current_S_sum = sum(S.values())
            if current_S_sum > S_prev_sum:
                # Favorable traverse, reward the actions
                for node in Of:
                    ai = automata[node]
                    if ai.last_action and ai.last_action in ai.action_probs:
                        # Reward the last action
                        r = 0.1  # Reward parameter
                        ai.action_probs[ai.last_action] += r * (1 - ai.action_probs[ai.last_action])
                        # Decrease other probabilities
                        for action in ai.action_probs:
                            if action != ai.last_action:
                                ai.action_probs[action] -= r * ai.action_probs[action]
            else:
                # Penalize the actions
                for node in Of:
                    ai = automata[node]
                    if ai.last_action and ai.last_action in ai.action_probs:
                        # Penalize the last action
                        s = 0.1  # Penalty parameter
                        ai.action_probs[ai.last_action] -= s * ai.action_probs[ai.last_action]
                        # Increase other probabilities
                        num_actions = len(ai.action_probs)
                        for action in ai.action_probs:
                            if action != ai.last_action:
                                ai.action_probs[action] += s * (1 / (num_actions - 1)) * (1 - ai.action_probs[action])
            # Normalize probabilities
            for node in Of:
                ai = automata[node]
                total_prob = sum(ai.action_probs.values())
                if total_prob > 0:
                    ai.action_probs = {action: prob / total_prob for action, prob in ai.action_probs.items()}
    
            # Reset sets
            Pa = set(all_nodes)
            for node in automata:
                automata[node].activity_level = 'Passive'
            Fi = set()
            Ac = set()
            Of = set()
    
            # Select a new starting node
            vs = random.choice(list(Pa))
            automata[vs].activity_level = 'Active'
            Ac.add(vs)
            Pa.remove(vs)
            S_prev_sum = current_S_sum
        else:
            Of.update(Fi)
            Fi = set()
    
        k += 1

    # Sort the visited nodes
    sorted_nodes = sorted(S.items(), key=lambda item: item[1], reverse=True)
    num_sampled_nodes = int(len(all_nodes) * sampling_ratio)
    sampled_nodes = [node for node, count in sorted_nodes[:num_sampled_nodes]]
    
    # Construct the sampled network Gs
    Gs_edges = []
    for node in sampled_nodes:
        for neighbor in outgoing_neighbors.get(node, set()):
            if neighbor in sampled_nodes:
                Gs_edges.append((node, neighbor))
    
    return sampled_nodes, Gs_edges
