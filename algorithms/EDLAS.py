import random
import numpy as np
import networkx as nx


class Automaton:
    def __init__(self, node_id, neighbors):
        self.node_id = node_id
        self.neighbors = neighbors
        self.activity_level = 'Passive'
        self.action_probs = {neighbor: 1 / len(neighbors) for neighbor in neighbors} if neighbors else {}
        self.last_action = None


def edlas_sampling(graph, K=1000, sampling_ratio=0.5):
    """
    Performs eDLA-based sampling on a directed NetworkX graph.

    Parameters:
    - graph: NetworkX DiGraph representing the directed network.
    - K: Maximum number of iterations.
    - sampling_ratio: Fraction of nodes to include in the sampled network.

    Returns:
    - sampled_graph: NetworkX DiGraph containing the sampled network.
    """
    all_nodes = set(graph.nodes())
    outgoing_neighbors = {node: set(graph.successors(node)) for node in all_nodes}
    automata = {}
    for node in all_nodes:
        out_neighbors = outgoing_neighbors.get(node, set())
        automata[node] = Automaton(node_id=node, neighbors=out_neighbors)

    S = {node: 0 for node in all_nodes}  
    Pa = set(all_nodes) 
    Ac = set() 
    Of = set()
    Fi = set()
    S_prev_sum = 0
    k = 1
    vs = random.choice(list(Pa))
    automata[vs].activity_level = 'Active'
    Ac.add(vs)
    Pa.remove(vs)

    while k < K:
        if not Ac:
            if Pa:
                vs = random.choice(list(Pa))
            else:
                vs = random.choice(list(all_nodes))
            automata[vs].activity_level = 'Active'
            Ac.add(vs)
            Pa.discard(vs)

        As_node = random.choice(list(Ac))
        As = automata[As_node]
        Fi = {As_node}
        Ac.remove(As_node)
        if As.action_probs:
            neighbors = list(As.action_probs.keys())
            probabilities = list(As.action_probs.values())
            vm = np.random.choice(neighbors, p=probabilities)
            As.last_action = vm
            S[As_node] += 1
            S[vm] += 1
            N_vm = [n for n in outgoing_neighbors.get(vm, set()) if automata[n].activity_level == 'Passive']
            for neighbor in N_vm:
                automata[neighbor].activity_level = 'Active'
                Ac.add(neighbor)
                Pa.discard(neighbor)
        else:
            vm = None
        if len(Of) >= len(all_nodes) or not Ac:
            current_S_sum = sum(S.values())
            if current_S_sum > S_prev_sum:
                for node in Of:
                    ai = automata[node]
                    if ai.last_action and ai.last_action in ai.action_probs:
                        r = 0.1  # reward parameter
                        ai.action_probs[ai.last_action] += r * (1 - ai.action_probs[ai.last_action])
                        # Decrease other probabilities
                        for action in ai.action_probs:
                            if action != ai.last_action:
                                ai.action_probs[action] -= r * ai.action_probs[action]
            else:
                for node in Of:
                    ai = automata[node]
                    if ai.last_action and ai.last_action in ai.action_probs:
                        s = 0.1  # penalty parameter
                        ai.action_probs[ai.last_action] -= s * ai.action_probs[ai.last_action]
                        num_actions = len(ai.action_probs)
                        for action in ai.action_probs:
                            if action != ai.last_action:
                                ai.action_probs[action] += s * (1 / (num_actions - 1)) * (1 - ai.action_probs[action])
       
            for node in Of:
                ai = automata[node]
                total_prob = sum(ai.action_probs.values())
                if total_prob > 0:
                    ai.action_probs = {action: prob / total_prob for action, prob in ai.action_probs.items()} # normalize probabilities
            Pa = set(all_nodes)
            for node in automata:
                automata[node].activity_level = 'Passive'
            Fi = set()
            Ac = set()
            Of = set()

            vs = random.choice(list(Pa))
            automata[vs].activity_level = 'Active'
            Ac.add(vs)
            Pa.remove(vs)
            S_prev_sum = current_S_sum
        else:
            Of.update(Fi)
            Fi = set()

        k += 1
    sorted_nodes = sorted(S.items(), key=lambda item: item[1], reverse=True)
    num_sampled_nodes = int(len(all_nodes) * sampling_ratio)
    sampled_nodes = [node for node, count in sorted_nodes[:num_sampled_nodes]]
    sampled_graph = nx.DiGraph()
    for node in sampled_nodes:
        for neighbor in outgoing_neighbors.get(node, set()):
            if neighbor in sampled_nodes:
                sampled_graph.add_edge(node, neighbor)

    return sampled_graph
