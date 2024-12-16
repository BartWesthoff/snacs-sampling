import random
from collections import deque
import networkx as nx

def icla_ns_directed(graph, f, tau):
    """
    Implements the ICLA-NS algorithm for directed graphs.

    Parameters:
        graph (nx.DiGraph): The directed NetworkX graph.
        f (float): Fraction of nodes to sample.
        tau (float): Convergence threshold.

    Returns:
        Gs (nx.DiGraph): The sampled subgraph.
    """
    num_nodes = graph.number_of_nodes()
    num_sampled_nodes = int(f * num_nodes)
    sampled_nodes = set(random.sample(list(graph.nodes), num_sampled_nodes))
    q = deque(sampled_nodes)
    automata = {node: list(set(graph.successors(node)).union({node})) for node in graph.nodes}
    state = {node: random.choice(automata[node]) for node in sampled_nodes}

    while q:
        vi = q.popleft()
        automaton_vi = automata[vi]
        selected_action = state[vi]  
        if selected_action in automaton_vi: 
            if selected_action in sampled_nodes:
                pass
            else:
                state[vi] = random.choice(automaton_vi)
        else:
            state[vi] = random.choice(automaton_vi)

        if selected_action not in sampled_nodes:
            sampled_nodes.add(selected_action)
            state[selected_action] = random.choice(automata[selected_action])
            q.append(selected_action)

        if len(sampled_nodes) / num_nodes >= tau:
            break
    Gs = nx.DiGraph()
    for u, v in graph.edges:
        if u in sampled_nodes and v in sampled_nodes:
            Gs.add_edge(u, v)
    return Gs