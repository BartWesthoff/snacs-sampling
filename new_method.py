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


def afe_sampling_directed(all_nodes, outgoing_neighbors, K=1000, sampling_ratio=0.5,
                          reward_strength=0.1, penalty_strength=0.1, update_interval=50):
    """
    Adaptive Frontier Expansion (AFE) sampling for a directed graph.

    Parameters:
    - all_nodes: A set of all node IDs.
    - outgoing_neighbors: Dictionary mapping each node to its set of outgoing neighbors.
    - K: Maximum number of iterations.
    - sampling_ratio: Fraction of nodes to include in the final sampled network.
    - reward_strength: Magnitude of reward updates.
    - penalty_strength: Magnitude of penalty updates.
    - update_interval: Frequency (in iterations) at which frontier scores are recalculated.

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
    Pa = set(all_nodes)  # Passive
    Ac = set()  # Active
    Fi = set()  # Fire
    Of = set()  # Off

    # Select a starting node at random
    vs = random.choice(list(Pa))
    automata[vs].activity_level = 'Active'
    Ac.add(vs)
    Pa.remove(vs)

    def compute_frontier_scores():
        """Compute frontier scores based on visitation frequencies."""
        # Lower visited nodes should have higher frontier scores
        visits = np.array(list(S.values()))
        # For simplicity, let’s define a frontier threshold as the median of visited counts
        threshold = np.median(visits) if len(visits) > 0 else 0
        # Frontier score = max(0, threshold - visit_count), normalized later
        raw_scores = {node: max(0, threshold - S[node]) for node in all_nodes}
        total = sum(raw_scores.values()) if sum(raw_scores.values()) > 0 else 1
        frontier_scores = {node: raw_scores[node]/total for node in all_nodes}
        return frontier_scores

    frontier_scores = compute_frontier_scores()

    k = 1
    while k <= K:
        if not Ac:
            # If no active automata, select a new start
            candidates = Pa if Pa else all_nodes
            vs = random.choice(list(candidates))
            automata[vs].activity_level = 'Active'
            Ac.add(vs)
            Pa.discard(vs)

        # Pick an active automaton to "fire"
        As_node = random.choice(list(Ac))
        As = automata[As_node]
        Ac.remove(As_node)
        Fi = {As_node}

        # Automaton chooses an action (outgoing neighbor) based on current probabilities
        if As.action_probs:
            neighbors = list(As.action_probs.keys())
            probabilities = list(As.action_probs.values())
            vm = np.random.choice(neighbors, p=probabilities)
            As.last_action = vm

            # Update visitation
            S[As_node] += 1
            S[vm] += 1

            # Activate neighbors of vm if they are passive
            for nbr in outgoing_neighbors.get(vm, set()):
                if automata[nbr].activity_level == 'Passive':
                    automata[nbr].activity_level = 'Active'
                    Ac.add(nbr)
                    Pa.discard(nbr)
        else:
            vm = None

        # Determine reward or penalty:
        # If the chosen action leads to a node with a high frontier score (less visited), reward
        # If it leads to a node with a low frontier score (already well covered), penalize
        if vm is not None and As.last_action is not None and As.last_action in As.action_probs:
            action = As.last_action
            # Compare frontier score of the reached node
            f_score = frontier_scores.get(vm, 0)

            if f_score > 0.5:  
                # Node is considered frontier-like, reward action
                old_prob = As.action_probs[action]
                increase = reward_strength * (1 - old_prob)
                As.action_probs[action] = old_prob + increase
                # Slightly decrease others
                for a in As.action_probs:
                    if a != action:
                        As.action_probs[a] *= (1 - reward_strength)
            else:
                # Node is well visited, penalize action
                old_prob = As.action_probs[action]
                decrease = penalty_strength * old_prob
                As.action_probs[action] = old_prob - decrease
                # Part of penalty: Slightly boost other actions
                other_actions = [a for a in As.action_probs if a != action]
                if other_actions:
                    for a in other_actions:
                        # Increase each other action proportionally
                        As.action_probs[a] += (penalty_strength * (1 / len(other_actions))) * (1 - As.action_probs[a])

            # Normalize probabilities
            As.action_probs = normalize_probs(As.action_probs)

        # Move fired automaton to Off
        Of.update(Fi)
        Fi = set()

        # Periodically update frontier scores
        if k % update_interval == 0:
            frontier_scores = compute_frontier_scores()

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


# Load the dataset
df = pd.read_csv('data/soc-sign-bitcoinotc.csv.gz', names=["SOURCE", "TARGET", "RATING", "TIME"])
df['TIME'] = pd.to_datetime(df['TIME'])

# Create the set of all nodes and the dictionary of outgoing neighbors
all_nodes = set(df['SOURCE']).union(set(df['TARGET']))
outgoing_neighbors = {}
for source, target in zip(df['SOURCE'], df['TARGET']):
    if source not in outgoing_neighbors:
        outgoing_neighbors[source] = set()
    outgoing_neighbors[source].add(target)

# Apply AFE sampling
sampled_nodes, sampled_edges = afe_sampling_directed(
    all_nodes=all_nodes,
    outgoing_neighbors=outgoing_neighbors,
    K=1000,  # Number of iterations
    sampling_ratio=0.1,  # 50% of nodes to be sampled
    reward_strength=0.1,  # Reward strength for learning probabilities
    penalty_strength=0.1,  # Penalty strength for learning probabilities
    update_interval=50  # Update interval for frontier scores
)

# Output the results
print("Sampled Nodes:", len(sampled_nodes))
print("Sampled Edges:", len(sampled_edges))

