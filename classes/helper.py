import networkx as nx
class Helper:

    def __init__(self):
        pass

    @staticmethod
    def make_snapshot_nodes(G, percentage):
        """
        Creates a single snapshot of the graph containing the specified percentage of nodes.

        Args:
            G (nx.Graph or nx.DiGraph): The input graph.
            percentage (float): The percentage of nodes to include in the snapshot (e.g., 0.2 for 20%).

        Returns:
            nx.Graph or nx.DiGraph: A graph snapshot containing the specified percentage of nodes.
        """
        if not (0 < percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1 (exclusive).")
        nodes = list(G.nodes)
        num_nodes = int(len(nodes) * percentage)
        snapshot_nodes = nodes[:num_nodes]
        snapshot_graph = G.subgraph(snapshot_nodes)
        return snapshot_graph.copy()
    @staticmethod   
    def make_snapshots_nodes(G, N):
        """
        Creates snapshots of the graph based on the number of nodes.

        Args:
            G (nx.Graph or nx.DiGraph): The input graph.
            N (int): The number of snapshots to create.

        Returns:
            list of nx.Graph or nx.DiGraph: A list of graph snapshots.
        """
        G = nx.DiGraph(sorted(G.edges(data=True), key=lambda x: x[2]['time']))
        nodes = list(G.nodes)
        chunk_size = len(nodes) // N 
        leftover = len(nodes) % N  
        snapshots = []
        snapshot_nodes = []
        for i in range(N):
            snapshot_nodes_local = nodes[i * chunk_size: (i + 1) * chunk_size]
            snapshot_nodes = snapshot_nodes_local + snapshot_nodes
            if i < leftover:
                snapshot_nodes.append(nodes[N * chunk_size + i])
            snapshot_graph = G.subgraph(snapshot_nodes)
            snapshots.append(snapshot_graph.copy())
        return snapshots
    
    @staticmethod
    def make_snapshots_edges(G, N):
        edges = list(G.edges(data=True))
        chunk_size = len(edges) // N
        print(f'chunk size {chunk_size}')
        edge_chunks = [edges[i * chunk_size: (i + 1) * chunk_size] for i in range(N)]
        leftover = len(edges) % N
        for i in range(leftover):
            edge_chunks[i].append(edges[N * chunk_size + i])
        snapshots = []
        snapshot_graph = nx.DiGraph()
        for chunk in edge_chunks:
            snapshot_graph.add_edges_from(chunk)
            snapshots.append(snapshot_graph.copy())
        return snapshots