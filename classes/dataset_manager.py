import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class DatasetManager:
    def __init__(self, folder_path, file_extension='csv'):
        """
        Initialize the DatasetManager with a folder path and file extension.
        
        Parameters:
            folder_path (str): Path to the folder containing datasets.
            file_extension (str): File extension of datasets (default: 'csv').
        """
        self.folder_path = folder_path
        self.file_extension = file_extension
        self.graphs = {} 

    def load_edgelists(self, source_col='source', target_col='target', weight_col=None, timestamp_col=None):
        """
        Load all datasets with the specified file extension as edgelists.
        
        Parameters:
            source_col (str): Column name for the source node.
            target_col (str): Column name for the target node.
            weight_col (str): Column name for the edge weights (optional).
            timestamp_col (str): Column name for edge timestamps (optional).
        """
        file_list = [f for f in os.listdir(self.folder_path) if f.endswith(self.file_extension)]
        for file in file_list:
            file_path = os.path.join(self.folder_path, file)
            df = pd.read_csv(file_path)
            if weight_col and timestamp_col:
                G = nx.from_pandas_edgelist(df, source=source_col, target=target_col, edge_attr=[weight_col, timestamp_col], create_using=nx.DiGraph())
            elif weight_col:
                G = nx.from_pandas_edgelist(df, source=source_col, target=target_col, edge_attr=weight_col, create_using=nx.DiGraph())
            else:
                G = nx.from_pandas_edgelist(df, source=source_col, target=target_col, create_using=nx.DiGraph())
            self.graphs[file] = G
        print(f"Loaded {len(self.graphs)} edgelists as graphs.")

    def preview_graphs(self):
        """
        Print basic information about each graph.
        """
        for name, G in self.graphs.items():
            print(f"Graph: {name}")
            print(f"Number of nodes: {G.number_of_nodes()}")
            print(f"Number of edges: {G.number_of_edges()}\n")

    def plot_graph(self, graph_name, save=False):
        """
        Plot a specific graph.
        
        Parameters:
            graph_name (str): Name of the graph.
            save (bool): Whether to save the plot (default: False).
        """
        if graph_name not in self.graphs:
            print(f"Graph '{graph_name}' not found.")
            return
        G = self.graphs[graph_name]
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)  # Layout for visualization
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
        plt.title(f"{graph_name} Visualization")
        if save:
            plt.savefig(f"{graph_name}_plot.png")
        plt.show()

    def plot_all_graphs(self, save=False):
        """
        Plot all graphs.
        
        Parameters:
            save (bool): Whether to save the plots (default: False).
        """
        for name in self.graphs.keys():
            self.plot_graph(name, save=save)

# Usage Example
# Initialize the class
# manager = DatasetManager(folder_path='data')

# # Load datasets as edgelists
# manager.load_edgelists(source_col='SOURCE', target_col='TARGET', timestamp_col='TIMESTAMP')


# # Preview graph information
# manager.preview_graphs()

# Plot specific graph
# manager.plot_graph('example.csv', save=True)

# Plot all graphs
# manager.plot_all_graphs(save=True)