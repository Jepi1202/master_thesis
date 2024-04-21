import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors

def display_graph(edges, node_positions, edge_values, minVal = 0, maxVal = 2, display = False, colorbar = False):
    G = nx.Graph()
    G.add_nodes_from(node_positions.keys())
    G.add_edges_from(edges)

    # Choose a colormap, for example, 'viridis'
    cmap = plt.cm.jet

    # Normalize values to range [0, 1] for colormap
    norm = mcolors.Normalize(vmin=minVal, vmax=maxVal)

    # Create a colormap scalar mappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Normalize edge values manually
    normalized_edge_values = [norm(val) for val in edge_values]

    # Draw the graph with colored edges
    nx.draw(G, pos=node_positions, with_labels=True, font_weight='bold', node_size=50, node_color='skyblue', font_color='black', font_size=8, edgelist=edges, edge_color=normalized_edge_values, cmap=cmap, linewidths=2)
    # nx.draw(G,  with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=8, edgelist=edges, edge_color=normalized_edge_values, cmap=cmap, linewidths=2)

    # Add a colorbar to show the mapping of values to colors
    if colorbar:
        plt.colorbar(sm, label='Edge Values')
    
    if display:
        plt.show()