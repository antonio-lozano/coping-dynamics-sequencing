"""Plotting helpers (shared styles and visualization functions)."""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import seaborn as sns
import numpy as np
import networkx as nx

sns.set(style="whitegrid")


def save(fig, path, dpi=300):
    """Save figure to both PNG and PDF."""
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def plot_chord_diagram(matrix, labels, colors, title=None, ax=None, r=1.0, gap=0.05, arc_width=0.06):
    """
    Draw a chord diagram where transitions between behaviors are visualized with arcs.
    
    Parameters
    ----------
    matrix : np.ndarray or pd.DataFrame
        Transition matrix (n x n)
    labels : list
        Labels for each state/behavior
    colors : dict
        Mapping of label -> color
    title : str, optional
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        Axes to draw on
    r : float
        Radius of the chord diagram
    gap : float
        Gap between segments
    arc_width : float
        Width of the outer arcs
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    n = len(labels)
    segment = 2 * np.pi / n
    mid_angles = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw outer arcs (wedges) for each behavior
    for i, label in enumerate(labels):
        mid_angle = (i + 0.5) * segment
        arc_start = mid_angle - (segment / 2 - gap / 2)
        arc_end = mid_angle + (segment / 2 - gap / 2)
        mid_angles.append(mid_angle)

        wedge = Wedge(center=(0, 0),
                      r=r,
                      theta1=np.degrees(arc_start),
                      theta2=np.degrees(arc_end),
                      width=arc_width,
                      facecolor=colors.get(label, "#888888"),
                      edgecolor='none')
        ax.add_patch(wedge)

        # Label positioning
        label_radius = r + arc_width * 1.5
        angle_deg = np.degrees(mid_angle) - 90
        ax.text(label_radius * np.cos(mid_angle),
                label_radius * np.sin(mid_angle),
                label,
                ha='center', va='center',
                fontsize=10,
                rotation=angle_deg if -90 <= angle_deg <= 90 else angle_deg + 180,
                rotation_mode='anchor',
                color="#4d4d4d")

    # Get max absolute weight for scaling (handles negative values for difference matrices)
    if hasattr(matrix, 'values'):
        all_vals = matrix.values.flatten()
    elif hasattr(matrix, 'max'):
        all_vals = np.array([[matrix.iloc[i, j] for j in range(n)] for i in range(n)]).flatten()
    else:
        all_vals = matrix.flatten()
    
    max_abs_weight = np.max(np.abs(all_vals))
    if max_abs_weight == 0:
        max_abs_weight = 1  # Avoid division by zero
        
    endpoint_radius = r - arc_width * 0.9
    delta_angle = 0.12  # Offset for bidirectional edges

    # Draw chord connections
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            if hasattr(matrix, 'iloc'):
                weight_ij = matrix.iloc[i, j]
                weight_ji = matrix.iloc[j, i]
            else:
                weight_ij = matrix[i, j]
                weight_ji = matrix[j, i]
            
            # Skip near-zero weights
            if abs(weight_ij) < 1e-6:
                continue
                
            theta_source = mid_angles[i]
            # Offset destination angle if bidirectional
            if abs(weight_ji) > 1e-6 and i < j:
                theta_dest = mid_angles[j] - delta_angle
            elif abs(weight_ji) > 1e-6 and i >= j:
                theta_dest = mid_angles[j] + delta_angle
            else:
                theta_dest = mid_angles[j]
            
            # Line width based on absolute weight (log scale for better visualization)
            abs_weight = abs(weight_ij)
            lw = (np.log1p(abs_weight) / np.log1p(max_abs_weight)) * 12 + 0.5
            
            x_source = endpoint_radius * np.cos(theta_source)
            y_source = endpoint_radius * np.sin(theta_source)
            x_dest = endpoint_radius * np.cos(theta_dest)
            y_dest = endpoint_radius * np.sin(theta_dest)
            
            # Color: use source behavior color for positive, red-ish for negative (increased/decreased)
            if weight_ij > 0:
                edge_color = colors.get(labels[i], "#888888")
                alpha = 0.7
            else:
                # For negative values (decreased transitions), use a muted blue
                edge_color = "#668AB2"
                alpha = 0.6
            
            # Bezier curve through center
            verts = [(x_source, y_source), (0, 0), (x_dest, y_dest)]
            path = Path(verts, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            patch = PathPatch(path, facecolor='none', 
                              edgecolor=edge_color, 
                              lw=lw, alpha=alpha)
            ax.add_patch(patch)

    ax.set_xlim(-1.4 * r, 1.4 * r)
    ax.set_ylim(-1.4 * r, 1.4 * r)
    
    if title:
        ax.set_title(title, fontsize=12, color="#4d4d4d", pad=15)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    return ax


def plot_difference_network(diff_matrix, labels, colors, title=None, ax=None, figsize=(10, 8)):
    """
    Plot a difference network showing how transitions differ between groups.
    
    Node color and size represent the net effect (sum of incoming differences).
    Edge color represents the direction of change (red = increased in ELS, blue = decreased).
    
    Parameters
    ----------
    diff_matrix : pd.DataFrame or np.ndarray
        Difference matrix (e.g., ELS - Control)
    labels : list
        Labels for each state/behavior
    colors : dict
        Mapping of label -> color (used for reference, network uses diverging colors)
    title : str, optional
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        Axes to draw on
    figsize : tuple
        Figure size if creating new figure
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # Diverging colormap for differences
    cmap_colors = ['#A62C2C', '#BF4146', '#FAF2E6', '#80A1C2', '#668AB2']
    cmap_edge = mcolors.LinearSegmentedColormap.from_list("diff_cmap", cmap_colors)
    
    n = len(labels)
    G = nx.DiGraph()
    G.add_nodes_from(labels)
    
    # Get values as numpy array
    if hasattr(diff_matrix, 'values'):
        values = diff_matrix.values
    else:
        values = diff_matrix
    
    abs_max_edge = np.abs(values).max()
    if abs_max_edge == 0:
        abs_max_edge = 1
    norm_edge = mcolors.Normalize(vmin=-abs_max_edge, vmax=abs_max_edge)
    
    edges, edge_colors, edge_widths = [], [], []
    scale_edge = 6
    
    for i, source in enumerate(labels):
        for j, target in enumerate(labels):
            if i == j:
                continue
            weight = values[i, j]
            if abs(weight) < 1e-6:
                continue
            G.add_edge(source, target, weight=weight)
            edges.append((source, target))
            edge_colors.append(cmap_edge(norm_edge(weight)))
            edge_widths.append(scale_edge * abs(weight) / abs_max_edge + 0.5)
    
    # Node effects (sum of incoming transitions)
    node_effect = {label: values[:, i].sum() for i, label in enumerate(labels)}
    node_effect_values = np.array(list(node_effect.values()))
    node_abs_max = np.abs(node_effect_values).max() or 1.0
    norm_node = mcolors.Normalize(vmin=-node_abs_max, vmax=node_abs_max)
    node_colors_list = [cmap_edge(norm_node(node_effect[node])) for node in labels]
    
    base_size = 800
    additional_size_factor = 2000
    node_sizes = [base_size + additional_size_factor * (abs(node_effect[node]) / node_abs_max)
                  for node in labels]
    
    # Circular layout
    pos = nx.circular_layout(G)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=node_sizes, 
                           ax=ax, edgecolors='#888888', linewidths=0.5)
    
    # Draw curved edges
    for (source, target), color, width in zip(edges, edge_colors, edge_widths):
        source_pos, target_pos = np.array(pos[source]), np.array(pos[target])
        mid_pos = (source_pos + target_pos) / 2
        # Curve control point perpendicular to edge
        perp = np.array([-(target_pos[1] - source_pos[1]), target_pos[0] - source_pos[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-6) * 0.15
        control_pos = mid_pos + perp
        
        path = Path([source_pos, control_pos, target_pos], 
                    [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        patch = PathPatch(path, fill=False, edgecolor=color, linewidth=width, alpha=0.8)
        ax.add_patch(patch)
    
    # Labels outside nodes
    label_pos = {node: (coords[0] * 1.35, coords[1] * 1.35) for node, coords in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=9, font_weight='normal', 
                            font_color='#4D4D4D', ax=ax)
    
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=11, color="#4d4d4d", pad=10)
    
    return ax, cmap_edge, norm_edge


def add_difference_colorbar(fig, cmap, norm, position=[0.85, 0.25, 0.02, 0.5], label="Δ Probability"):
    """
    Add a colorbar for difference network plots.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
    cmap : colormap
    norm : Normalize instance
    position : list
        [left, bottom, width, height] in figure coordinates
    label : str
        Colorbar label
    """
    cax = fig.add_axes(position)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(label, fontsize=9, color="#4d4d4d")
    cbar.ax.tick_params(labelsize=8, colors="#4d4d4d")
    for spine in cax.spines.values():
        spine.set_edgecolor("#4d4d4d")
    return cbar
