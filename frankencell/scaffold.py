import networkx as nx
from collections.abc import Iterable
from mira.plots.base import map_colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dynclipy.dataset import add_adder, Dataset
from .utils import write_cell_info
import dynclipy as dyn

add_adder(Dataset, "add_prior_information")
add_adder(Dataset, "add_cell_waypoints")

def check_definition(G):
    
    assert isinstance(G, nx.DiGraph)
    
    for node in G.nodes:
        assert 'mixing_weights' in G.nodes[node]
        assert type(node) == str
        
    for edge in G.edges:
        if 'weight' in G.edges[edge]:
            assert G.edges[edge]['weight'] >= 0
            

def _sample_path(G, root):
    
    def get_weight(currnode, child):
        edge = G.edges[(currnode, child)]
        
        if 'weight' in edge:
            return edge['weight']
        else:
            return 1.
        
    def markov_step(currnode, visited_nodes, path = []):
        
        visited_nodes.append(currnode)
        path.append(currnode)
        
        if len(G.adj[currnode]) == 0:
            return path
        
        transition_weights = np.array([
            get_weight(currnode, nextnode)
            for nextnode in G.adj[currnode]
        ])
        
        transition_probs = transition_weights/transition_weights.sum()
        
        node_idx = np.random.choice(len(transition_probs), p = transition_probs)
        nextnode = list(G.adj[currnode])[node_idx]
        
        if nextnode in visited_nodes:
            return path
        
        return markov_step(nextnode, visited_nodes, path = path)
        
    return markov_step(root, [])


def cartesian_projection(weights):
    
    angle_vec = np.linspace(0, 2 * np.pi, weights.shape[-1], endpoint=False)
    angle_vec_sin = np.cos(angle_vec)
    angle_vec_cos = np.sin(angle_vec)

    x = np.sum(weights * angle_vec_sin, axis=1)
    y = np.sum(weights * angle_vec_cos, axis=1)
    return x, y


def draw_graph(G, ax = None, edge_attr = 'weight',
     **graph_kwargs):

    default_kwargs = {
        'arrows' : True,
        'with_labels' : True,
        'node_color' : 'white',
    }

    default_kwargs.update(graph_kwargs)
    
    for node in G.nodes:
        assert 'mixing_weights' in G.nodes[node]
        
    node_names, mixing_weights = list(zip(*[
       ( n, G.nodes[n]['mixing_weights']) for n in G.nodes
    ]))
    
    assert all([len(mixing_weights[i]) == len(mixing_weights[0]) 
                for i in range(1, len(mixing_weights))])
    
    weights = np.array(mixing_weights)
        
    x,y = cartesian_projection(weights)
    
    pos_dict = {node_name : (x, y) for node_name, x, y in zip(node_names, x, y)}
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
    
    nx.draw_networkx(
            G, pos = pos_dict,
            ax = ax, 
            width = list(nx.get_edge_attributes(G, edge_attr).values()) ,
            **default_kwargs,
    )
    ax.axis('off')
    
    return ax
    
def _sigmoid(x):
    return 1/(1+np.exp(-x))

def fill_scaffold(
    G, root_node,
    n_cells = 1000,
    ptime_alpha = 1.,
    ptime_beta = 1.,
    sigmoid_approach = True,
    sigmoid_aggression = 7,
    seed = None):
    
    np.random.seed(seed)
    
    min_sigmoid, max_sigmoid = _sigmoid(-0.5 * sigmoid_aggression), _sigmoid(0.5 * sigmoid_aggression)
    
    def get_edgelen(G, edge):
        if 'length' in G.edges[edge]:
            return G.edges[edge]['length']
        else:
            return 1.
    
    cells = []
    for _ in range(n_cells):
        
        pseudotime = np.random.beta(ptime_alpha, ptime_beta)
        path = _sample_path(G, root_node)
                        
        edges = list(zip(path[:-1], path[1:]))
        edge_lengths = np.array([get_edgelen(G, edge) for edge in edges])
        edge_time = np.concatenate([[0],np.cumsum(edge_lengths/edge_lengths.sum())])

        ptime = np.random.beta(ptime_alpha, ptime_beta)

        time_idx = np.argmax(ptime < edge_time)
        edge = edges[time_idx - 1]

        start_time, end_time = edge_time[time_idx-1], edge_time[time_idx]
        progress = (ptime - start_time)/(end_time - start_time)

        x = sigmoid_aggression*(progress - 0.5)
        sigmoid_progress = (_sigmoid(x) - min_sigmoid)/(max_sigmoid - min_sigmoid)

        mixing_weight = np.array(G.nodes[edge[0]]['mixing_weights']) * (1-sigmoid_progress) + \
            np.array(G.nodes[edge[1]]['mixing_weights'])*sigmoid_progress

        cells.append((ptime, sigmoid_progress, mixing_weight, tuple(edge)))

    random_order = np.random.permutation(len(cells))
    cells = [cells[j] for j in random_order]
    
    return {
        col : np.array(vals)
        for col, vals in 
            zip(['pseudotime','progress','mixing_weights','edge'], 
               list(zip(*cells)))
    }



def draw_cells(G, cell_info, jitter = 0.01, ax = None,
              palette = 'Set2', graph_kwargs = {},
              scatter_kwargs = {}, color = 'edge',
              plot_graph = True, vmax = None, vmin = None,
              na_color = 'lightgrey', add_legend = False):
    
    cell_x, cell_y = cartesian_projection(
        np.array(cell_info['mixing_weights']))
    
    np.random.seed(0)
    def get_jitter():
        return (np.random.rand(len(cell_x)) - 0.5)*jitter
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
    
    try:
        if color == 'edge':
            cell_color_attr = [e[1] for e in cell_info[color]]
        else:
            cell_color_attr = list(cell_info[color])
    
        cell_colors = map_colors(ax, cell_color_attr, palette = palette,
                                add_legend = add_legend, vmin = vmin, vmax = vmax,
                                na_color = na_color)
    except KeyError:
        cell_colors = color,

    ax.scatter(
        x = cell_x + get_jitter(), y = cell_y + get_jitter(),
        c = cell_colors,
        **scatter_kwargs
    )
    
    if plot_graph:
        draw_graph(G, ax, **graph_kwargs)

    return ax


def append_cell_info(G, *cell_infos):

    #confirm that all cell infos have the same columns
    info_cols = [tuple(sorted(info.keys())) for info in cell_infos]
    assert all([cols == info_cols[0] for cols in info_cols])

    appended_info = {}

    for col in info_cols[0]:
        appended_info[col] = \
            [val if not col == 'edge' else tuple(val) for info in cell_infos for val in info[col]]


    assert all(
        edge in G.edges
        for edge in appended_info['edge']
    )

    return appended_info


def format_dynverse_dataset(
    G, cell_info, root_node, output_path,
):

    edge = cell_info.pop('edge')
    mixing_weights = np.array(cell_info.pop('mixing_weights'))
    progress = cell_info['progress']
    pseudotime = cell_info['pseudotime']
    cell_ids = np.arange(len(edge)).astype(str)

    edge = np.array([x[1] for x in edge])
    
    branch_progressions = pd.DataFrame(
        {
            'cell_id' : cell_ids, 
            'branch_id' : edge.astype(str),
            'percentage' : progress
        }
    )
    
    branch_names = np.unique(edge)
    
    branches_df = pd.DataFrame(
        {'branch_id' : branch_names.astype(str), 
        'directed' : [True]*len(branch_names), 
        'length' : [1]*len(branch_names)}
    )

    edges = np.array(list(nx.dfs_edges(G, source = root_node))[1:]).astype(str)
    branch_network = pd.DataFrame(
        [e for e in edges if e[1] in branch_names],
        columns = ['from', 'to']
    )

    start_cell = cell_ids[np.argmin(pseudotime)]
    
    end_cells = []
    for branch in branch_names:
        if G.out_degree(branch) == 0:
            branch_cells = branch_progressions.set_index('branch_id').loc[branch]
            end_cells.append(
                branch_cells.iloc[branch_cells.percentage.argmax()].cell_id
            )

    is_start_cell = start_cell == cell_ids
    if len(end_cells) == 1:
        is_end_cell = end_cells[0] == cell_ids
    else:
        is_end_cell = np.isin(cell_ids, end_cells)


    trajectory = dyn.wrap_data(cell_ids = cell_ids)\
            .add_branch_trajectory(
                branch_network = branch_network,
                branches = branches_df,
                branch_progressions = branch_progressions,
            ).add_prior_information(
                start_id = [start_cell],
                end_id = end_cells
            ).add_root([start_cell])\
            .add_cell_waypoints()\
            #.add_expression(counts, counts)

    trajectory.write_output(output_path)

    write_cell_info(output_path, {
        **{'mix_weight_' + str(i) : mixing_weights[:, i] for i in range(mixing_weights.shape[1])},
        'is_end_cell' : is_end_cell,
        'is_start_cell' : is_start_cell,
        'edge' : edge,
        **{k : np.array(v) for k, v in cell_info.items()},
    })