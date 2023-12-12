# -*- coding: utf-8 -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Provide graph functions for analyzing functional connectomes."""

import csv
from itertools import combinations, combinations_with_replacement, product

import networkx as nx
import numpy as np
import pandas as pd


# with help from
# https://programminghistorian.org/en/lessons/exploring-and-analyzing-network-data-with-python
def gen_base_graph_from_atlas(atlas, atlas_delim=","):
    """Generate a networkx graph containing atlas/parcellation information.

    Parameters
    ----------
    atlas : str or file path
        Atlas file containing at minimum: [index, label]
        for parcellation data. Optionally, include additional
        node attributes (e.g., hemisphere, resolution, color, etc.).
    atlas_delim : str, optional
        Delimiter used to read the `atlas`, by default ",".

    Returns
    -------
    :class:`networkx.Graph`
        A networkx graph containing nodes with labels
        from the `atlas` given. Edges are the
        ((n*n-1)/2) + n combination of node pairs,
        which includes self-loop identity pairs.

    Examples
    --------
    >>> from meierlab.networks import graph
    >>> atlas = 'schaefer2018/atlas.csv'
    >>> G = graph.gen_base_graph_from_atlas(atlas)
    >>> G.nodes()
    NodeView(('LH_Vis_1', 'LH_Vis_2', 'LH_Vis_3',
    'LH_Vis_4', 'LH_Vis_5', 'RH_Vis_1', 'RH_Vis_2',
    'RH_Vis_3', 'RH_Vis_4', 'RH_Vis_5', 'LH_Default_Temp_1',
    'LH_Default_Temp_2', 'LH_Default_Temp_3',
    'LH_Default_Temp_4','LH_Default_Temp_5',
    'LH_Default_Temp_6', 'LH_Default_Temp_7',
    'LH_Default_Temp_8', 'LH_Default_Temp_9',
    'LH_Default_Temp_10','RH_Default_Temp_1',
    'RH_Default_Temp_2', 'RH_Default_Temp_3',
    'RH_Default_Temp_4', 'RH_Default_Temp_5',
    'RH_Default_Temp_6','RH_Default_Temp_7',
    'RH_Default_Temp_8'))
    """
    G = nx.Graph()

    with open(atlas, 'r') as atlas_file:
        atlas_reader = csv.reader(atlas_file, delimiter=atlas_delim)

        # get header
        attributes = next(atlas_reader)

        # read in the rest
        nodes = [n for n in atlas_reader]

    node_names = [n[1] for n in nodes]

    # get all pair combinations, include the identity
    edges = list(combinations_with_replacement(node_names, 2))

    # add to the graph
    G.add_nodes_from(node_names)
    G.add_edges_from(edges)

    # add node attributes
    for idx, attr in enumerate(attributes):
        if attr == "label":
            continue

        attr_dict = {}
        for node in nodes:
            attr_dict[node[1]] = node[idx]

        nx.set_node_attributes(G, attr_dict, attr)

    return G


def gen_graph_from_matrix(
    G, atlas, matrix_file, atlas_delim=",", matrix_delim="\t", to_fz=True
):
    r"""Generate a (weighted, undirected) graph from a matrix file.

    Parameters
    ----------
    atlas : str or file path
        Atlas parcellation used to generate the main graph.
    matrix_file : str or file path
        A text file containing edge weight information (e.g., a correlation matrix).
    atlas_delim : str, optional
        Delimiter used to read the `atlas`, by default ","
    matrix_delim : str, optional
        Delimiter used to read the `matrix_file`, by default "\t"
    to_fz : bool, optional
        Fisher-Z transform the correlations, default True.

    Returns
    -------
    :class:`networkx.Graph`
        A graph with additional edge weight attributes.

    Examples
    --------
    >>> from meierlab.networks import graph
    >>> atlas = 'schaefer2018/atlas.csv'
    >>> sub_file = 'sub-001.tsv'
    >>> G = graph.gen_base_graph_from_atlas(atlas)
    >>> g = graph.gen_graph_from_matrix(G,atlas,sub_file)
    >>> edge_weights = [d[edge_attr] for (_,_,d) in g.edges(data=True)]
    >>> edge_weights[0:10]
    [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    """
    # read in the matrix and atlas (for ROI labels)
    corr_df = pd.read_csv(matrix_file, delimiter=matrix_delim, header=None)
    atlas_df = pd.read_csv(atlas, delimiter=atlas_delim)

    corr_df.columns = atlas_df['label']
    corr_df.index = atlas_df['label']

    # fisher z-score the correlations
    if to_fz:
        fz_df = corr_df.apply(np.arctanh)
    else:
        fz_df = corr_df.copy()

    # zero out negative values
    fz_df[fz_df < 0] = 0

    # assign matrix values ('weights') to each edge/ROI-pair
    weights = {}
    for source, target in G.edges():
        weights[(source, target)] = {"weight": fz_df.loc[source][target]}

    nx.set_edge_attributes(G, weights)

    return G


def gen_basic_metrics(G):
    """Assign some basic graph theory metrics to nodes in a graph.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        Graph to calculate metrics for.

    Returns
    -------
    :class:`networkx.Graph`
        Graph with 'degree', 'betweenness',
        and 'eigenvector' centrality attributes
        added to nodes.

    Examples
    --------
    >>> import networkx as nx
    >>> from meierlab.networks import graph
    >>> atlas = 'schaefer2018/atlas.csv'
    >>> sub_file = 'sub-001.tsv'
    >>> G = graph.gen_base_graph_from_atlas(atlas)
    >>> g = graph.gen_graph_from_matrix(G,atlas,sub_file)
    >>> g = graph.gen_basic_metrics(g)
    >>> eigenvectors = nx.get_node_attributes(g,'eigenvector')
    >>> print(eigenvectors['RH_Default_Temp_8'])
    0.1889822365046136
    """
    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')

    betweenness_dict = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, betweenness_dict, 'betweenness')

    eigenvector_dict = nx.eigenvector_centrality(G)
    nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')

    return G


def gen_subnetwork_list(G, subnetwork_label="RSN"):
    """Generate a list of 'subnetworks' used in an atlas/parcellation.

    Useful for creating subgraphs.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        Graph containing nodes with a `subnetwork_label` attribute.
    subnetwork_label : str, optional
        Node label attribute to search, by default "RSN"

    Returns
    -------
    list
        A list of 'subnetworks' that nodes in `G` belong to.

    Examples
    --------
    >>> from meierlab.networks import graph
    >>> atlas = 'schaefer2018/atlas.csv'
    >>> G = graph.gen_base_graph_from_atlas(atlas)
    >>> rsn_list = graph.gen_subnetwork_list(G)
    >>> print(rsn_list)
    ['DMN','Visual']
    """
    # use the labels from the graph to generate list of (unique) subnetworks
    subnetwork_list = []
    for node in G.nodes():
        subnetwork_list.append(G.nodes[node][subnetwork_label])

    subnetwork_list = sorted(list(set(subnetwork_list)))

    return subnetwork_list


def gen_subnetwork_subgraphs(G, subnetwork_list, subnetwork_label="RSN"):
    """Generate 'subgraphs' according to the given 'subnetworks'.

    These subgraphs can be used to explore
    individual networks (e.g., resting-state networks)
    within a system.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        A graph containing nodes with a `subnetwork_label` attribute.
    subnetwork_list : list
        A list of subnetworks that nodes in `G` belong to.
    subnetwork_label : str, optional
        Node label attribute to search, by default "RSN".

    Returns
    -------
    list
        A list of subgraphs: :class:`networkx.Graph`
        with each subgraph corresponding to nodes in
        each network of `subnetwork_list`.

    Examples
    --------
    >>> from meierlab.networks import graph
    >>> atlas = 'schaefer2018/atlas.csv'
    >>> sub_file = 'sub-001.tsv'
    >>> G = graph.gen_base_graph_from_atlas(atlas)
    >>> g = graph.gen_graph_from_matrix(G,atlas,sub_file)
    >>> rsn_list = graph.gen_subnetwork_list(g)
    >>> subgraphs = graph.gen_subnetwork_subgraphs(g, rsn_list)
    >>> subgraphs[0].nodes()
    NodeView(('LH_Default_Temp_1', 'LH_Default_Temp_2',
    'LH_Default_Temp_3', 'LH_Default_Temp_4',
    'LH_Default_Temp_5', 'LH_Default_Temp_6',
    'LH_Default_Temp_7', 'LH_Default_Temp_8',
    'LH_Default_Temp_9', 'LH_Default_Temp_10',
    'RH_Default_Temp_1', 'RH_Default_Temp_2',
    'RH_Default_Temp_3', 'RH_Default_Temp_4',
    'RH_Default_Temp_5', 'RH_Default_Temp_6',
    'RH_Default_Temp_7', 'RH_Default_Temp_8'))
    """
    # for each subnetwork, add node to subgraph if it belongs to the subnetwork
    subgraphs = []
    for subnetwork in subnetwork_list:
        node_list = []
        for node in G.nodes():
            if G.nodes[node][subnetwork_label] == subnetwork:
                node_list.append(node)
        subgraph = G.subgraph(node_list)
        subgraphs.append(subgraph)

    return subgraphs


def gen_subnetwork_pairs(G, subgraph_list, subnetwork_label="RSN"):
    """Generate pairs of networks from graph `G` to explore and compare.

    Parameters
    ----------
    subgraph_list : list
        A list of subnetwork subgraphs.
    subnetwork_label : str, optional
        Node label attribute to search, by default "RSN".

    Returns
    -------
    dict
        A dictionary with keys as pairs of subnetworks
        (e.g., DMN-VIS) and values as a subgraph with
        nodes where edges connect nodes from both subnetworks.

    Examples
    --------
    >>> from meierlab.networks import graph
    >>> atlas = 'schaefer2018/atlas.csv'
    >>> sub_file = 'sub-001.tsv'
    >>> G = graph.gen_base_graph_from_atlas(atlas)
    >>> g = graph.gen_graph_from_matrix(G,atlas,sub_file)
    >>> rsn_list = graph.gen_subnetwork_list(g)
    >>> subgraphs = graph.gen_subnetwork_subgraphs(g, rsn_list)
    >>> rsn_pairs = graph.gen_subnetwork_pairs(g, subgraphs)
    >>> rsn_pairs
    {('DMN', 'Visual'): <networkx.classes.graph.Graph at 0x120960460>}
    """
    # get all pairs of subnetworks (excluding identity)
    network_pairs = list(combinations(subgraph_list, 2))

    # join subgraph pairs and label according to subnetwork
    paired_subgraphs = {}
    for net_1, net_2 in network_pairs:
        nodes_1 = list(net_1.nodes())
        nodes_2 = list(net_2.nodes())
        edge_list = list(product(nodes_1, nodes_2))

        combined = G.edge_subgraph(edge_list)
        label_1 = list(nx.get_node_attributes(net_1, subnetwork_label).values())[0]
        label_2 = list(nx.get_node_attributes(net_2, subnetwork_label).values())[0]
        paired_subgraphs[(label_1, label_2)] = combined

    return paired_subgraphs


def get_within_network_connectivity(subgraph, edge_attr="weight"):
    """Calculate the average weight of edges in a subgraph.

    This provides the effective 'within-network' connectivity.

    Parameters
    ----------
    subgraph : :class:`networkx.Graph.subgraph`
        A subgraph with edge attributes: `edge_attr` to average.
    edge_attr : str, optional
        Edge attribute label, by default "weight".

    Returns
    -------
    :class:`numpy.float64`
        Average of weighted edges in `subgraph`.

    Examples
    --------
    >>> from meierlab.networks import graph
    >>> atlas = 'schaefer2018/atlas.csv'
    >>> sub_file = 'sub-001.tsv'
    >>> G = graph.gen_base_graph_from_atlas(atlas)
    >>> g = graph.gen_graph_from_matrix(G,atlas,sub_file)
    >>> rsn_list = graph.gen_subnetwork_list(g)
    >>> subgraphs = graph.gen_subnetwork_subgraphs(g, rsn_list)
    >>> avg = graph.get_within_network_connectivity(subgraphs[0])
    >>> avg
    0.39999999999999997
    """
    # just need the weight values, exclude nans in the mean calculation
    sg = subgraph.copy()
    sg.remove_edges_from(nx.selfloop_edges(sg))
    edge_weights = [d[edge_attr] for (_, _, d) in sg.edges(data=True)]
    avg = np.nanmean(edge_weights)

    return avg


def get_between_network_connectivity(subgraphs, edge_attr="weight"):
    """Calculate the average of weighted edges of each subgraph pair.

    This provides the effective 'between-network' connectivity of
    each subgraph pair.

    Parameters
    ----------
    subgraphs : dict
        Dictionary of subgraph network pairs and their connecting nodes.
    edge_attr : str, optional
        Edge attribute label, by default "weight".

    Returns
    -------
    dict
        Dictionary of network pairs and their average connectivity.

    Examples
    --------
    >>> from meierlab.networks import graph
    >>> atlas = 'schaefer2018/atlas.csv'
    >>> sub_file = 'sub-001.tsv'
    >>> G = graph.gen_base_graph_from_atlas(atlas)
    >>> g = graph.gen_graph_from_matrix(G,atlas,sub_file)
    >>> rsn_list = graph.gen_subnetwork_list(g)
    >>> subgraphs = graph.gen_subnetwork_subgraphs(g, rsn_list)
    >>> rsn_pairs = graph.gen_subnetwork_pairs(g, subgraphs)
    >>> averages = graph.get_between_network_connectivity(rsn_pairs)
    >>> averages
    {('DMN', 'Visual'): 0.5}
    """
    # like within_connectivity, but across all pairs of networks
    averages = {}
    for net_pair, subgraph in subgraphs.items():
        sg = subgraph.copy()
        sg.remove_edges_from(nx.selfloop_edges(sg))
        edge_weights = [d[edge_attr] for (_, _, d) in sg.edges(data=True)]
        avg = np.nanmean(edge_weights)
        averages[net_pair] = avg

        # ensure we get averages only from valid connections
        averages[f"{net_pair}_nodes"] = len(
            [
                d[edge_attr]
                for (_, _, d) in sg.edges(data=True)
                if not np.isnan(d[edge_attr])
            ]
        )

    return averages


def get_rsn_connectivity_to_all(rsn_list, rsn_pairs, edge_attr="weight"):
    """Get RSN connectivity.

    For each RSN in a list and the corresponding RSN pairs, calculate:
    - It's average connectivity to each other network given.
    - The number of valid nodes that connect the RSN to it's pair.
    This helps generate an understanding of how connected a particular
    RSN is to each RSN in the parcellation, taking into account that
    not all nodes in an RSN have a valid connection to each other node
    in another RSN.

    Parameters
    ----------
    rsn_list : list
        List of RSNs of interest.
    rsn_pairs : dict
        Dictionary of pairs containing the RSN(s) of interest and their related RSNs.
    edge_attr : str, optional
        Edge attribute label, default "weight".

    Examples
    --------
    >>> from meierlab.networks import graph
    >>> atlas = 'schaefer2018/atlas.csv'
    >>> sub_file = 'sub-001.tsv'
    >>> G = graph.gen_base_graph_from_atlas(atlas)
    >>> g = graph.gen_graph_from_matrix(G,atlas,sub_file)
    >>> rsn_list = graph.gen_subnetwork_list(g)
    >>> subgraphs = graph.gen_subnetwork_subgraphs(g, rsn_list)
    >>> rsn_pairs = graph.gen_subnetwork_pairs(g, subgraphs)
    >>> rsns_to_all = graph.get_rsn_connectivity_to_all(rsn_list,rsn_pairs)
    >>> rsns_to_all
    {'DMN_to_all': 0.5493061443340548, 'DMN_to_all_nodes': 180}
    """
    rsn_to_all = {}
    for idx, rsn in enumerate(rsn_list):
        sum_nodes = 0
        sum_weights = 0
        pair_list = [pair for pair in rsn_pairs if rsn in pair and "_nodes" not in pair]

        for pair in pair_list:
            sum_nodes += len(
                [
                    d[edge_attr]
                    for (_, _, d) in rsn_pairs[pair].edges(data=True)
                    if not np.isnan(d[edge_attr])
                ]
            )
            weights = [
                d[edge_attr]
                for (_, _, d) in rsn_pairs[pair].edges(data=True)
                if not np.isnan(d[edge_attr])
            ]
            sum_weights += np.sum(weights)

        avg = sum_weights / sum_nodes
        rsn_to_all[f"{rsn}_to_all"] = avg
        rsn_to_all[f"{rsn}_to_all_nodes"] = sum_nodes

        return rsn_to_all
