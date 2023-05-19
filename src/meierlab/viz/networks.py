# -*- coding: utf-8 -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import networkx as nx
from nilearn import plotting


def get_atlas_coords(atlas_img):
    """Helper function to generate atlas coordinates if none are provided.

    Parameters
    ----------
    atlas_img : str or file path
        Path to Nifti atlas/parcellation image.

    Returns
    -------
    numpy.ndarray
        Atlas coordinates.
    """
    return plotting.find_parcellation_cut_coords(atlas_img)


def get_subgraph_coords(subgraph, atlas_coords):
    """Helper function to get plotting coordinates for a `subgraph` from the full atlas.

    Parameters
    ----------
    subgraph : :class:`networkx.Graph.subgraph`
        Subgraph with 'index' values attributed to nodes.
    atlas_coords : numpy.ndarray
        Coordinate values for an atlas/parcellation.

    Returns
    -------
    numpy.ndarray
        Atlas coordinates for the given `subgraph`.
    """
    idx_list = list(nx.get_node_attributes(subgraph,"index").values())
    coords = atlas_coords[[int(idx) for idx in idx_list]]

    return coords


def plot_connectome_from_graph(g, atlas_coords, threshold="80%"):
    """Plot full connectome from a graph `g`.

    Parameters
    ----------
    g : :class:`networkx.Graph`
        Network graph to plot.
    atlas_coords : numpy.ndarray
        Nifti file of atlas/parcellation.
    threshold : str, optional
        Edge threshold for visualization, by default "80%".

    Returns
    -------
    view : :class:`nilearn.plotting.html_connectome.ConnectomeView`
        Nilearn connectome viewer. Save as an html page or view using 'open_in_browser' method.
    """
    
    mtx = nx.convert_matrix.to_numpy_array(g)
    view = plotting.plot_connectome(mtx,atlas_coords,threshold)

    return view


def plot_subgraph_connectome(subgraph, atlas_coords, threshold="80%"):
    """Plot the connectome for a `subgraph`, e.g. plot only one subnetwork.

    Parameters
    ----------
    subgraph : :class:`networkx.Graph.subgraph`
        Subgraph to plot.
    atlas_coords : numpy.ndarray
        Atlas coordinates.
    threshold : str, optional
        Edge threshold for viewing, by default "80%".

    Returns
    -------
    view : :class:`nilearn.plotting.html_connectome.ConnectomeView`
        Nilearn connectome viewer with the subgraph. Save as an html page or view using 'open_in_browser' method.
    """
    coords = get_subgraph_coords(subgraph, atlas_coords)
    mtx = nx.convert_matrix.to_numpy_array(subgraph)
    view = plotting.view_connectome(mtx,coords,threshold)

    return view


def plot_subgraph_nodes(subgraph, atlas_coords, size=10):
    """Plot the nodes of a `subgraph`.

    Parameters
    ----------
    subgraph : :class:`networkx.Graph.subgraph`
        Subgraph with nodes containing label and color information.
    atlas_coords : numpy.ndarray
        Atlas coordinates.
    size : int, optional
        Node marker size, by default 10.

    Returns
    -------
    view : :class:`nilearn.plotting.html_connectome.ConnectomeView`
        Nilearn connectome viewer with the nodes from the subgraph. Save as an html page of view using 'open_in_browser' method.
    """
    coords = get_subgraph_coords(subgraph, atlas_coords)
    colors = list(nx.get_node_attributes(subgraph,"color").values())
    labels = list(nx.get_node_attributes(subgraph,"color").keys())
    view = plotting.view_markers(coords, colors, size, labels)

    return view

