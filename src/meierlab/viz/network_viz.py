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

    Examples
    --------
    >>> from meierlab.viz import network_viz as nv
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

    Examples
    --------
    >>> from meierlab.viz import network_viz as nv
    """
    idx_list = list(nx.get_node_attributes(subgraph, "index").values())
    coords = atlas_coords[[int(idx) - 1 for idx in idx_list]]

    return coords


def plot_connectome_from_graph(g, atlas_coords, threshold="80%", dim="2d"):
    """Plot full connectome from a graph `g`.

    Parameters
    ----------
    g : :class:`networkx.Graph`
        Network graph to plot.
    atlas_coords : numpy.ndarray
        Nifti file of atlas/parcellation.
    threshold : str, optional
        Edge threshold for visualization, by default "80%".
    dim : str, optional
        View plot in 2d or 3d, by default "2d".

    Returns
    -------
    view : :class:`nilearn.plotting.html_connectome.ConnectomeView`
        Nilearn connectome viewer. Save as an html page or view using 'open_in_browser' method.

    Examples
    --------
    >>> from meierlab.viz import network_viz as nv
    """

    mtx = nx.convert_matrix.to_numpy_array(g)
    if dim == "2d":
        view = plotting.plot_connectome(mtx, atlas_coords, edge_threshold=threshold)
    else:
        view = plotting.view_connectome(mtx, atlas_coords, edge_threshold=threshold)

    return view


def plot_subgraph_connectome(subgraph, atlas_coords, threshold="80%", dim="2d"):
    """Plot the connectome for a `subgraph`, e.g. plot only one subnetwork.

    Parameters
    ----------
    subgraph : :class:`networkx.Graph.subgraph`
        Subgraph to plot.
    atlas_coords : numpy.ndarray
        Atlas coordinates.
    threshold : str, optional
        Edge threshold for viewing, by default "80%".
    dim : str, optional
        View plot in 2d or 3d, by default "2d".

    Returns
    -------
    view : :class:`nilearn.plotting.html_connectome.ConnectomeView`
        Nilearn connectome viewer with the subgraph. Save as an html page or view using 'open_in_browser' method.

    Examples
    --------
    >>> from meierlab.viz import network_viz as nv
    """
    coords = get_subgraph_coords(subgraph, atlas_coords)
    mtx = nx.convert_matrix.to_numpy_array(subgraph)

    if dim == "2d":
        view = plotting.plot_connectome(mtx, coords, edge_threshold=threshold)
    else:
        view = plotting.view_connectome(mtx, coords, edge_threshold=threshold)

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

    Examples
    --------
    >>> from meierlab.viz import network_viz as nv
    """
    coords = get_subgraph_coords(subgraph, atlas_coords)
    colors = list(nx.get_node_attributes(subgraph, "color").values())
    labels = list(nx.get_node_attributes(subgraph, "label").values())
    view = plotting.view_markers(coords, colors, size, labels)

    return view
