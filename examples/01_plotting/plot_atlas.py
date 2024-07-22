"""
Basic Atlas plotting
====================

Plot the regions of a reference atlas or parcellation.
Schaefer 2018, 400 Parcels, 7 Networks, with updated labels.
"""

from meierlab import datasets
from meierlab.networks import graph as ng
from meierlab.viz import network_viz as nv

schaefer = datasets.load_updated_schaefer()
atlas_coords = nv.get_atlas_coords(schaefer.atlas)

G = ng.gen_base_graph_from_atlas(schaefer.labels)
rsn_list = ng.gen_subnetwork_list(G)

######################################################
# Visualize the full network
# --------------------------
full_connectome = nv.plot_connectome_from_graph(G, atlas_coords)
full_connectome.show()

######################################################
# Visualize a single subnetwork, in this case Visual
# --------------------------------------------------
vis = [rsn for rsn in rsn_list if rsn == "Visual"]
vis_graph = ng.gen_subnetwork_subgraphs(G, vis)

vis = nv.plot_subgraph_connectome(vis_graph[0], atlas_coords)
vis.show()

###########################################################
# Visualize the nodes of a subnetwork, in this case Visual
# --------------------------------------------------------

vis_nodes = nv.plot_subgraph_nodes(vis_graph[0], atlas_coords)
# open in browser or save html with vis_nodes.save_as_html()
# vis_nodes.open_in_browser()
