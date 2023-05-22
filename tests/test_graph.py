import pytest
import networkx as nx
from pkg_resources import resource_filename as pkgrf
from meierlab.networks import graph as ng

@pytest.fixture
def atlas():
    return pkgrf("tests/data/atlas.csv")


@pytest.fixture
def sub_file():
    return pkgrf("tests/data/sub-atlas.tsv")


@pytest.fixture
def G():
    return ng.gen_base_graph_from_atlas(atlas)

@pytest.fixture
def g():
    return ng.gen_graph_from_matrix(G,atlas,sub_file)


@pytest.fixture
def rsn_list():
    return ["DMN","Visual"]


def test_gen_base_graph_from_atlas(atlas):
    assert type(G) == nx.Graph
    assert list(G.nodes()) == [
        'LH_Vis_1',
        'LH_Vis_2',
        'LH_Vis_3',
        'LH_Vis_4',
        'LH_Vis_5',
        'RH_Vis_1',
        'RH_Vis_2',
        'RH_Vis_3',
        'RH_Vis_4',
        'RH_Vis_5',
        'LH_Default_Temp_1',
        'LH_Default_Temp_2',
        'LH_Default_Temp_3',
        'LH_Default_Temp_4',
        'LH_Default_Temp_5',
        'LH_Default_Temp_6',
        'LH_Default_Temp_7',
        'LH_Default_Temp_8',
        'LH_Default_Temp_9',
        'LH_Default_Temp_10',
        'RH_Default_Temp_1',
        'RH_Default_Temp_2',
        'RH_Default_Temp_3',
        'RH_Default_Temp_4',
        'RH_Default_Temp_5',
        'RH_Default_Temp_6',
        'RH_Default_Temp_7',
        'RH_Default_Temp_8'
    ]


def test_gen_graph_matrix(g):
    assert type(g) == nx.Graph


def test_gen_basic_metrics(G):
    pass


def test_gen_subnetwork_list(G, rsn_list):
    assert ng.gen_subnetwork_list(G) == rsn_list


def test_gen_subnetwork_subgraphs(G, rsn_list):
    pass


def test_gen_subnetwork_pairs():
    pass


def test_get_within_network_connectivity():
    pass


def test_get_between_network_connectivity():
    pass
