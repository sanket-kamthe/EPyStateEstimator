import pytest
from MomentMatching.Nodes import build_nodes


def test_build_nodes(N=5, dim=2):
    nodes = build_nodes(N=N, dim=dim)
    assert nodes[0].prev_node == None
    assert nodes[-1].next_node == None
    assert nodes[1].next_node == nodes[2]
    assert nodes[1].prev_node == nodes[0]


def test_build_nodes_index(N=5, dim=2):
    nodes = build_nodes(N, dim)

    for i, node in enumerate(nodes):
        assert node.index == i


# def test_