import pytest
from MomentMatching.Nodes import build_nodes, node_system
from Systems import TestDynamics


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


def test_node_system(N=23, dim=1):
    system = TestDynamics()
    data = system.simulate(N)
    x_true, x_noisy, y_true, y_noisy = zip(*data)
    nodes = build_nodes(N, dim)
    nodes = node_system(nodes, system, y_noisy)
    for i, node in enumerate(nodes):
        assert node.trans_func(i) == 0.1 * i + i%10
        assert node.meas == y_noisy[i]


# def test_