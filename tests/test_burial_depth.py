import numpy as np

from interfaceshapeai.geometry.burial_depth import geometric_burial_depth


def test_center_atom_more_buried_than_isolated_atom():
    # A dense cluster of atoms around the origin, plus one far-away isolated atom.
    cluster = np.random.RandomState(0).normal(scale=1.0, size=(20, 3))
    isolated = np.array([[100.0, 100.0, 100.0]])
    coords = np.vstack([cluster, isolated])

    depth = geometric_burial_depth(coords, probe_radius=3.0, normalize=True)

    assert depth.shape == (21,)
    assert depth.min() >= 0.0 and depth.max() <= 1.0
    assert depth[-1] == 0.0  # isolated atom has zero neighbors -> most exposed
    assert depth[:20].mean() > depth[-1]


def test_single_atom_is_zero():
    depth = geometric_burial_depth(np.array([[0.0, 0.0, 0.0]]))
    assert depth.shape == (1,)
    assert depth[0] == 0.0


def test_empty_input():
    depth = geometric_burial_depth(np.zeros((0, 3)))
    assert depth.shape == (0,)


def test_invalid_shape_raises():
    try:
        geometric_burial_depth(np.zeros((5, 2)))
        assert False, "expected ValueError"
    except ValueError:
        pass
