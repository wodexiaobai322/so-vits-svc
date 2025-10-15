import numpy as np

from inference import infer_tool


def test_pad_array_extends_symmetrically():
    arr = np.array([1, 2, 3])
    padded = infer_tool.pad_array(arr, 7)
    assert padded.shape == (7,)
    assert np.array_equal(padded[:2], [0, 0])
    assert np.array_equal(padded[-2:], [0, 0])
    assert np.array_equal(padded[2:5], [1, 2, 3])


def test_pad_array_noop_for_long_array():
    arr = np.ones(5)
    padded = infer_tool.pad_array(arr, 3)
    assert padded is arr


def test_split_list_by_n_generates_slices():
    source = list(range(10))
    chunks = list(infer_tool.split_list_by_n(source, 3))
    assert chunks[0] == [0, 1, 2]
    assert chunks[1] == [3, 4, 5]
    assert chunks[-1] == [9]


def test_fill_a_to_b_repeats_first():
    a = [1]
    b = [1, 2, 3, 4]
    infer_tool.fill_a_to_b(a, b)
    assert a == [1, 1, 1, 1]
