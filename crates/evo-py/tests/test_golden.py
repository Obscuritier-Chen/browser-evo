from pathlib import Path

import numpy as np

import evo_py

GOLDEN = Path(__file__).resolve().parents[2] / "evo-core" / "tests" / "golden"


def test_normal_text_golden_bits():
    checked = 0
    for line in (GOLDEN / "normal_bits.txt").read_text().splitlines():
        seed_hex, index_hex, bits_hex = line.split()
        value = np.float32(evo_py.normal(int(seed_hex, 16), int(index_hex, 16)))
        assert value.view(np.uint32) == np.uint32(int(bits_hex, 16)), line
        checked += 1
    assert checked == 4 * 4096


def test_million_normal_golden_bits():
    expected = np.fromfile(GOLDEN / "normal_1e6.bin", dtype="<u4")
    assert expected.size == 1_000_000
    out = np.zeros(expected.size, dtype=np.float32)
    evo_py.fill_perturbation(42, out)
    np.testing.assert_array_equal(out.view(np.uint32), expected)


def test_es_update_golden_bits():
    lines = (GOLDEN / "es_update.txt").read_text().splitlines()
    dim = 16
    seeds = np.arange(1, 9, dtype=np.uint64)
    weights = np.array([0.5, -0.25, 0.125, 0.0, -0.5, 0.75, -0.125, 0.25], dtype=np.float32)
    theta = np.array([(i - 4) * 0.25 for i in range(dim)], dtype=np.float32)
    m = np.zeros(dim, dtype=np.float32)
    v = np.zeros(dim, dtype=np.float32)
    grad = np.zeros(dim, dtype=np.float32)
    row = 0
    for step in range(1, 4):
        evo_py.es_gradient(seeds, weights, 16, 0.5, grad)
        evo_py.adam_step(theta, grad, m, v, step, 0.0625)
        for index in range(dim):
            _, _, bits_hex = lines[row].split()
            assert theta[index].view(np.uint32) == np.uint32(int(bits_hex, 16)), (step, index)
            row += 1
    assert row == len(lines)


def test_ranks_and_sgd():
    fitness = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    evo_py.centered_ranks(fitness, out)
    np.testing.assert_array_equal(out, np.array([0.5, -0.5, 0.0], dtype=np.float32))
    theta = np.array([1.0, -1.0], dtype=np.float32)
    evo_py.sgd_step(theta, np.array([0.5, -0.5], dtype=np.float32), 0.1)
    np.testing.assert_allclose(theta, [0.95, -0.95], atol=1e-6)
