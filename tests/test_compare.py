import pytest

from multimodal_rag.eval.compare import answerable_within, paired_bootstrap, reciprocal_rank


def test_reciprocal_rank():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def test_answerable_within_matches_rank_cutoff():
    assert answerable_within(3, k=3) == 1.0
    assert answerable_within(4, k=3) == 0.0
    assert answerable_within(None, k=5) == 0.0


def test_identical_methods_give_zero_interval():
    assert paired_bootstrap([0.2, 0.5, 1.0], [0.2, 0.5, 1.0]) == (0.0, 0.0, 0.0)


def test_constant_difference_gives_point_interval():
    # Every question improves by exactly 0.25, so every resample's mean is 0.25.
    mean, low, high = paired_bootstrap([0.0, 0.25, 0.5], [0.25, 0.5, 0.75])
    assert (mean, low, high) == pytest.approx((0.25, 0.25, 0.25))


def test_mean_is_inside_interval_and_seed_makes_it_repeatable():
    base = [0.0, 1.0, 0.5, 0.0, 0.2, 1.0]
    other = [1.0, 1.0, 0.5, 0.5, 0.0, 1.0]
    # differences: 1, 0, 0, 0.5, -0.2, 0 -> mean 1.3 / 6
    first = paired_bootstrap(base, other)
    assert first[0] == pytest.approx(1.3 / 6)
    assert first[1] <= first[0] <= first[2]
    assert paired_bootstrap(base, other) == first


def test_lengths_must_match():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])
