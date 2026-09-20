from src.data.loaders import available_datasets, load_preference_subset


def test_synthetic_toy_is_offline_and_deterministic():
    a = load_preference_subset("synthetic-toy", n=20, seed=0)
    b = load_preference_subset("synthetic-toy", n=20, seed=0)
    assert len(a) == 20
    assert a.prompts == b.prompts
    assert a.chosen == b.chosen
    assert a.rejected == b.rejected


def test_unknown_dataset_raises():
    try:
        load_preference_subset("does-not-exist", n=5)
    except ValueError as e:
        assert "does-not-exist" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_registry_lists_expected_datasets():
    names = available_datasets()
    assert "synthetic-toy" in names
    assert "hh-rlhf-helpful" in names
    assert "hh-rlhf-harmless" in names
    assert "shp" in names
