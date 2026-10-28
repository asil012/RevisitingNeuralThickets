import numpy as np

from neural_thickets.data import Bundle


def test_score_bundle_covers_all_primary_cells_without_text():
    arrays = Bundle.default().npz("score_data.npz")
    cells = {key.split(".")[1] for key in arrays if key.startswith("cells.")}
    assert cells == {
        f"{model}_{task}"
        for model in ("olmo", "q3", "q7", "q32")
        for task in ("countdown", "gsm8k", "mbpp")
    }
    for cell in cells:
        prefix = f"cells.{cell}"
        reward = arrays[f"{prefix}.selection_reward"]
        assert reward.shape[1] == 200 and reward.dtype.kind == "f"
        assert arrays[f"{prefix}.selection_correct"].shape == reward.shape
        assert arrays[f"{prefix}.selection_valid"].shape == reward.shape
        assert arrays[f"{prefix}.top50_answer_group"].shape[0] == 50
        assert arrays[f"{prefix}.anchor_answer_group"].dtype == np.int16
    assert all(array.dtype.kind not in {"O", "S", "U"} for array in arrays.values())


def test_score_bundle_candidate_and_question_contracts():
    arrays = Bundle.default().npz("score_data.npz")
    rng = np.random.default_rng(42)
    seeds = rng.choice(2**31, size=5000, replace=False)
    sigmas = rng.choice([0.0005, 0.001, 0.002], size=5000)
    for model, n_candidates in (("olmo", 5000), ("q3", 5000), ("q7", 5000), ("q32", 1000)):
        for task, n_questions in (("countdown", 1400), ("gsm8k", 1319), ("mbpp", 500)):
            cell = f"cells.{model}_{task}"
            np.testing.assert_array_equal(arrays[f"{cell}.candidate_ids"], np.arange(n_candidates))
            np.testing.assert_array_equal(arrays[f"{cell}.candidate_seeds"], seeds[:n_candidates])
            np.testing.assert_array_equal(arrays[f"{cell}.candidate_sigmas"], sigmas[:n_candidates])
            np.testing.assert_array_equal(arrays[f"{cell}.selection_qids"], np.arange(200))
            ranking = arrays[f"{cell}.selection_ranking"]
            reward = arrays[f"{cell}.selection_reward"]
            stable = np.argsort(-reward.mean(axis=1), kind="stable")
            np.testing.assert_array_equal(ranking, stable)
            top50 = arrays[f"{cell}.top50_candidate_ids"]
            if (model, task) == ("q32", "countdown"):
                assert set(top50) == set(stable[:50])
            else:
                np.testing.assert_array_equal(top50, ranking[:50])
            test_ids = arrays[f"{cell}.test_candidate_ids"]
            expected_rows = (
                500
                if (model, task) == ("olmo", "countdown")
                else 50
                if (model, task) == ("q32", "countdown")
                else n_candidates
            )
            assert len(test_ids) == expected_rows
            correct = arrays[f"{cell}.test_correct"]
            valid = arrays[f"{cell}.test_valid"]
            test_reward = arrays[f"{cell}.test_reward"]
            assert correct.shape == valid.shape == test_reward.shape == (expected_rows, n_questions)
            assert test_reward.dtype == np.float64
            assert np.all(correct <= valid)
            for family in ("top50", "anchor"):
                groups = arrays[f"{cell}.{family}_answer_group"]
                family_valid = arrays[f"{cell}.{family}_valid"]
                assert groups.shape == family_valid.shape
                assert np.all(groups[~family_valid] == -1)
                assert groups.min() >= -1
            fresh = arrays[f"{cell}.fresh_anchor_correct"]
            assert fresh.shape == (5, n_questions)
            assert arrays[f"{cell}.fresh_anchor_valid"].shape == fresh.shape


def test_headline_and_selection_transfer_matrices():
    bundle = Bundle.default()
    headline = bundle.npz("headline_data.npz")
    assert headline["correct"].shape == headline["reward"].shape == (1000, 1600)
    assert headline["reward"].dtype == np.float64
    assert headline["decay_reward"].shape == (1000, 1600)
    assert headline["decay_correct"].shape == headline["decay_valid"].shape == (1000, 1600)
    assert np.all(headline["decay_correct"] <= headline["decay_valid"])
    transfer = bundle.npz("selection_transfer_data.npz")
    assert transfer["selection_transfer.gsm8k.reward"].shape == (5000, 1000)
    assert transfer["selection_transfer.mbpp.reward"].shape == (5000, 400)
    assert transfer["selection_transfer.gsm8k.holdout_accuracy"].shape == (5000,)
    assert transfer["selection_transfer.mbpp.holdout_accuracy"].shape == (5000,)


def test_density_bundle_covers_all_18_cells_and_saved_orders():
    data = Bundle.default().npz("density_data.npz")
    for task, n_questions in (("countdown", 1400), ("gsm8k", 1319), ("mbpp", 500)):
        orders = data[f"density_plan.{task}.orders"]
        assert orders.shape == (200, n_questions)
        np.testing.assert_array_equal(
            np.sort(orders, axis=1), np.tile(np.arange(n_questions), (200, 1))
        )
        for model in ("olmo", "q05", "q15", "q3", "q7", "q32"):
            if model != "olmo":
                greedy = f"density.{model}_{task}.greedy"
                np.testing.assert_array_equal(
                    data[f"{greedy}.qids"], data[f"density_plan.{task}.qids"]
                )
                np.testing.assert_array_equal(data[f"{greedy}.seeds"], [42])
                assert data[f"{greedy}.correct"].shape == (1, n_questions)
                assert data[f"{greedy}.valid"].shape == (1, n_questions)
                assert data[f"{greedy}.reward"].shape == (1, n_questions)
            for family in ("weights", "sampling"):
                prefix = f"density.{model}_{task}.{family}"
                correct = data[f"{prefix}.correct"]
                valid = data[f"{prefix}.valid"]
                reward = data[f"{prefix}.reward"]
                assert correct.shape == valid.shape == reward.shape
                assert correct.shape[1] == n_questions
                assert reward.dtype == np.float64
                assert np.all(correct <= valid)


def test_uspto_bundle_is_per_question_and_text_free():
    data = Bundle.default().npz("uspto_data.npz")
    assert data["candidate_ids"].shape == (20,)
    for condition in ("original", "shuffled"):
        assert data[f"{condition}.expert_correct"].shape == (20, 1001)
        assert data[f"{condition}.anchor_correct"].shape == (1, 1001)
        assert np.all(data[f"{condition}.expert_correct"] <= data[f"{condition}.expert_valid"])
