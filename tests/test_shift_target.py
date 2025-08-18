import pandas as pd

from analysis.correlation_utils import corr_weights, shift_target


def test_shift_target_rotation():
    corr_df = pd.DataFrame(
        [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )

    target, peers = "AAA", ["BBB", "CCC"]
    w0 = corr_weights(corr_df, target, peers)
    assert set(w0.index) == set(peers)

    # Shift forward: BBB becomes target
    new_target, new_peers = shift_target(target, ["AAA", "BBB", "CCC"])
    assert new_target == "BBB"
    assert new_peers == ["AAA", "CCC"]
    w1 = corr_weights(corr_df, new_target, new_peers)
    assert set(w1.index) == set(new_peers)

    # Shift backward: return to original target
    back_target, back_peers = shift_target(new_target, ["AAA", "BBB", "CCC"], shift=-1)
    assert back_target == target
    assert back_peers == peers
