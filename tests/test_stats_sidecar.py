"""Self-check for the standardiser sidecar provenance helpers.

Run: python tests/test_stats_sidecar.py
Checks (assert-based, no framework):
  1. save_stats then load_stats round-trips mean/std.
  2. a matching --stats override is accepted.
  3. a DISAGREEING --stats override raises (the wrong-stats guard).
  4. a missing sidecar with no override raises.
"""
import os
import tempfile

import numpy as np

from bfd_cnf.models.bijections import (
    RawMomentStandardize, load_stats, save_stats, stats_sidecar_path,
)

MEAN = np.array([3.4, 3.1, 0.0, 0.0])
STD = np.array([0.42, 0.48, 0.19, 0.19])


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        flow = os.path.join(d, "prior.eqx")  # the .eqx need not exist for stats IO
        r2s = RawMomentStandardize(mean=MEAN, std=STD)
        save_stats(flow, r2s)

        # 1. round-trip
        back = load_stats(flow)
        assert np.allclose(back.mean, MEAN) and np.allclose(back.std, STD)

        # 2. matching override accepted
        good = os.path.join(d, "good.npz")
        np.savez(good, mean=MEAN, std=STD)
        load_stats(flow, override=good)

        # 3. disagreeing override raises
        bad = os.path.join(d, "bad.npz")
        np.savez(bad, mean=MEAN + 0.5, std=STD)
        try:
            load_stats(flow, override=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("disagreeing --stats override should raise")

        # 4. missing sidecar, no override raises
        os.remove(stats_sidecar_path(flow))
        try:
            load_stats(flow)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing sidecar with no override should raise")

    print("OK: stats sidecar save/load/guard self-check passed.")


if __name__ == "__main__":
    main()
