Read the final section of HANDOFF.md — "2026-08-24: the 6000-step default was
the bias; `rho` is retired; the centroid layer is now conditioned on shear" —
and carry out its "NEXT SESSION, in order" list.

Context you need up front:

- `models/centroid.py` now conditions the centroid layer's coefficients on g
  (two extra net inputs, 4 -> 6), and `centroid.py train` samples g and lenses
  the drawn copies by each copy's OWN exact derivatives. Those derivatives came
  from bfd's `makeTemplates` all along; `imsims/copies.py` was discarding nine
  of the ten Pqr slots and now stores them.
- Consequently EVERY copies catalog and EVERY centroid checkpoint on disk is
  stale. Step 1 re-renders the catalog (~3.5 GB, was ~1 GB), step 2 retrains.
- `bias.py` no longer peels a g-conditioned centroid layer, so deep runs are
  slower and heavier than the ~110 min the last one took. Size it with
  `--n-targets 20000` before committing to a full 200k run.
- A gauss2_deep run isolating the converged shear layer (60k shear + the OLD
  g = 0 centroid layer) was started and CANCELLED with no result. There is no
  log. HANDOFF.md describes it as "step 0", says why it is worth redoing, and
  notes it needs the pre-change centroid code checked out. Decide whether to do
  it or go straight to step 1.

Working rules for this project:
- Quote gauss2 m1 with the SEED spread (sd 0.0015 at 60k), never a single run's
  bootstrap (1.5e-4). Anything hoping to move m1 by less than ~0.003 needs 3
  seeds or paired arms.
- val NLL cannot rank flows for shear bias — it moved 3e-4 nats across a ladder
  where m1 moved 13x. Use `shear.py check` to compare training runs.
- `dm_dg` is `(n, 2, 5)`, G INDEX FIRST. `reshape(-1, 5, 2)` succeeds silently
  and transposes the data. There is a test pinning it.
- Do not spend effort on `rho`'s bound or parameterisation, and do not retry the
  `e_dir` substitution. Both are measured dead ends; the handoff says why.
- No fine-tuning, and only converged flows: `shear.py train --steps 60000
  --deriv-weight 1e4`, `bulk.py train --steps 150000`. The CLI defaults (6000
  and 4000) are both wrong and are what produced every stale number in the repo.

Nothing is committed. `git status` shows the working tree in both this repo and
../bfd_cnf_imsims; review the diff before you build on it.
