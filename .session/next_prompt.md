Read the last two sections of HANDOFF.md -- "2026-08-24 (late): the Mr/Mf ~ 3.1
'instability' was a stale chart, and it is fixed" and "2026-08-24 (overnight):
the fix is committed and the chain is rebuilt; the response fit improved 6x,
the noisy bias did NOT" -- and pick up from there.

Where things stand:

- A real bug was found, fixed and committed (`03e3551`).
  `RawMomentStandardize.mean/std` are trainable and `bulk.train` moves them a
  lot; `shear.py --init` grafts that trained chart in under a freshly built
  `ShearResponse` whose frozen `chart_loc`/`chart_scale`/`e_scale` still hold
  the raw-sample values. `_chart_spin0_jac` rebuilds logit(u) from those, so
  the 1/(1-u) size Jacobian came out wrong in a way that varies along
  z1 = Mr/Mf. That was the ENTIRE long-open "shear response is locally
  unstable at Mr/Mf ~ 3.1" phenomenon. `bulk.sync_chart_constants` fixes it;
  65 tests pass. That work is DONE -- do not re-open it.
- The gauss2 chain was rebuilt on the fix. The response fit improved ~6x
  (dm/dg Mr 4.28% -> 0.66%, the r~3.1 peak 6.81% -> 0.81%, profile now flat)
  and the noiseless m1 is unchanged-to-slightly-better (+0.00047 +/- 0.00017
  against a -0.0028 +/- 0.0015 seed mean). **The shear layer is fine.**
- But the deep noisy measurement got worse: windowed-corrected m1
  +0.01530 +/- 0.00284 at 20k targets, against -0.01005 +/- 0.00170 before.
  And the windowed-corrected and unwindowed numbers, which used to AGREE
  (that agreement was one of the three checks that localised the residual),
  now differ by 0.037: +0.0153 windowed vs -0.0218 unwindowed. ESS is healthy
  (median 638, frac<10 = 0.000), so it is not the importance sampling.

FIRST THING: two runs were left going overnight and should be finished. Read

    logs_deep_nocent_shear_gauss2.txt
    logs_deep_nocent_pre_chartsync_shear_gauss2.txt

(deep, 20k targets, `--no-centroid`, on the post-fix and pre-fix shear flows).
With the two with-centroid numbers already in the handoff these complete a 2x2
-- (old shear, new shear) x (with centroid, without) -- on the same targets.
The reference for the no-centroid column is -0.0103 +/- 0.0027. That 2x2 says
whether the +0.0153 is the shear layer at depth or the centroid layer
retrained on top of it. If either run died, relaunch it; the exact command is
in the handoff.

Then the real question, and it is the more interesting one: **why do the
windowed-corrected and unwindowed estimators now disagree by 0.037 when they
used to agree?** The selection terms (eq. 40/45-46, `--window-size`/
`--window-flux`) are valid only for a cut on each arm's own observed M; a
sharper response layer changes what lands inside the window, so the correction
is being exercised differently than before. Worth checking whether the
disagreement tracks the window edges, and whether it survives with the centroid
layer off.

Working rules, unchanged:

- Quote gauss2 m1 with the SEED spread (sd ~0.0015-0.002 at 60k), never a
  single run's bootstrap.
- val NLL cannot rank flows for shear-response fit quality -- confirmed again
  tonight (42.4112 to four decimals on flows differing 6x in dm/dg accuracy).
- `dm_dg` is `(n, 2, 5)`, G INDEX FIRST; `d2m_dg2` is `(n, 3, 5)`.
  `reshape(-1, 5, 2)` succeeds silently and transposes the data.
- No fine-tuning, and only converged flows: `shear.py train --steps 60000
  --deriv-weight 1e4`, `bulk.py train --steps 150000`, `centroid.py train
  --steps 16000`. Do not search these.
- `bias.py` on gauss2_deep now NEEDS `--batch-budget 65536` or it OOMs the
  16 GB card in the first `pqr_streamed` batch. Pure device chunking, does not
  change the arithmetic.
- When testing a hypothesis with a toy, verify the toy's own optimisation
  converged before interpreting it -- and prefer a controlled knob in the REAL
  training loop over a simpler proxy. That is exactly why tonight's bug was
  found and five earlier proxy toys all came back null.

Not yet done, in rough priority order: a 200k-target rerun on the rebuilt
chain (the 20k number is not quotable); retraining bulgedisc and sersic
against the fix (their chart drift is the same size, so every non-gauss2
shear/centroid checkpoint on disk is still void); the centroid layer's 4.36x
ellipticity-response error, untouched by any of this; and the open design
question of whether `RawMomentStandardize.mean/std` should be trainable at all.

`flows/pre_chartsync/` holds the pre-fix gauss2 shear and centroid
checkpoints -- keep them, every paired comparison needs them.
