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

- The 2x2 that isolates it has since finished, and it points at the CENTROID
  layer. gauss2_deep, 20k targets, same targets throughout:

  | shear flow | centroid | windowed, corrected | unwindowed |
  |---|---|---|---|
  | pre-fix | off | -0.01084 +/- 0.00268 | -0.00937 +/- 0.00462 |
  | post-fix | off | -0.00771 +/- 0.00264 | -0.00992 +/- 0.00156 |
  | pre-fix | on (200k) | -0.01005 +/- 0.00170 | ~ -0.010 |
  | post-fix | on | +0.01530 +/- 0.00284 | -0.02184 +/- 0.00243 |

  With the centroid layer OFF everything is healthy: the two shear flows agree,
  the post-fix one is marginally better, and windowed/unwindowed agree in both
  rows. The blowup and the estimator divergence appear ONLY with the newly
  retrained centroid layer. So it is not the shear layer, not the chart fix,
  and NOT the selection terms.

FIRST THING: one control was left running to make the fourth cell exact at 20k
(the pre-fix with-centroid number above is a 200k run). Read

    logs_deep_cent_pre_chartsync.txt

If it lands near -0.010 like its 200k counterpart, the 2x2 is clean and the
centroid retrain is confirmed as the sole culprit. If it died, the command is
in the handoff.

Then the real question: **what does `centroid.py train` do wrong on top of a
sharp shear layer?** Its own check already reports a 4.36x ellipticity response
against the catalog (3.65e-2 vs 8.36e-3) -- untouched by the chart fix, and
previously masked by a shear layer that was itself wrong by 4-5% in dm/dg. The
spin-0 shifts are fine (right sign, 72-86% of catalog). Suspect the spin-2 path.
Note the centroid layer is g-conditioned and sign-constrained as of dc41c8c;
that part is validated, do not re-open it.

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
