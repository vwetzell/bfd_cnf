#!/bin/bash
# Test BFD-faithful per-copy weight nda·L (detj_copy dropped) vs centroidfix (nda·detj·L).
# Train (continue from centroidfix) -> integrate -> paired Δm.  Runs unattended.
set -e
cd /home/vwetzell/gitrepos/bfd_cnf

echo "=== [1/3] continued training: nda·L (drop detj_copy) ==="
python -m bfd_cnf.converge_train --loss nll \
  --prior-in flows/prior_flow_xy_nll_centroidfix.eqx \
  --prior-out flows/prior_flow_xy_nll_ndaL.eqx \
  --learning-rate 1e-5 --lr-schedule cosine --lr-final 1e-7 --lr-decay-steps 100000 \
  --block-steps 10000 --max-blocks 10 --min-blocks 10 \
  --tag nll_ndaL

echo "=== [2/3] integration ==="
python -m bfd_cnf.integrate_grid --independent \
  --n-targets 100000 --flux-min 1500 --flux-max 90000 \
  --prior flows/prior_flow_xy_nll_ndaL.eqx \
  --out data/pqr_grid_independent_ndaL.npz

echo "=== [3/3] paired Δm (centroidfix -> ndaL, isolates detj_copy removal) ==="
JAX_PLATFORMS=cpu python - <<'PYEOF'
import numpy as np
o=np.load('data/pqr_grid_independent.npz')            # original (nda·detj·L, detj g=0... pre centroid)
a=np.load('data/pqr_grid_independent_centroidfix.npz')# nda·detj·L + centroid fix
b=np.load('data/pqr_grid_independent_ndaL.npz')       # nda·L + centroid fix
def gt(pqr):
    P=pqr[:,0]; Q=pqr[:,1:3]
    R=np.stack([np.stack([pqr[:,3],pqr[:,5]],-1),np.stack([pqr[:,5],pqr[:,4]],-1)],-2)
    ok=np.isfinite(P)&(P>0); return Q/P[:,None], np.einsum('ni,nj->nij',Q,Q)/P[:,None,None]**2-R/P[:,None,None], ok
def sol(Qt,Rt): return np.linalg.solve(Rt.sum(0)+1e-10*np.eye(2),Qt.sum(0))[0]
def m_of(f):
    Qp,Rp,okp=gt(f['pqr_p']); Qm,Rm,okm=gt(f['pqr_m'])
    return (sol(Qp[okp],Rp[okp])-sol(Qm[okm],Rm[okm]))/0.04-1
print(f"original (pre-fix)      m = {m_of(o):+.4f}")
print(f"centroidfix (nda·detj·L) m = {m_of(a):+.4f}")
print(f"ndaL       (nda·L)       m = {m_of(b):+.4f}")
# paired bootstrap Δm = ndaL - centroidfix (same targets/resamples)
Qap,Rap,okap=gt(a['pqr_p']); Qam,Ram,okam=gt(a['pqr_m'])
Qbp,Rbp,okbp=gt(b['pqr_p']); Qbm,Rbm,okbm=gt(b['pqr_m'])
kp=okap&okbp; km=okam&okbm
Qap,Rap,Qbp,Rbp=Qap[kp],Rap[kp],Qbp[kp],Rbp[kp]; Qam,Ram,Qbm,Rbm=Qam[km],Ram[km],Qbm[km],Rbm[km]
rng=np.random.default_rng(0); dm=[]
for _ in range(2000):
    ip=rng.integers(0,kp.sum(),kp.sum()); im=rng.integers(0,km.sum(),km.sum())
    ma=(sol(Qap[ip],Rap[ip])-sol(Qam[im],Ram[im]))/0.04-1
    mb=(sol(Qbp[ip],Rbp[ip])-sol(Qbm[im],Rbm[im]))/0.04-1
    dm.append(mb-ma)
dm=np.array(dm)
print(f"\npaired Δm (detj_copy removal) = {dm.mean():+.4f} ± {dm.std():.4f}  ({abs(dm.mean()/dm.std()):.1f}σ)  P(Δ<0)={100*(dm<0).mean():.0f}%")
PYEOF
echo "=== DONE ==="
