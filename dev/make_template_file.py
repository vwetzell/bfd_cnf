#!/usr/bin/env python
"""
Create a BFD template HDF5 file from phase 11 template_moments_container_*.npy outputs.

Single node:
    python make_template_file.py <input_folder> <output.hdf5> --sigma_flux 5.2 ...

Multi-node (Slurm) — numpy is OpenBLAS, so cap threads per task. Run roughly one
MPI rank per input file (so every file processes in parallel) and use --workers
to fan each file's objects across the remaining idle cores via fork (the input is
loaded once per rank and shared copy-on-write). With 166 files on 4x128-core nodes:
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    srun --nodes=4 --tasks-per-node=42 --cpu-bind=cores \
        python make_template_file.py <input_folder> <output.hdf5> --workers 3 ...
"""
import argparse
import glob
import multiprocessing as mp
import os

import warnings

import numpy as np
import h5py
from astropy.table import Table
from astropy.io import fits as _fits

warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)
from mpi4py import MPI

import bfd
import bfd.momentcalc as _bfd_mc
from bfd.momenttable import TemplateTable, STable
from bfd.pqr import Pqr
from bfd.moment import Template as _Template


def _patch_bfd_compat():
    """Add classes that moved out of bfd.momentcalc in newer bfd versions."""
    import bfd
    import bfd.momentcalc as mc
    for name in ('KBlackmanHarris', 'KSigmaWeight'):
        if not hasattr(mc, name):
            setattr(mc, name, getattr(bfd, name))


_patch_bfd_compat()


def _stack_hdf5_incremental(tmp_files, out_path):
    """Concatenate TemplateTable HDF5 chunks without loading all into RAM.

    Each intermediate HDF5 stores a single compound dataset at path 'templates'
    plus a 'templates.__table_column_meta__' string dataset.  We copy the
    metadata once and grow the compound dataset one file at a time via h5py
    resizable datasets, keeping peak RAM to one chunk at a time.
    """
    total = 0
    with h5py.File(out_path, 'w') as fout:
        main_dset = None
        for i, f in enumerate(tmp_files):
            with h5py.File(f, 'r') as fin:
                data = fin['templates'][:]      # one chunk into RAM
                n = data.shape[0]
                if i == 0:
                    # Copy column/table metadata (small string dataset)
                    if 'templates.__table_column_meta__' in fin:
                        fout.create_dataset(
                            'templates.__table_column_meta__',
                            data=fin['templates.__table_column_meta__'][:],
                        )
                    # Create resizable compound dataset
                    main_dset = fout.create_dataset(
                        'templates',
                        data=data,
                        maxshape=(None,),
                        chunks=(min(n, 50_000),),
                    )
                else:
                    old_n = main_dset.shape[0]
                    main_dset.resize(old_n + n, axis=0)
                    main_dset[old_n:] = data
                total += n
                del data
    return total


def _stack_fits_streaming(fits_tmp_files, fits_out):
    """Concatenate per-chunk FITS binary tables without loading all into RAM.

    Strategy:
      1. Count rows from each file's header (no data loading).
      2. Write a 1-row skeleton to establish the correct column structure.
      3. Patch NAXIS2 in the header bytes and extend the file on disk with
         zeros so the total data area matches.
      4. Seek-and-write each chunk's raw data bytes sequentially.
    Peak RAM is one chunk's worth of data at a time.
    """
    BLOCK = 2880

    # Count rows without loading data
    n_per_file = []
    for f in fits_tmp_files:
        with _fits.open(f, memmap=True) as h:
            n_per_file.append(int(h[1].header['NAXIS2']))
    total_rows = sum(n_per_file)

    # Write 1-row skeleton to establish column structure
    with _fits.open(fits_tmp_files[0], memmap=True) as h:
        naxis1 = int(h[1].header['NAXIS1'])
        sample = np.array(h[1].data[:1])

    _fits.HDUList([_fits.PrimaryHDU(), _fits.BinTableHDU(data=sample)]).writeto(
        fits_out, overwrite=True
    )

    # Locate data start and patch NAXIS2 in-place
    with _fits.open(fits_out) as h:
        data_start = h[1].fileinfo()['datLoc']

    naxis2_card = (f'NAXIS2  = {total_rows:>20d} / number of rows in table'
                   .ljust(80))
    assert len(naxis2_card) == 80
    with open(fits_out, 'r+b') as fp:
        header_bytes = fp.read(data_start)
        idx = header_bytes.find(b'NAXIS2  ')
        fp.seek(idx)
        fp.write(naxis2_card.encode('ascii'))

    # Extend file to hold all rows (zero-filled padding is fine for FITS)
    padded = ((naxis1 * total_rows + BLOCK - 1) // BLOCK) * BLOCK
    total_file_size = data_start + padded
    with open(fits_out, 'r+b') as fp:
        fp.seek(0, 2)
        if total_file_size > fp.tell():
            fp.write(b'\x00' * (total_file_size - fp.tell()))

    # Write each chunk's data bytes sequentially (one chunk in RAM at a time)
    row_start = 0
    with open(fits_out, 'r+b') as fp:
        for i, f in enumerate(fits_tmp_files):
            with _fits.open(f, memmap=True) as h:
                chunk_bytes = bytes(h[1].data.tobytes())
            fp.seek(data_start + row_start * naxis1)
            fp.write(chunk_bytes)
            row_start += n_per_file[i]


def _mom_deriv(tmpl, idx):
    return np.concatenate([tmpl.even[:, idx], tmpl.odd[:, idx]]).astype(np.float64)


def _template_fits_rows(result, cov_packed):
    """Build one minimal-FITS dict per template from a list of Template objects."""
    rows = []
    cov64 = cov_packed.astype(np.float64)
    for tmpl in result:
        rows.append({
            'id':       np.int64(tmpl.id),
            'moments':  tmpl.even[:, Pqr.D0].astype(np.float64),
            'cov':      cov64,
            'dm_dg':    np.array([_mom_deriv(tmpl, Pqr.DG1),
                                   _mom_deriv(tmpl, Pqr.DG2)], dtype=np.float64),
            'd2m_dg2':  np.array([_mom_deriv(tmpl, Pqr.DG1DG1),
                                   _mom_deriv(tmpl, Pqr.DG1DG2),
                                   _mom_deriv(tmpl, Pqr.DG2DG2)], dtype=np.float64),
            # First-order (odd) moments [MX, MY] in MOMENT units — the X that the
            # ELBO's L(X|C_X) weights against C_X (=cov.odd, ~sigmaXY). BFD pairs
            # these with sigmaXY in chisq_xy (momentcalc.py:548). NOT xy_kept, the
            # sky-frame centroid shift — that is a different quantity, ~sigmaXY/Mf
            # smaller, and leaves L≈1 (inert) if fed to a moment-units C_X.
            'centroid': tmpl.odd[:, Pqr.D0].astype(np.float64),
            'nda':      np.float64(tmpl.nda),
        })
    return rows


# Flush a sub-file chunk every this many templates so peak RAM is bounded by one
# batch regardless of grid density (high sigma_max / small sigma_step explode the
# template count per object). Chunks are restitched by the existing streaming
# stackers.  ~200k templates ~= 60 MB HDF5 + ~0.7 GB transient FITS build.
BATCH = 200_000


# Loaded input dict for the file being processed. Set in the parent before the
# worker Pool is forked so children inherit it copy-on-write (no per-worker copy,
# no pickling the ~1.6 GB across the process boundary).
_INPUT = None


def _process_ids(out_path_base, worker_id, sub_ids, sigma_xy, sigma_flux, sn_min,
                 sigma_max, sigma_step, xy_max, weight_specs, sample_specs):
    """Turn one worker's slice of object ids into numbered chunk files.

    Chunks are tagged by worker (``_w{W}_b{NNN}``); the HDF5 and FITS chunks for a
    given (worker, batch) are written from the same ids in the same order, so the
    sorted-glob stack keeps them row-aligned. Reads objects from the module global
    _INPUT (fork-inherited)."""
    m = _INPUT

    def new_tab():
        t = TemplateTable(weightSpecs=weight_specs, sampleSpecs=sample_specs)
        t.meta['tierNumber'] = 0
        return t

    tab = new_tab()
    fits_rows = []
    batch_idx = 0
    n_in_batch = 0
    n_added = 0
    n_failed = 0

    def flush():
        nonlocal tab, fits_rows, batch_idx, n_in_batch
        if n_in_batch == 0:
            return
        stem = f'{out_path_base}_w{worker_id}_b{batch_idx:03d}'
        tab.write(stem + '.hdf5', overwrite=True)
        if fits_rows:
            keys = list(fits_rows[0].keys())
            fits_tab = Table({k: np.array([r[k] for r in fits_rows]) for k in keys})
            fits_tab.write(stem + '_summary.fits', overwrite=True)
        batch_idx += 1
        tab = new_tab()
        fits_rows = []
        n_in_batch = 0

    for ID in sub_ids:
        ret = m[ID]['moments'].makeTemplates(
            sigma_xy,
            sigmaFlux=sigma_flux,
            sn_min=sn_min,
            sigmaMax=sigma_max,
            sigmaStep=sigma_step,
            xy_max=xy_max,
        )
        if len(ret) == 2:  # early failure: (False, error_string)
            n_failed += 1
            continue
        flag, result, xy_kept, area_integral = ret
        result = [t for t in result if isinstance(t, _Template)]
        if flag and area_integral < 1.01 and len(result) > 0:
            for tmpl in result:
                tmpl.id = int(ID)
            tab.add(result)
            n_in_batch += len(result)
            n_added += len(result)
            try:
                cov_packed = m[ID]['moments'].getCovariance()[0].pack()
                fits_rows.extend(_template_fits_rows(result, cov_packed))
            except Exception:
                pass
        else:
            n_failed += 1

        if n_in_batch >= BATCH:  # flush at object boundary, not mid-object
            flush()

    flush()  # remainder
    return n_added


def process_file(path, sigma_xy, sigma_flux, sn_min, sigma_max, sigma_step, xy_max,
                 weight_specs, sample_specs, out_path_base, n_workers):
    global _INPUT
    _INPUT = np.load(path, allow_pickle=True).item()
    ids = list(_INPUT.keys())
    base = os.path.basename(out_path_base)
    print(f'    {base}: {len(ids)} objects across {n_workers} worker(s)', flush=True)

    common = (sigma_xy, sigma_flux, sn_min, sigma_max, sigma_step, xy_max,
              weight_specs, sample_specs)
    try:
        if n_workers <= 1:
            return _process_ids(out_path_base, 0, ids, *common)
        # Strided slices spread the rare bright (template-heavy) objects across
        # workers so no single worker becomes the long pole.
        tasks = [(out_path_base, w, ids[w::n_workers], *common)
                 for w in range(n_workers)]
        try:
            # fork (not spawn) so workers share _INPUT copy-on-write.
            with mp.get_context('fork').Pool(n_workers) as pool:
                return sum(pool.starmap(_process_ids, tasks))
        except OSError as e:
            # ENOMEM here is fork() refused under node memory-commit pressure, not
            # a real heap exhaustion. Fall back to serial so the file still finishes
            # (serial doesn't fork, so it fits). Drop any partial worker chunks first.
            print(f'    {base}: parallel fork failed ({e}); processing serially',
                  flush=True)
            for stale in glob.glob(f'{out_path_base}_w*'):
                os.remove(stale)
            return _process_ids(out_path_base, 0, ids, *common)
    finally:
        _INPUT = None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(
        description='Create a BFD template file from phase 11 moment container outputs.'
    )
    parser.add_argument('input_folder', help='Folder containing template_moments_container_*.npy')
    parser.add_argument('output_file', help='Output HDF5 file (e.g. templates.hdf5)')
    parser.add_argument('--sigma_flux', type=float, required=True)
    parser.add_argument('--sigma_xy',   type=float, required=True)
    parser.add_argument('--sigma_step', type=float, required=True,
                        help='Max template spacing in sigma units')
    parser.add_argument('--sigma_max',  type=float, required=True,
                        help='Max chi-sq radius for shifted copies')
    parser.add_argument('--xy_max',     type=float, default=4.0,
                        help='Max centroid shift in sky units (default: 4.0)')
    parser.add_argument('--sn_min',     type=float, default=3.0,
                        help='Minimum S/N flux cut (default: 3.0)')
    parser.add_argument('--filter_sigma', type=float, default=0.65,
                        help='BFD weight function sigma (default: 0.65)')
    parser.add_argument('--weight_n',   type=int,   default=4,
                        help='BFD weight function N (default: 4)')
    parser.add_argument('--workers',    type=int,   default=1,
                        help='Fork this many processes per rank to split each '
                             'file\'s objects across idle cores (default: 1)')
    args = parser.parse_args()

    weight_specs = {
        'weightType': 'KSigma',
        'weightN': args.weight_n,
        'weightSigma': args.filter_sigma,
    }
    sample_specs = {
        'sigmaMax': args.sigma_max,
        'sigmaStep': args.sigma_step,
        'fluxMin': args.sn_min * args.sigma_flux,
        'sigmaXY': args.sigma_xy,
        'sigmaFlux': args.sigma_flux,
    }

    files = sorted(glob.glob(os.path.join(args.input_folder, 'template_moments_container_*.npy')))

    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_file)), '_tmp_templates')
    if rank == 0:
        os.makedirs(tmp_dir, exist_ok=True)
        print(f'Found {len(files)} input files across {size} ranks')
    comm.Barrier()  # ensure tmp_dir exists before workers start writing

    # Each rank processes every size-th file starting at rank
    idx = rank
    while idx < len(files):
        f = files[idx]
        base = os.path.basename(f).replace('template_moments_container', 'tmp').replace('.npy', '')
        out_path_base = os.path.join(tmp_dir, base)
        done_marker = out_path_base + '.done'

        if os.path.exists(done_marker):
            print(f'[rank {rank}] skip (done): {base}', flush=True)
        else:
            print(f'[rank {rank}] processing: {os.path.basename(f)}', flush=True)
            # Drop any partial chunks from a prior crashed run before reprocessing
            for stale in glob.glob(f'{out_path_base}_w*') + glob.glob(f'{out_path_base}_b*'):
                os.remove(stale)
            try:
                n_added = process_file(
                    f, args.sigma_xy, args.sigma_flux, args.sn_min,
                    args.sigma_max, args.sigma_step, args.xy_max,
                    weight_specs, sample_specs, out_path_base, args.workers,
                )
                open(done_marker, 'w').close()
                print(f'[rank {rank}] done: {base} ({n_added} templates)', flush=True)
            except Exception as e:
                print(f'[rank {rank}] ERROR on {base}: {e}', flush=True)

        idx += size

    comm.Barrier()

    # Rank 0 stacks all intermediates into the final output
    if rank == 0:
        tmp_files = sorted(glob.glob(os.path.join(tmp_dir, 'tmp_*.hdf5')))
        print(f'\nStacking {len(tmp_files)} intermediate HDF5 files...')
        total = _stack_hdf5_incremental(tmp_files, args.output_file)
        print(f'Done. Written to {args.output_file} ({total} templates total)')

        fits_tmp_files = sorted(glob.glob(os.path.join(tmp_dir, 'tmp_*_summary.fits')))
        if fits_tmp_files:
            fits_out = os.path.splitext(args.output_file)[0] + '_summary.fits'
            print(f'Stacking {len(fits_tmp_files)} FITS summary files...')
            _stack_fits_streaming(fits_tmp_files, fits_out)
            print(f'FITS summary written to {fits_out}')


if __name__ == '__main__':
    main()
