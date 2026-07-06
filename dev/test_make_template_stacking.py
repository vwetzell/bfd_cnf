#!/usr/bin/env python
"""Check that the sub-file batched chunks restack, in sorted order, into the exact
in-order concatenation with HDF5 and FITS rows still aligned by id.

This is the contract make_template_file.py relies on after switching to per-batch
chunks. Runs without the multi-GB pscratch inputs by feeding the real stackers
synthetic chunks.

    python dev/test_make_template_stacking.py
"""
import glob
import importlib.util
import os
import tempfile

import numpy as np
import h5py
from astropy.table import Table

_spec = importlib.util.spec_from_file_location(
    'mtf', os.path.join(os.path.dirname(__file__), 'make_template_file.py'))
mtf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mtf)


def _write_hdf5_chunk(path, ids):
    dt = np.dtype([('id', '<i8'), ('derivs', '<f4', (7, 10)), ('nda', '<f4')])
    rec = np.zeros(len(ids), dtype=dt)
    rec['id'] = ids
    rec['nda'] = ids  # marker so we can confirm payload travels with the id
    with h5py.File(path, 'w') as f:
        f.create_dataset('templates', data=rec)
        f.create_dataset('templates.__table_column_meta__',
                         data=np.array([b'meta']))


def _write_fits_chunk(path, ids):
    Table({'id': np.asarray(ids, np.int64),
           'moments': np.asarray(ids, np.float64)[:, None] * np.ones((1, 5))}
          ).write(path, overwrite=True)


def test_stacking_preserves_order_and_alignment():
    # Mimic the real chunk naming: per input file, per worker, per batch. Sorted
    # glob order (file, worker, batch) defines the expected stacked order; HDF5 and
    # FITS must come out identically ordered so rows stay aligned by id.
    plan = {
        'tmp_fileA_w0_b000': [0, 1, 2],
        'tmp_fileA_w0_b001': [3, 4],
        'tmp_fileA_w1_b000': [5, 6],
        'tmp_fileB_w0_b000': [10, 11],
        'tmp_fileC_w0_b000': [20],
        'tmp_fileC_w1_b000': [21, 22, 23],
    }
    with tempfile.TemporaryDirectory() as d:
        for base, ids in plan.items():
            _write_hdf5_chunk(os.path.join(d, base + '.hdf5'), ids)
            _write_fits_chunk(os.path.join(d, base + '_summary.fits'), ids)

        h5_chunks = sorted(glob.glob(os.path.join(d, 'tmp_*.hdf5')))
        fits_chunks = sorted(glob.glob(os.path.join(d, 'tmp_*_summary.fits')))
        expected = [i for base in sorted(plan) for i in plan[base]]

        out_h5 = os.path.join(d, 'out.hdf5')
        out_fits = os.path.join(d, 'out_summary.fits')
        total = mtf._stack_hdf5_incremental(h5_chunks, out_h5)
        mtf._stack_fits_streaming(fits_chunks, out_fits)

        assert total == len(expected), (total, len(expected))
        with h5py.File(out_h5, 'r') as f:
            h5_ids = f['templates']['id'][:]
            h5_nda = f['templates']['nda'][:]
        f_ids = np.asarray(Table.read(out_fits)['id'])

        assert list(h5_ids) == expected, list(h5_ids)
        assert np.array_equal(h5_ids, f_ids), 'HDF5/FITS rows not aligned by id'
        assert np.array_equal(h5_nda.astype(np.int64), h5_ids), 'payload detached from id'
    print('OK: order preserved and HDF5/FITS rows aligned across batched chunks')


if __name__ == '__main__':
    test_stacking_preserves_order_and_alignment()
