"""Regression checks for non-executable dataset formats."""

import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp

from quatec.data import _load_mapping, _load_matrix


class _WriteMarkerOnLoad:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (Path.write_text, (self.marker, "deserialized"))


class DataLoadingSecurityTests(unittest.TestCase):
    def test_included_json_mappings_still_load(self):
        root = Path(__file__).resolve().parents[1] / "data"
        for name, size in (("entry2index.json", 29531), ("ec2idx.json", 4963)):
            with self.subTest(name=name):
                mapping = _load_mapping(root / name)
                self.assertEqual(len(mapping), size)
                self.assertEqual(set(mapping.values()), set(range(size)))

    def test_pickle_payload_never_executes_with_any_mapping_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed.txt"
            payload = pickle.dumps(_WriteMarkerOnLoad(marker))
            for suffix in (".pkl", ".pickle", ".bin", ".json"):
                with self.subTest(suffix=suffix):
                    path = root / ("mapping" + suffix)
                    path.write_bytes(payload)
                    with self.assertRaises((ValueError, UnicodeDecodeError)):
                        _load_mapping(path)
                    self.assertFalse(marker.exists())

    def test_mapping_requires_json_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps([0, 1, 2]))
            with self.assertRaisesRegex(ValueError, "JSON object"):
                _load_mapping(path)

    def test_numeric_dense_and_sparse_matrices_still_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
            np.save(root / "matrix.npy", expected)
            sp.save_npz(root / "matrix.npz", sp.csr_matrix(expected))
            for name in ("matrix.npy", "matrix.npz"):
                with self.subTest(name=name):
                    np.testing.assert_array_equal(_load_matrix(root / name), expected)

    def test_numpy_object_array_does_not_execute_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed.txt"
            path = root / "matrix.npy"
            np.save(path, np.asarray([_WriteMarkerOnLoad(marker)], dtype=object))
            with self.assertRaises(ValueError):
                _load_matrix(path)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
