"""Downloader size-cap and checksum regression tests."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as File
from download_coinfra import select_files, verified


class DownloaderTests(unittest.TestCase):
    def test_budget_and_missing_files(self):
        files = [File(rfilename="a.tar.zst", size=10), File(rfilename="b.tar.zst", size=20)]
        self.assertEqual(len(select_files(files, ["a.tar.zst", "a.tar.zst"], 10)), 1)
        with self.assertRaises(ValueError):
            select_files(files, ["a.tar.zst", "b.tar.zst"], 29)
        with self.assertRaises(ValueError):
            select_files(files, ["missing"], 100)
        with self.assertRaises(ValueError):
            select_files([File(rfilename="a.tar.zst", size=None)], ["a.tar.zst"], 100)

    def test_checksum_detects_same_size_corruption(self):
        content = b"test"
        f = File(size=4, lfs={"sha256": hashlib.sha256(content).hexdigest()})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive"
            path.write_bytes(content)
            self.assertTrue(verified(path, f))
            path.write_bytes(b"oops")
            self.assertFalse(verified(path, f))
            path.write_bytes(b"shorter-or-longer")
            self.assertFalse(verified(path, f))


if __name__ == "__main__":
    unittest.main()
