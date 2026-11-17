"""Input isolation checks; runnable without the cluster's Torch environment."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('interim', Path(__file__).resolve().parents[1]/'tools/encoder/interim_compare.py')
interim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interim)


class InterimTests(unittest.TestCase):
    def fake(self):
        return SimpleNamespace(
            validate=lambda p: dict(adapt_seeds=[42,43,44], shards=8),
            checked=lambda p,n: dict(passed=True),
            items=lambda p,m: [dict(key=s,split=s) for s in ('train','val','test')],
            feature=lambda p,b,r: (p,b,r['key']),
            sha256=lambda p: 'hash', identity=lambda: {})

    def test_only_training_validation_and_redirected_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            study, out = Path(tmp)/'study', Path(tmp)/'analysis'
            c = self.fake()
            m, identity = interim.configure(c, study, out, 42)
            self.assertEqual(m['adapt_seeds'], [42])
            self.assertEqual([r['split'] for r in c.items(out,m)], ['train','val'])
            self.assertEqual(c.feature(out,'causal',dict(key='train'))[0],study.resolve())
            with self.assertRaisesRegex(ValueError,'test features'):
                c.feature(out,'causal',dict(key='test'))
            self.assertFalse(study.exists())

    def test_output_inside_study_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)/'study'
            with self.assertRaisesRegex(ValueError,'separate'):
                interim.configure(self.fake(),study,study/'interim',42)

    def test_incomplete_extraction_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.fake()
            c.checked = lambda p,n: dict(passed=False)
            with self.assertRaisesRegex(ValueError,'Incomplete extraction'):
                interim.configure(c,Path(tmp)/'study',Path(tmp)/'analysis',42)


if __name__ == '__main__':
    unittest.main()
