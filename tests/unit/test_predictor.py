import unittest

from router.rl.predictor import BucketedOutputLengthPredictor


class PredictorTests(unittest.TestCase):
    def test_default_is_bounded(self):
        result = BucketedOutputLengthPredictor().predict("gsm8k", 100, 256)
        self.assertLessEqual(result.mean, 256)
        self.assertGreaterEqual(result.p90, result.mean)

    def test_observations_update_quantile(self):
        predictor = BucketedOutputLengthPredictor(alpha=1.0)
        for value in [10, 20, 30, 100]:
            predictor.observe("gsm8k", 100, 256, value)
        result = predictor.predict("gsm8k", 100, 256)
        self.assertEqual(result.mean, 100)
        self.assertEqual(result.p90, 100)

    def test_export_restore(self):
        predictor = BucketedOutputLengthPredictor()
        predictor.observe("gsm8k", 100, 256, 42)
        restored = BucketedOutputLengthPredictor.restore(predictor.export())
        self.assertEqual(restored.predict("gsm8k", 100, 256).samples, 1)


if __name__ == "__main__":
    unittest.main()
