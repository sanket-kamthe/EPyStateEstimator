import unittest
import numpy as np
from MomentMatching.TimeSeriesModel import GaussianNoise

SEED = 12345


class TestNoiseModel(unittest.TestCase):
    def test_defaults(self):
        eta = GaussianNoise()
        np.testing.assert_equal(eta.dimension, 1)
        np.testing.assert_allclose(eta.params.Q, np.eye(1))
        np.testing.assert_allclose(eta.mean, 0.0)
        np.testing.assert_allclose(eta.cov, np.eye(1))

    def test_1drandom_sample(self):
        eta = GaussianNoise()

        np.random.seed(seed=SEED)
        actual = eta.sample()

        np.random.seed(seed=SEED)
        desired = np.random.multivariate_normal(mean=np.zeros((1,), dtype=float),
                                                cov=np.eye(1))

        np.testing.assert_allclose(actual, desired)



if __name__ == '__main__':
    unittest.main()
