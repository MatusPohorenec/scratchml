from sklearn.metrics import max_error as SkME
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scratchml.metrics import max_error
from ..utils import generate_regression_dataset, repeat
import unittest
import numpy as np


class Test_MaxError(unittest.TestCase):
    """
    Unittest class created to test the Max Error metric implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the Max Error and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=10000, n_features=10, n_targets=1)

        sklr = SkLinearRegression()

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)

        sklr_score = SkME(y, sklr_prediction)
        score = max_error(y, sklr_prediction, derivative=False)

        assert np.abs(score - sklr_score) < 0.1

    @repeat(1)
    def test_2(self):
        """
        Test that the NotImplementedError is raised when derivative=True.
        """
        X, y = generate_regression_dataset(n_samples=100, n_features=10, n_targets=1)
        y_hat = np.random.rand(100, 1)

        with self.assertRaises(NotImplementedError):
            max_error(y, y_hat, derivative=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
