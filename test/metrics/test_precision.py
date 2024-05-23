import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import precision_score
from scratchml.metrics import precision
from test.utils import generate_classification_dataset, repeat

class Test_Precision(unittest.TestCase):
    @repeat(10)
    def test_1(self):
        X, y = generate_classification_dataset(
            n_features=10,
            n_samples=10000,
            n_classes=2
        )

        sklr = SkLogisticRegression(
            penalty=None,
            fit_intercept=True,
            max_iter=1000000,
            tol=1e-4
        )

        sklr.fit(X, y)
        
        sklr_prediction = sklr.predict(X)
        sklr_score = precision_score(y, sklr_prediction)
        acc_score = precision(y, sklr_prediction)

        assert np.abs(acc_score - sklr_score) < 0.1

if __name__ == "__main__":
    unittest.main(verbosity=2)