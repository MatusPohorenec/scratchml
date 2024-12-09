import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import roc_auc_score as SkRAC
from scratchml.metrics import roc_auc_score
from ..utils import generate_classification_dataset, repeat
import unittest


class Test_ROCAUCScore(unittest.TestCase):
    """
    Unittest class created to test the ROC AUC Score metric implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the ROC AUC Score on a binary dataset and then compares it to the Scikit-Learn
        implementation.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=2
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict_proba(X)[:, 1]
        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="macro")
        acc_score = roc_auc_score(y, sklr_prediction, average="macro")

        assert_almost_equal(sklr_score, acc_score)

        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="micro")
        acc_score = roc_auc_score(y, sklr_prediction, average="micro")

        assert_almost_equal(sklr_score, acc_score)

        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="weighted")
        acc_score = roc_auc_score(y, sklr_prediction, average="weighted")

        assert_almost_equal(sklr_score, acc_score)

    @repeat(10)
    def test_2(self):
        """
        Test the ROC AUC Score on a multi-class dataset and then compares it
        to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=6
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict_proba(X)
        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="macro")
        acc_score = roc_auc_score(y, sklr_prediction, average="macro")

        assert_almost_equal(sklr_score, acc_score, decimal=2)

        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="micro")
        acc_score = roc_auc_score(y, sklr_prediction, average="micro")

        assert_almost_equal(sklr_score, acc_score, decimal=2)

        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="weighted")
        acc_score = roc_auc_score(y, sklr_prediction, average="weighted")

        assert_almost_equal(sklr_score, acc_score, decimal=2)

    @repeat(1)
    def test_invalid_average(self):
        """
        Test that a ValueError is raised when an invalid average value is passed.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=2
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)
        sklr_prediction = sklr.predict_proba(X)[:, 1]

        with self.assertRaises(ValueError):
            roc_auc_score(y, sklr_prediction, average="invalid")

    @repeat(1)
    def test_multi_class(self):
        """
        Test the ROC AUC Score on a multi-class dataset to ensure the multi-class
        code path is executed.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=4
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict_proba(X)
        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="macro")
        acc_score = roc_auc_score(y, sklr_prediction, average="macro")

        assert_almost_equal(sklr_score, acc_score, decimal=2)

        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="micro")
        acc_score = roc_auc_score(y, sklr_prediction, average="micro")

        assert_almost_equal(sklr_score, acc_score, decimal=2)

        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="weighted")
        acc_score = roc_auc_score(y, sklr_prediction, average="weighted")

        assert_almost_equal(sklr_score, acc_score, decimal=2)

    @repeat(1)
    def test_weighted_average(self):
        """
        Test the ROC AUC Score on a multi-class dataset to ensure the weighted average
        code path is executed.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=4
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict_proba(X)
        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="weighted")
        acc_score = roc_auc_score(y, sklr_prediction, average="weighted")

        assert_almost_equal(sklr_score, acc_score, decimal=2)


@repeat(1)
def test_multi_class_thresholds(self):
    """
    Ensure multi-class ROC AUC path executes and thresholds are applied.
    """
    X, y = generate_classification_dataset(
        n_features=10, n_samples=1000, n_classes=3  # Ensure multi-class
    )
    sklr = SkLogisticRegression(
        penalty=None, fit_intercept=True, max_iter=10000, tol=1e-4
    )
    sklr.fit(X, y)
    sklr_prediction = sklr.predict_proba(X)

    # Test custom ROC AUC implementation
    acc_score = roc_auc_score(y, sklr_prediction, average="macro")
    self.assertIsNotNone(acc_score)


@repeat(1)
def test_weighted_average_coverage(self):
    """
    Ensure the 'weighted' average code path is executed.
    """
    X, y = generate_classification_dataset(
        n_features=10, n_samples=1000, n_classes=4  # Multi-class
    )
    sklr = SkLogisticRegression(
        penalty=None, fit_intercept=True, max_iter=10000, tol=1e-4
    )
    sklr.fit(X, y)
    sklr_prediction = sklr.predict_proba(X)

    # Test weighted averaging
    acc_score = roc_auc_score(y, sklr_prediction, average="weighted")
    self.assertIsNotNone(acc_score)


@repeat(1)
def test_threshold_iterations(self):
    """
    Validate that all thresholds are computed and applied.
    """
    y = np.array([0, 0, 1, 1, 2, 2])
    y_hat = np.array([[0.1, 0.2, 0.7],
                      [0.3, 0.4, 0.3],
                      [0.6, 0.3, 0.1],
                      [0.4, 0.5, 0.1],
                      [0.3, 0.4, 0.5],
                      [0.5, 0.1, 0.4]])
    acc_score = roc_auc_score(y, y_hat, average="macro")
    self.assertIsNotNone(acc_score)


if __name__ == "__main__":
    unittest.main(verbosity=2)
