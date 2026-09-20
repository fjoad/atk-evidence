"""Direct implementations of the paper's detectors, added as they are tested."""

import sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC


def random_forest(seed: int = 20260920, workers: int = 4) -> RandomForestClassifier:
    """Section III-C, p.2679: 100 estimators; remaining choices are explicit."""
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=workers,
        random_state=seed,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    )


def adaboost(seed: int = 20260920) -> AdaBoostClassifier:
    """III-B.2(b)/III-C: 50 trees; historical defaults declared in ADABOOST_PILOT."""
    if sklearn.__version__ != "1.5.2":
        raise RuntimeError("SAMME.R requires the pinned AdaBoost scikit-learn 1.5.2 environment")
    return AdaBoostClassifier(
        estimator=None,  # Stock depth-one decision tree; fitted parameters saved.
        n_estimators=50, learning_rate=1.0, algorithm="SAMME.R", random_state=seed,
    )


def svm(seed: int = 20260920) -> SVC:
    """III-C p.2679: C=1, sigmoid; omissions fixed in SVM_PILOT.md."""
    return SVC(C=1.0, kernel="sigmoid", degree=3, gamma="scale", coef0=0.0,
               shrinking=True, probability=False, tol=0.001, cache_size=200,
               class_weight=None, verbose=False, max_iter=-1,
               decision_function_shape="ovr", break_ties=False, random_state=seed)
