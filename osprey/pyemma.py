from sklearn.base import clone as skl_clone
from pyemma._ext.sklearn.base import clone as pym_clone
import numpy as np


def clone(estimator, safe=True):
    """
    Dispatches either scikit-learn's clone or pyemma's version
    If the estimator is a list then it checks whether any of the elements are PyEMMA classes.

    :param estimator: the estimator being cloned
    :param safe: argument for the clone function
    :return: cloned estimator
    """
    estimator_type = type(estimator)
    if estimator_type in (list, tuple, set, frozenset):
        is_pyemma = np.any([x.__module__.startswith('pyemma') for x in estimator])
    else:
        is_pyemma = estimator.__module__.startswith('pyemma')
    if is_pyemma:
        return pym_clone(estimator, safe=safe)
    else:
        return skl_clone(estimator, safe=safe)


