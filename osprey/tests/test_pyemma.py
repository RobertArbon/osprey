from __future__ import print_function, absolute_import, division
# try:
#     from numpy.testing import dec
#     skipif = dec.skipif
# except ModuleNotFoundError:
#     from numpy.testing.decorators import skipif
# try:
#     import pyemma
#     HAVE_PYEMMA = True
# except ImportError:
#     HAVE_PYEMMA = False
from osprey.pyemma import clone


def test_clone_pyemma():
    # Test it creates a new instance of a pyemma class
    from pyemma.coordinates.transform import TICA
    estimator_old = TICA(lag=1)
    params_old = estimator_old.get_params()
    estimator_new = clone(estimator_old)
    params_new = estimator_new.get_params()
    assert isinstance(estimator_new, type(estimator_old)) & (estimator_new is not estimator_old)


def test_clone_sklearn():
    # Test it creates a new instance of a sklearn class
    from sklearn.cluster import KMeans
    estimator_old = KMeans()
    params_old = estimator_old.get_params()
    estimator_new = clone(estimator_old)
    params_new = estimator_new.get_params()
    assert isinstance(estimator_new, type(estimator_old)) & (estimator_new is not estimator_old)

