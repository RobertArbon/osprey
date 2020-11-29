from __future__ import print_function, absolute_import, division

import numpy as np
from nose.plugins.skip import SkipTest
from six import iteritems
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from osprey.fit_estimator import fit_and_score_estimator

try:
    __import__('msmbuilder.msm')
    HAVE_MSMBUILDER = True
except:
    HAVE_MSMBUILDER = False

try:
    __import__('pyemma.msm')
    HAVE_PYEMMA = True
except:
    HAVE_PYEMMA = False

# TODO remove compat with py<=3.6
try:
    from numpy.testing import dec
    skipif = dec.skipif
except ModuleNotFoundError:
    from numpy.testing.decorators import skipif

def test_1():
    X, y = make_regression(n_samples=300, n_features=10)

    lasso = Lasso()
    params = {'alpha': 2}
    cv = 6
    out = fit_and_score_estimator(lasso, params, cv=cv, X=X, y=y, verbose=0)

    param_grid = dict((k, [v]) for k, v in iteritems(params))
    g = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=cv)
    g.fit(X, y)

    np.testing.assert_almost_equal(out['mean_test_score'],
                                   g.cv_results_['mean_test_score'][0])

    test_scores = np.hstack(
        [g.cv_results_['split{}_test_score'.format(i)] for i in range(cv)])
    assert np.all(out['test_scores'] == test_scores)

@skipif(not HAVE_MSMBUILDER, "this test requires MSMBuilder")
def test_2():
    from msmbuilder.msm import MarkovStateModel

    X = [np.random.randint(2, size=10), np.random.randint(2, size=11)]
    out = fit_and_score_estimator(MarkovStateModel(), {'verbose': False},
                                  cv=2,
                                  X=X,
                                  y=None,
                                  verbose=0)
    np.testing.assert_array_equal(out['n_train_samples'], [11, 10])
    np.testing.assert_array_equal(out['n_test_samples'], [10, 11])


@skipif(not HAVE_PYEMMA, "this test requires PyEMMA")
def test_3():
    from pyemma.msm import MaximumLikelihoodMSM

    X = [np.random.randint(2, size=10), np.random.randint(2, size=11)]
    out = fit_and_score_estimator(MaximumLikelihoodMSM(), {'score_k': 2},
                                  cv=2,
                                  X=X,
                                  y=None,
                                  verbose=0)
    np.testing.assert_array_equal(out['n_train_samples'], [11, 10])
    np.testing.assert_array_equal(out['n_test_samples'], [10, 11])
