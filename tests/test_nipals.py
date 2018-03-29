import logging

import matplotlib

import pandas as pd
import pytest

from nipals import nipals

testdata = [
    [pd.np.nan, 67, 90, 98, 120],
    [pd.np.nan, 71, 93, 102, 129],
    [65, 76, 95, 105, 134],
    [50, 80, 102, 130, 138],
    [60, 82, 97, 135, 151],
    [65, 89, 106, 137, 153],
    [75, 95, 117, 133, 155]
]

testdata_full = [
    [50, 67, 90, 98, 120],
    [55, 71, 93, 102, 129],
    [65, 76, 95, 105, 134],
    [50, 80, 102, 130, 138],
    [60, 82, 97, 135, 151],
    [65, 89, 106, 137, 153],
    [75, 95, 117, 133, 155]
]

testdata_class = pd.DataFrame(testdata)
testdata_class['C1'] = [0, 0, 1, 1, 1, 0, 0]
testdata_class['C2'] = [1, 1, 0, 0, 0, 1, 1]
testdata_class = testdata_class.set_index(['C1', 'C2'], append=True)

yarn = pd.read_csv("tests/yarn.csv", index_col=0)
yarn_scores = pd.read_csv("tests/yarn_scores.csv", index_col=0).values
yarn_loadings = pd.read_csv("tests/yarn_loadings.csv", index_col=0).values
yarn_weights = pd.read_csv("tests/yarn_weights.csv", index_col=0).values
yarn_missing_scores = pd.read_csv("tests/yarn_missing_scores.csv", index_col=0, usecols=[0,2,3,4]).values
yarn_missing_loadings = pd.read_csv("tests/yarn_missing_loadings.csv", index_col=0).values
yarn_missing_weights = pd.read_csv("tests/yarn_missing_weights.csv", index_col=0).values

oliveoil = pd.read_csv("tests/oliveoil.csv", index_col=0)
oliveoil_scores = pd.read_csv("tests/oliveoil_scores.csv", index_col=0)
oliveoil_missing_y = pd.read_csv("tests/oliveoil_miss.csv", index_col=0)
oliveoil_missing_y_scores = pd.read_csv("tests/oliveoil_miss_scores.csv", index_col=0)

def test_init_from_df():
    # It should be possible to init a nipals class from a Pandas DataFrame
    assert nipals.Nipals(pd.DataFrame(testdata)).__class__ == nipals.Nipals


def test_init_from_data():
    # It should be possible to init a nipals class from something that
    # can be made into a Pandas DataFrame
    assert nipals.Nipals(testdata).__class__ == nipals.Nipals


def test_run_pca():
    nip = nipals.Nipals(testdata)
    # Set startcol to get same results as R nipals:nipals (rounding errors give slightly different rotation otherwise)
    assert nip.fit(startcol=1)
    pd.np.testing.assert_almost_equal(list(nip.eig), [
        4.876206673116805,
        2.044242214135278,
        1.072810327696312,
        0.23701331933712338,
        0.14325051166784325
    ])
    # Also run without startcol set to make sure that it works as well. But compare to self data
    assert nip.fit()
    pd.np.testing.assert_almost_equal(list(nip.eig), [
        4.876216689582536,
        2.044275687396918,
        1.072805497059184,
        0.23696073749622645,
        0.14327789003413574
    ])


def test_run_pca_gramschmidt():
    nip = nipals.Nipals(testdata)
    with pytest.raises(NotImplementedError):
        nip.fit(gramschmidt=True)


def test_call_with_too_large_ncomp(caplog):
    nip = nipals.Nipals(testdata)
    assert nip.fit(ncomp=10)
    assert caplog.record_tuples == [
        (
            'root',
            logging.WARNING,
            'ncomp is larger than the max dimension of the x matrix.\n'
            'fit will only return 5 components'
        ),
    ]


def test_run_pca_without_na():
    nip = nipals.Nipals(testdata_full)
    assert nip.fit()
    pd.np.testing.assert_almost_equal(list(nip.eig), [
        5.020518433605382,
        1.879323465996815,
        1.1081766447275905,
        0.17225187199265019,
        0.06936702860594454
    ])


def test_fail_from_maxiter():
    nip = nipals.Nipals(testdata_full)
    with pytest.raises(RuntimeError):
        nip.fit(maxiter=1)


def test_run_pca_with_set_ncomp():
    nip = nipals.Nipals(testdata_full)
    assert nip.fit(ncomp=2)
    pd.np.testing.assert_almost_equal(list(nip.eig), [5.020518433605, 1.879323465996])


def test_run_pca_with_precentered_data():
    centered = pd.DataFrame(testdata_full)
    centered = centered - centered.mean()
    nip = nipals.Nipals(centered)
    assert nip.fit(center=False, ncomp=2)
    pd.np.testing.assert_almost_equal(list(nip.eig), [5.020518433605, 1.879323465996])


def test_run_pca_with_prescaled_data():
    scaled = pd.DataFrame(testdata_full)
    scaled = (scaled - scaled.mean()) / scaled.std(ddof=1)
    nip = nipals.Nipals(scaled)
    assert nip.fit(center=False, scale=False, ncomp=2)
    pd.np.testing.assert_almost_equal(list(nip.eig), [5.020518433605, 1.879323465996])

def test_run_pca_check_scores_with_sweep():
    nip = nipals.Nipals(testdata)
    assert nip.fit(ncomp=2, eigsweep=True, startcol=1)
    pd.np.testing.assert_almost_equal(nip.scores.values, [
        [-0.5585132, 0.1224190],
        [-0.3801627, 0.1703718],
        [-0.2026926, 0.3163937],
        [-0.0337608, -0.6915786],
        [0.1285239, -0.4501362],
        [0.3562110, -0.2048250],
        [0.5982563, 0.3647261],
    ], 3)

def test_run_pca_check_scores():
    nip = nipals.Nipals(testdata)
    assert nip.fit(ncomp=2)
    pd.np.testing.assert_almost_equal(nip.scores.values, [
        [-2.72332498,  0.25021637],
        [-1.85369271,  0.34827344],
        [-0.98854112,  0.64658032],
        [-0.16429534, -1.41375825],
        [ 0.62686116, -0.92008715],
        [ 1.7370958, -0.41837244],
        [ 2.91721621,  0.746181]
    ])

def test_predict_from_pca():
    nip = nipals.Nipals(testdata)
    nip.fit(ncomp=2)
    assert nip.predict(pd.DataFrame([[63, 70, 98, 110, 124],
        [51, 82, 102, 110, 108]]))
    pd.np.testing.assert_almost_equal(nip.pred.values, [[-1.4465766,  0.4500705],
        [-1.6229739, -0.340578 ]])

def test_predict_from_pca_with_sweep():
    nip = nipals.Nipals(testdata)
    nip.fit(ncomp=2, eigsweep=True)
    assert nip.predict(pd.DataFrame([[63, 70, 98, 110, 124],
        [51, 82, 102, 110, 108]]))
    pd.np.testing.assert_almost_equal(nip.pred.values, [[-0.2966596,  0.2201614],
        [-0.3328347, -0.1666008]])

def test_plot():
    nip = nipals.Nipals(testdata)
    nip.fit(ncomp=2)
    plt = nip.plot()
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_plot_classes():
    nip = nipals.Nipals(testdata_class)
    nip.fit(ncomp=2)
    plt = nip.plot(classlevels=['C1','C2'])
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_plot_classoptions():
    nip = nipals.Nipals(testdata_class)
    nip.fit(ncomp=2)
    plt = nip.plot(classlevels=['C1','C2'], markers=['s', 'o'], classcolors=['red', 'black'])
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_pred_plot():
    nip = nipals.Nipals(testdata_class)
    nip.fit(ncomp=2)
    nip.predict(pd.DataFrame([[63, 70, 98, 110, 124],
        [51, 82, 102, 110, 108]]))
    plt = nip.plot(classlevels=['C1','C2'], plotpred=True)
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_plot_labels():
    nip = nipals.Nipals(testdata_class)
    nip.fit(ncomp=2)
    nip.predict(pd.DataFrame([[63, 70, 98, 110, 124],
        [51, 82, 102, 110, 108]], index=['a', 'b']))
    plt = nip.plot(classlevels=['C1','C2'], plotpred=True, labels='C1', predlabels=0)
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_pred_plot_labels():
    nip = nipals.Nipals(testdata_class)
    nip.fit(ncomp=2)
    nip.predict(pd.DataFrame([
        [63, 70, 98, 110, 124, 0, 1, 0],
        [51, 82, 102, 110, 108, 1, 0, 0]
    ], index=['a', 'b']).set_index([5,6,7], append=True))
    plt = nip.plot(classlevels=['C1','C2'], plotpred=True, labels='C1', predlevels=[5,6], predlabels=0)
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_pred_plot_options():
    nip = nipals.Nipals(testdata_class)
    nip.fit(ncomp=2)
    nip.predict(pd.DataFrame([
        [63, 70, 98, 110, 124, 0, 1, 0],
        [51, 82, 102, 110, 108, 1, 0, 0]
    ]).set_index([5,6,7], append=True))
    with pytest.raises(KeyError):
        plt = nip.plot(classlevels=['C1','C2'], plotpred=True, predlevels=[5,7])
    plt = nip.plot(classlevels=['C1','C2'], plotpred=True, predlevels=[5,6])
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_loadings_plot():
    nip = nipals.Nipals(testdata_class)
    nip.fit(ncomp=2)
    plt = nip.loadingsplot()
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_pls():
    pls = nipals.PLS(yarn.iloc[:,:268], yarn.iloc[:, 268])
    assert pls.fit(ncomp=6, scale=False)
    pd.np.testing.assert_almost_equal(pls.scores.values, yarn_scores)
    pd.np.testing.assert_almost_equal(pls.loadings.values, yarn_loadings)
    pd.np.testing.assert_almost_equal(pls.weights.values, yarn_weights)

def test_pls_nocenter_and_scale():
    pls = nipals.PLS(yarn.iloc[:,:268] - yarn.iloc[:,:268].mean(), yarn.iloc[:, 268])
    assert pls.fit(ncomp=6, scale=False, center=False)
    pd.np.testing.assert_almost_equal(pls.scores.values, yarn_scores)
    pd.np.testing.assert_almost_equal(pls.loadings.values, yarn_loadings)
    pd.np.testing.assert_almost_equal(pls.weights.values, yarn_weights)

def test_pls_missing_x():
    tmpx = yarn.iloc[:,:268]
    csel = [171, 107, 222, 150, 76, 205, 63, 19, 121, 183]
    rsel = [23, 0, 3, 22, 15, 21, 19, 7, 19, 5]
    for r, c in zip(rsel, csel):
        tmpx.iloc[r,c] = pd.np.nan
    pls = nipals.PLS(tmpx, yarn.iloc[:, 268])
    assert pls.fit(ncomp=3)
    pd.np.testing.assert_almost_equal(pls.scores.values, yarn_missing_scores, 5)
    pd.np.testing.assert_almost_equal(pls.loadings.values, yarn_missing_loadings)
    pd.np.testing.assert_almost_equal(pls.weights.values, yarn_missing_weights)

def test_pls_nondf():
    pls = nipals.PLS(yarn.iloc[:,:268].values, yarn.iloc[:, 268].values)
    assert pls.fit(ncomp=6, scale=False)
    pd.np.testing.assert_almost_equal(pls.scores.values, yarn_scores)
    pd.np.testing.assert_almost_equal(pls.loadings.values, yarn_loadings)
    pd.np.testing.assert_almost_equal(pls.weights.values, yarn_weights)

def test_pls_optionvariations(caplog):
    pls = nipals.PLS(yarn.iloc[:,:268], yarn.iloc[:, 268])
    assert pls.fit()
    assert pls.fit(ncomp=500, startcol=0)
    assert caplog.record_tuples == [
        (
            'root',
            logging.WARNING,
            'ncomp is larger than the max dimension of the x matrix.\n'
            'fit will only return 28 components'
        ),
    ]
    with pytest.raises(RuntimeError):
        pls.fit(maxiter=1)

def test_pls_multiy():
    pls = nipals.PLS(oliveoil.iloc[:, :5], oliveoil.iloc[:, 5:])
    assert pls.fit(ncomp=2)
    pd.np.testing.assert_almost_equal(pls.scores.values, oliveoil_scores * [-1, 1], 4)

def test_pls_missing_y():
    pls = nipals.PLS(oliveoil_missing_y.iloc[:, :5], oliveoil_missing_y.iloc[:, 5:])
    assert pls.fit(ncomp=2)
    pd.np.testing.assert_almost_equal(pls.scores.values, oliveoil_missing_y_scores * [-1, 1], 3)

def test_pls_score_plot():
    pls = nipals.PLS(oliveoil_missing_y.iloc[:, :5], oliveoil_missing_y.iloc[:, 5:])
    assert pls.fit(ncomp=2)
    plt = pls.plot()
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_pls_load_plot():
    pls = nipals.PLS(oliveoil_missing_y.iloc[:, :5], oliveoil_missing_y.iloc[:, 5:])
    assert pls.fit(ncomp=2)
    plt = pls.loadingsplot()
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_pls_load_plot_no_y():
    pls = nipals.PLS(oliveoil_missing_y.iloc[:, :5], oliveoil_missing_y.iloc[:, 5:])
    assert pls.fit(ncomp=2)
    plt = pls.loadingsplot(showweights=False)
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt

def test_pls_load_plot_without_labels():
    pls = nipals.PLS(oliveoil_missing_y.iloc[:, :5], oliveoil_missing_y.iloc[:, 5:])
    assert pls.fit(ncomp=2)
    plt = pls.loadingsplot(labels=False)
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt


def test_pls_load_plot_with_markers():
    pls = nipals.PLS(oliveoil_missing_y.iloc[:, :5], oliveoil_missing_y.iloc[:, 5:])
    assert pls.fit(ncomp=2)
    plt = pls.loadingsplot(weightmarkers=['s', 'o', 'v', '^', '+', '3'])
    assert isinstance(plt, matplotlib.figure.Figure)
    return plt


def test_inf_values_in_dfs(caplog):
    olive_inf = oliveoil.replace([18.7, 75.1], pd.np.inf)
    nipals.Nipals(olive_inf)
    nipals.PLS(olive_inf.iloc[:, :5], olive_inf.iloc[:, 5:])
    assert caplog.record_tuples == [
        (
            'root',
            logging.WARNING,
            'Data contained infinite values, converting to missing values'
        ),
        (
            'root',
            logging.WARNING,
            'X data contained infinite values, converting to missing values'
        ),
        (
            'root',
            logging.WARNING,
            'Y data contained infinite values, converting to missing values'
        ),
    ]
