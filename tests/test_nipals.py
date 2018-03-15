import logging

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
