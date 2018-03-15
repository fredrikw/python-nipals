========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/python-nipals/badge/?style=flat
    :target: https://readthedocs.org/projects/python-nipals
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/fredrikw/python-nipals.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/fredrikw/python-nipals

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/fredrikw/python-nipals?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/fredrikw/python-nipals

.. |requires| image:: https://requires.io/github/fredrikw/python-nipals/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/fredrikw/python-nipals/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/fredrikw/python-nipals/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/fredrikw/python-nipals

.. |version| image:: https://img.shields.io/pypi/v/nipals.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/nipals

.. |commits-since| image:: https://img.shields.io/github/commits-since/fredrikw/python-nipals/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/fredrikw/python-nipals/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/nipals.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/nipals

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/nipals.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/nipals

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/nipals.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/nipals


.. end-badges

A module for calculation of PCA with the NIPALS algorithm. Based on the R package
`nipals <https://cran.r-project.org/package=nipals>`_.
Tested to give same results as nipals:nipals, with some rounding errors.

Please note that the Gram-Schmidt orthogonalization has not yet been implemented.

* Free software: MIT license

Installation
============

::

    pip install nipals

Documentation
=============

https://python-nipals.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
