# Implementation provenance and third-party dependencies

The distributed neural modules are local implementations of ideas described in:

- Challu et al., *N-HiTS: Neural Hierarchical Interpolation for Time Series
  Forecasting*, AAAI 2023, DOI 10.1609/aaai.v37i6.25854.
- Wang et al., *TimeMixer: Decomposable Multiscale Mixing for Time Series
  Forecasting*, ICLR 2024, https://openreview.net/forum?id=7oLshfEIC2.
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*,
  NeurIPS 2017.

The local files import PyTorch and NumPy, and do not import or vendor the
authors' N-HiTS/TimeMixer libraries. Their code headers describe local
adaptations. No third-party repository source, checkpoint or dataset is copied
by this code packaging step. No separate license file was found in the local
research source directory. The researchers' own code in this release is provided
under the root MIT license. This license does not grant rights over third-party
data or modify dependency licenses.

Runtime dependencies are installed from their own distributions:

- LightGBM: MIT, https://github.com/microsoft/LightGBM/blob/master/LICENSE
- PyTorch: BSD-style, https://github.com/pytorch/pytorch/blob/main/LICENSE
- NumPy: BSD-3-Clause, https://github.com/numpy/numpy/blob/main/LICENSE.txt
- SciPy: BSD-3-Clause, https://github.com/scipy/scipy/blob/main/LICENSE.txt
- pandas: BSD-3-Clause, https://github.com/pandas-dev/pandas/blob/main/LICENSE
- pvlib-python: BSD-3-Clause, https://github.com/pvlib/pvlib-python/blob/main/LICENSE

Dependency license files remain with their installed packages. Source hashes
and packaging changes are listed in `SOURCE_PROVENANCE.json`.
