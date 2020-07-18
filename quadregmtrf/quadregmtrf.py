import numpy as np

from sklearn.utils import check_consistent_length
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from scipy import linalg

import matplotlib.pyplot as plt


def _solve_quad_reg_cholesky(X, y, alpha, lambdas, Ms):
    A = safe_sparse_dot(X.T, X, dense_output=True)
    Xy = safe_sparse_dot(X.T, y, dense_output=True)

    A.flat[::X.shape[1] + 1] += alpha
    A += np.sum([lam * M for lam, M in zip(lambdas, Ms)])

    return linalg.solve(A, Xy, sym_pos=True, overwrite_a=True).T


class QuadRegMTRF(BaseEstimator):
    """Quadratically regularized multivariate temporal response function

    Takes multi-dimensional input `data`, a multi-dimensional stimulus
    of shape [time x dim1 x dim2 x ...] and response function `target`
    of shape [time x 1] or [time x channel]


    """
    def __init__(self, alpha, lambdas):
        """

        Parameters
        ----------
        alpha: float
            Parameter for ridge component of regularization
        lambdas: list of floats
            Parameters for quadratic components of regularization, which encourages
            smoothness along each dimension of the response function. The length of
            this list should be equal to the number of dimensions of data.
        """
        self.alpha = alpha
        self.lambdas = lambdas

    @staticmethod
    def _inds_to_M(inds):
        nelems = np.max(inds[:]) + 1
        M = np.zeros((nelems, nelems), dtype='int')
        for i, j in inds:
            M[i, i] += 1
            M[j, j] += 1
            M[i, j] -= 1
            M[j, i] -= 1
        return M

    def _get_Ms(self):

        Ms = []

        shape = self.rf_shape_
        nelems = shape[0] * shape[1]

        D = np.arange(nelems).reshape(shape)
        inds = np.hstack(list(zip(D.T, D.T[1:]))).T
        Ms.append(self._inds_to_M(inds))

        D = D.T
        inds = np.hstack(list(zip(D.T, D.T[1:]))).T
        Ms.append(self._inds_to_M(inds))

        return Ms

    def fit(self, data, targets, nt, nlag=None):
        """

        Parameters
        ----------
        data: array-like
            Multi-dimensional stimulus of dimensions [TIME x DIM1 x DIM2 x ...]
            This function will do the time lag and reshaping of this array. For
            instance, if this is a spectrogram input, the shape should be
            [TIME x FREQUENCY_BANDS]
        targets: array-like
            Neuronal response of one or multiple channels over time.
            Dimensions are [TIME x 1] or [TIME x CHANNELS]
        nt: int
            Depth of time window. This is the number of samples in time that will
            be used to create the stimulus window
        nlag: int, optional
            Defines the time difference in samples between the first time of the X window
            and the corresponding y.

            Default is that nlag = nt, meaning that the last sample of the window is at
            the time of the stimulus.

            When dealing with a stimulus, you will probably want to only include stimuli
            before the response in the regression, so nlag should be >= nt (unless you are
            doing a sanity check)

            This class can also be used to fit a data as a motor response. In this case,
            you will probably want data *after* the target times. To start the window at
            the time of the response, set nlag=0.

        Returns
        -------

        """

        if len(data.shape) != len(self.lambdas):
            raise ValueError('Length of lambdas should match the number of dimensions of data')

        check_consistent_length(data, targets)

        self.lagged_data = LaggedData(nt=nt, nlag=nlag)
        self.nfeats_ = data.shape[1]

        self.rf_shape_ = [nt, self.nfeats_]
        self.Ms_ = self._get_Ms()

        X = self.lagged_data.transform(data)

        y = targets[nlag:]
        self.offset_ = np.mean(y)
        y -= self.offset_  # remove need for intercept in regression

        self.coefs_ = _solve_quad_reg_cholesky(X, y, self.alpha, self.lambdas, self.Ms_)

        self.rf_ = self.coefs_.reshape(self.rf_shape_)

    def show_response_function(self, dt, cmap='RdBu_r', anchor_to_zero=True, ax=None):
        """Plot the response function

        Parameters
        ----------
        dt: float
            Sampling period in seconds
        cmap: str, optional
            name of colormap to use. Default: 'RdBu_r'
        anchor_to_zero: bool
            Make sure 0 is in the middle of the colormap. Only really makes sense for
            divergent colormaps. Default = True
        ax: plt.Axes

        Returns
        -------

        """

        if ax is None:
            fig, ax = plt.subplots()

        extent = (
            -dt * self.nlag,
            dt * (self.nt - self.nlag),
            0,
            self.nfeats_
        )

        kwargs = {}
        if anchor_to_zero:
            vmax = np.abs(self.rf_.ravel())
            kwargs.update(vmax=vmax, vmin=-vmax)

        return ax.imshow(self.rf_, cmap=cmap, extent=extent, aspect='auto', **kwargs)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return dict(alpha=self.alpha, lambdas=self.lambdas)

    def predict(self, data):

        check_is_fitted(self, 'coefs_')

        X = self.lagged_data.transform(data)

        return self.offset_ + np.dot(X, self.coefs_)

    @staticmethod
    def _more_tags():
        return dict(multioutput=True, stateless=True)


class LaggedData(TransformerMixin, BaseEstimator):
    def __init__(self, nt, *, nlag=None):
        if nlag is None:
            nlag = nt

        self.nt = nt
        self.nlag = nlag

        self.features_shape_ = None

    def fit(self, data):
        self.features_shape_ = data.shape[1:]

    def transform(self, data):

        if self.features_shape_ is not None:
            np.testing.assert_array_equal(
                data.shape[1:], self.features_shape_,
                "fit data shape does not match input data shape")

        data = data[:-self.nlag]
        X = np.vstack([data[i:i + self.nt].ravel()
                       for i in range(len(data) - self.nt + 1)])
        return X
