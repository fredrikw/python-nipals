import pandas as pd
import logging
# logging.basicConfig(level=logging.INFO)
import math


class Nipals(object):
    """A Nipals class that can be used for PCA.

    Initialize with a Pandas DataFrame or an object that can be turned into a DataFrame
    (e.g. an array or a dict of lists)"""
    def __init__(self, x_df):
        super(Nipals, self).__init__()
        if type(x_df) != pd.core.frame.DataFrame:
            x_df = pd.DataFrame(x_df)
        self.x_df = x_df

    def fit(
            self,
            ncomp=None,
            tol=0.000001,
            center=True,
            scale=True,
            maxiter=500,
            startcol=None,
            gramschmidt=False
        ):
        """The Fit method, will fit a PCA to the X data.

        Keyword arguments:
        ncomp - number of components, defaults to all
        tol - tolerance for convergence checking, defaults to 1E-6
        center - whether to center the data, defaults to True
        scale - whether to scale the data, defaults to True
        maxiter - maximum number of iterations before convergence is considered failed, defaults to 500
        startcol - column in X data to start iteration from, if set to None, the column with maximal variance is selected, defaults to None
        gramschmidt - wheter to run Gram-Schmidt orthogonalization, defaults to False. Not implemented!"""
        if gramschmidt:
            raise NotImplementedError
        if ncomp is None:
            ncomp = min(self.x_df.shape)
        elif ncomp > min(self.x_df.shape):
            ncomp = min(self.x_df.shape)
            logging.warning(
                'ncomp is larger than the max dimension of the x matrix.\n'
                'fit will only return {} components'.format(ncomp)
            )
        #Convert to np array
        self.x_mat = self.x_df.values
        if center:
            self.x_mean = pd.np.nanmean(self.x_mat, axis=0)
            self.x_mat = self.x_mat - self.x_mean
        if scale:
            self.x_std = pd.np.nanstd(self.x_mat, axis=0, ddof=1)
            self.x_mat = self.x_mat / self.x_std

        TotalSS = pd.np.nansum(self.x_mat*self.x_mat)
        nr, nc = self.x_mat.shape
        # initialize outputs
        PPp = pd.np.zeros((nc, nc))
        TTp = pd.np.zeros((nr, nr))
        eig = pd.np.empty((ncomp,))
        R2cum = pd.np.empty((ncomp,))
        loadings = pd.np.empty((nc, ncomp))
        scores = pd.np.empty((nr, ncomp))

        #NA handling
        x_miss = pd.np.isnan(self.x_mat)
        hasna = x_miss.any()
        if hasna:
            logging.info("Data has NA values")

        # self.x_mat_0 = pd.np.nan_to_num(self.x_mat)
        # t = [None] * ncomp
        # p = [None] * ncomp
        self.eig = []
        for comp in range(ncomp):
            #Set t to first column of X
            if startcol is None:
                xvar = pd.np.nanvar(self.x_mat, axis=0, ddof=1)
                startcol_use = pd.np.where(xvar == xvar.max())[0][0]
            else:
                startcol_use = startcol
            logging.info("PC {}, starting with column {}".format(comp, startcol_use))

            if hasna:
                self.x_mat_0 = pd.np.nan_to_num(self.x_mat)
                th = self.x_mat_0[:,startcol_use]
            else:
                th = self.x_mat[:,startcol_use]
            it = 0
            while True:
                # loadings
                if hasna:
                    T2 = pd.np.repeat(th*th, nc)
                    T2.shape = (nr, nc)
                    T2[x_miss] = 0
                    ph = self.x_mat_0.T.dot(th) / T2.sum(axis=0)
                else:
                    ph = self.x_mat.T.dot(th) / sum(th*th)
                # Gram Schmidt
                if gramschmidt and comp > 0:
                    #ph <- ph - PPp %*% ph
                    pass
                # Normalize
                ph = ph / math.sqrt(pd.np.nansum(ph*ph))

                # Scores
                th_old = th
                if hasna:
                    P2 = pd.np.repeat(ph*ph, nr)
                    P2.shape = (nc, nr)
                    P2[x_miss.T] = 0
                    th = self.x_mat_0.dot(ph) / P2.sum(axis=0)
                else:
                    th = self.x_mat.dot(ph) / sum(ph*ph)
                # Gram Schmidt
                if gramschmidt and comp > 0:
                    #th <- th - TTp %*% th
                    pass

                # Check convergence
                if pd.np.nansum((th-th_old)**2) < tol:
                    break
                it += 1
                if it >= maxiter:
                    raise RuntimeError(
                        "Convergence was not reached in {} iterations for component {}".format(
                            maxiter, comp
                        )
                    )

            # Update X
            self.x_mat = self.x_mat - pd.np.outer(th, ph)
            loadings[:,comp] = ph
            scores[:,comp] = th
            eig[comp] = pd.np.nansum(th*th)

            # Update (Ph)(Ph)' and (Th)(Th)' for next PC
            if gramschmidt:
                pass
            #   PPp = PPp + tcrossprod(ph)
            #   TTp = TTp + tcrossprod(th) / eig[h]
            # }

            # Cumulative proportion of variance explained
            R2cum[comp] <- 1 - (pd.np.nansum(self.x_mat*self.x_mat) / TotalSS)

        # "Uncumulate" R2
        self.R2 = pd.np.insert(pd.np.diff(R2cum), 0, R2cum[0])

        # Finalize eigenvalues and subtract from scores
        self.eig = pd.Series(pd.np.sqrt(eig))
        scores = scores / eig

        # Convert results to DataFrames
        self.scores = pd.DataFrame(scores, index=self.x_df.index, columns=["PC{}".format(i+1) for i in range(ncomp)])
        self.loadings = pd.DataFrame(loadings, index=self.x_df.columns, columns=["PC{}".format(i+1) for i in range(ncomp)])
        return True
