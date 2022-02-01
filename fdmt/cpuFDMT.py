#!/usr/bin/env python3

import sys
import numpy as np
from time import time
from attr import attrs, attrib, cmp_using

@attrs
class FDMT:
    fmin: float = attrib(default=400.1953125, help="Frequency of the lowest edge of the band, in MHz")
    fmax: float = attrib(default=800.1953125, help="Frequency of the highest edge of the band, in MHz")
    nchan: int = attrib(default=1024, help="Number of frequency channels")
    maxDT: int = attrib(default=2048, help="Number of time samples corresponding to the maximum delay between the top and bottom of the band (defines the maximum DM of the search)")
    A: np.ndarray = attrib(default=None, eq=cmp_using(eq=np.array_equal), help="A and B hold the states of successive iterations")
    B: np.ndarray = attrib(default=None, eq=cmp_using(eq=np.array_equal), help="A and B hold the states of successive iterations")
    df: float = attrib(init=False)
    fs: np.ndarray = attrib(init=False, eq=cmp_using(eq=np.array_equal))
    Q: list = attrib(init=False, eq=cmp_using(eq=np.array_equal), help="Q organizes memory locations of subtransforms for each iteration")


    def __attrs_post_init__(self):
        self.df = (self.fmax - self.fmin) / self.nchan
        self.fs = np.linspace(self.fmin+self.df/2, self.fmax-self.df/2, self.nchan, endpoint=True)


    def subDT(self, f, dF=self.df):
        "Get needed DT of subband to yield maxDT over entire band"
        loc = f**-2 - (f + dF) ** -2
        glo = self.fmin**-2 - self.fmax**-2
        return np.ceil((self.maxDT - 1) * loc / glo).astype(int) + 1


    def buildAB(self, numCols, dtype=np.uint32):
        numRowsA = (subDT(self.fs)).sum()
        numRowsB = (subDT(self.fs[::2], self.fs[2] - self.fs[0])).sum()
        self.A = np.zeros([numRowsA, numCols], dtype)
        self.B = np.zeros([numRowsB, numCols], dtype)


    def buildQ(self):
        self.Q = []
        for i in range(int(np.log2(self.nchan)) + 1):
            needed = subDT(self.fs[:: 2**i], self.df * 2**i)
            self.Q.append(np.cumsum(needed) - needed)


    def prep(self, cols, dtype=np.uint32):
        "Prepares necessary matrices for FDMT"
        buildAB(cols, dtype)
        buildQ()


    def fdmt(self, I, retDMT=False, verbose=False):
        """Computes DM Transform. If retDMT returns transform, else returns max sigma
        <Some info on requirements of I"""
        if I.dtype.itemsize < 4:
            I = I.astype(np.uint32)
        if self.A is None or self.A.shape[1] != I.shape[1] or self.A.dtype != I.dtype or True:
            prep(I.shape[1], I.dtype)

        t1 = time()
        fdmt_initialize(I)

        t2 = time()
        for i in range(1, int(np.log2(self.nchan)) + 1):
            src, dest = (self.A, self.B) if (i % 2 == 1) else (self.B, self.A)
            fdmt_iteration(src, dest, i)

        if verbose:
            t3 = time()
            print("Initializing time:  %.2f s" % (t2 - t1))
            print("Iterating time:  %.2f s" % (t3 - t2))
            print("Total time: %.2f s" % (t3 - t1))

        DMT = dest[:self.maxDT]
        if retDMT:
            return DMT
        noiseRMS = np.array([DMT[i, i:].std() for i in range(self.maxDT)])
        noiseMean = np.array([DMT[i, i:].mean() for i in range(self.maxDT)])
        sigmi = (DMT.T - noiseMean) / noiseRMS
        if verbose:
            print("Maximum sigma value: %.3f" % sigmi.max())
        return sigmi.max()


    def fdmt_initialize(self, I):
        self.A[self.Q[0], :] = I
        chDTs = subDT(self.fs)
        T = I.shape[1]
        commonDTs = [T for _ in range(1, chDTs.min())]
        DTsteps = list(np.where(chDTs[:-1] - chDTs[1:] != 0)[0])
        DTplan = commonDTs + DTsteps[::-1]
        for i, t in enumerate(DTplan, 1):
            self.A[self.Q[0][:t] + i, i:] = self.A[self.Q[0][:t] + i - 1, i:] + I[:t, :-i]
        for i, t in enumerate(DTplan, 1):
            # A[Q[0][:t]+i,i:] /= int(i+1)
            self.A[self.Q[0][:t] + i, i:] /= int(i + 1)


    def fdmt_iteration(self, src, dest, i):
        T = src.shape[1]
        dF = self.df * 2**i
        f_starts = self.fs[:: 2**i]
        f_ends = f_starts + dF
        f_mids = self.fs[2 ** (i - 1) :: 2**i]
        for i_F in range(self.nchan // 2**i):
            f0 = f_starts[i_F]
            f1 = f_mids[i_F]
            f2 = f_ends[i_F]
            cor = 0
            C = (f1**-2 - f0**-2) / (f2**-2 - f0**-2)
            C01 = ((f1 - cor) ** -2 - f0**-2) / (f2**-2 - f0**-2)
            C12 = ((f1 + cor) ** -2 - f0**-2) / (f2**-2 - f0**-2)
            for i_dT in range(subDT(f0, dF)):
                dT_mid01 = round(i_dT * C01)
                dT_mid12 = round(i_dT * C12)
                dT_rest = i_dT - dT_mid12
                dest[self.Q[i][i_F] + i_dT, :] = src[self.Q[i - 1][2 * i_F] + dT_mid01, :]
                dest[self.Q[i][i_F] + i_dT, dT_mid12:] += src[
                    self.Q[i - 1][2 * i_F + 1] + dT_rest, : T - dT_mid12
                ]


    def recursive_fdmt(self, I, depth=0, curMax=0):
        """Performs FDMT, downsamples and repeats recursively, returning max sigma
        <Some info on requirements for I>"""
        curMax = max(curMax, fdmt(I))
        if depth <= 0:
            return curMax
        else:
            I2 = (
                I[:, ::2] + I[:, 1::2]
                if (I.shape[1] % 2 == 0)
                else I[:, :-1:2] + I[:, 1::2]
            )
            return recursive_fdmt(I2, depth - 1, curMax)
