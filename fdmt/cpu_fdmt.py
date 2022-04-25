#!/usr/bin/env python3

from re import S
import sys
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
import numpy as np
from time import time
from attr import attrs, attrib, cmp_using

@attrs
class FDMT:
    '''
    Collection of attributes and helper arrays necessary for dispersion using
    FDMT.

    Parameters
    ---------
    fmin: float
        Frequency of the lowest edge of the band, in MHz
    fmax: float
        Frequency of the highest edge of the band, in MHz
    nchan: int
        Number of frequency channels
    maxDT: int
        Number of time samples corresponding to the maximum delay between the
        top and bottom of the band (defines the maximum DM of the search)
    '''
    fmin: float = attrib(default=400.1953125)
    fmax: float = attrib(default=800.1953125)
    nchan: int = attrib(default=1024)
    maxDT: int = attrib(default=2048)
    A: np.ndarray = attrib(default=None, eq=cmp_using(eq=np.array_equal))
    B: np.ndarray = attrib(default=None, eq=cmp_using(eq=np.array_equal))
    df: float = attrib(init=False)
    fs: np.ndarray = attrib(init=False, eq=cmp_using(eq=np.array_equal))
    Q: list = attrib(init=False, eq=cmp_using(eq=np.array_equal))


    def __attrs_post_init__(self):
        self.fs, self.df = fs,df = np.linspace(self.fmin, self.fmax, self.nchan, endpoint=False,retstep=True)


    def subDT(self, f, dF=None):
        "Get needed DT of subband to yield maxDT over entire band"
        if dF is None:
            dF = self.df
        loc = f**-2 - (f + dF) ** -2
        glo = self.fmin**-2 - self.fmax**-2
        return np.ceil((self.maxDT - 1) * loc / glo).astype(int) + 1


    def buildAB(self, numCols, dtype=np.uint32):
        numRowsA = (self.subDT(self.fs)).sum()
        numRowsB = (self.subDT(self.fs[::2], self.fs[2] - self.fs[0])).sum()
        self.A = np.zeros([numRowsA, numCols], dtype)
        self.B = np.zeros([numRowsB, numCols], dtype)


    def buildQ(self):
        self.Q = []
        for i in range(int(np.log2(self.nchan)) + 1):
            needed = self.subDT(self.fs[:: 2**i], self.df * 2**i)
            self.Q.append(np.cumsum(needed) - needed)


    def prep(self, cols, dtype=np.uint32):
        "Prepares necessary matrices for FDMT"
        self.buildAB(cols, dtype=dtype)
        self.buildQ()


    def fdmt(self, I, retDMT=False, verbose=False, padding=False,
             frontpadding=True):
        """Computes DM Transform. If retDMT returns transform, else returns max
        sigma.
        
        Parameters
        ==========
        I: np.ndarray
            The array of spectra to be dedispersed. Must have shape (nchan, nsamp)
            where nchan is the number of frequency channels and nsamp is the number
            of time samples. The 0th channel must be the lowest frequency channel.
        verbose: bool
            Whether to print info to stdout.
        padding: bool
            Whether to return additional, incompletely integrated time samples
            at the end of the dedispersed time series.
        frontpadding: bool
            Whether to return additional, incompletely integrated time samples
            at the beginning of the dedispersed time series.
        retDMT: bool
            Whether to return the DM transform instead of the SNRs.

        Returns
        =======
        dedisp: np.ndarray
            An array of dedispersed time series. Has shape
            (n_dms, nsamp_dedisp), where nsamp_dedisp is given as follows:
                - if padding and frontpadding:
                    nsamp_dedisp = nsamp + maxDT
                - if padding and not frontpadding:
                    nsamp_dedisp = nsamp
                - if not padding and frontpadding:
                    nsamp_dedisp = nsamp
                - if not padding and not frontpadding:
                    nsamp_dedisp = nsamp - maxDT
            Only returned if retDMT is True.
        sigmi: float
            The SNR of the most prominent feature in the DMT.
            Only returned if retDMT is False.
        """

        if I.dtype.itemsize < 4:
            I = I.astype(np.uint32)

        # Concatenate some zero padding arrays (but there may be zero such
        # arrays)
        concat_tuple = [I]
        if padding:
            concat_tuple.append(np.zeros((self.nchan, self.maxDT), dtype=I.dtype))
        if frontpadding:
            # We need to add maxDT time samples to the front of I when
            # frontpadding is desired, due to edge-related effects
            concat_tuple.insert(0, np.zeros((self.nchan, self.maxDT), dtype=I.dtype))

        I = np.concatenate(concat_tuple, axis=1)

        if self.A is None or self.A.shape[1] != I.shape[1] or self.A.dtype != I.dtype or True:
            self.prep(I.shape[1], dtype=I.dtype)

        t1 = time()
        self.fdmt_initialize(I)

        t2 = time()
        for i in range(1, int(np.log2(self.nchan)) + 1):
            src, dest = (self.A, self.B) if (i % 2 == 1) else (self.B, self.A)
            self.fdmt_iteration(src, dest, i)

        if verbose:
            t3 = time()
            print("Initializing time:  %.2f s" % (t2 - t1))
            print("Iterating time:  %.2f s" % (t3 - t2))
            print("Total time: %.2f s" % (t3 - t1))

        DMT = dest[:self.maxDT]

        if retDMT:
            # We need to cut off the first maxDT samples either way
            # because now frontpadding works by inserting maxDT samples' worth
            # of zeros at the front of I
            return DMT[:, self.maxDT:]
        noiseRMS = np.array([DMT[i, i:].std() for i in range(self.maxDT)])
        noiseMean = np.array([DMT[i, i:].mean() for i in range(self.maxDT)])
        sigmi = (DMT.T - noiseMean) / noiseRMS
        if verbose:
            print("Maximum sigma value: %.3f" % sigmi.max())
        return sigmi.max()


    def fdmt_initialize(self, I):
        self.A[self.Q[0], :] = I
        chDTs = self.subDT(self.fs)
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
        # Copy Q, src, and dest to shared memory.
        # Since we only need Q[i] and Q[i - 1], we only
        # need to copy those to shared memory.
        shm_Qi = SharedMemory(create=True, size=self.Q[i].nbytes)
        shm_Qim1 = SharedMemory(create=True, size=self.Q[i-1].nbytes)
        shm_src = SharedMemory(create=True, size=self.src.nbytes)
        shm_dest = SharedMemory(create=True, size=self.dest.nbytes)

        sharr_Qi = np.ndarray(self.Q.shape, buffer=shm_Qi.buf, dtype=self.Q[i].dtype)
        sharr_Qim1 = np.ndarray(self.Q.shape, buffer=shm_Qim1.buf, dtype=self.Q[i-1].dtype)
        sharr_src = np.ndarray(self.src.shape, buffer=shm_src.buf, dtype=src.dtype)
        sharr_dest = np.ndarray(self.dest.shape, buffer=shm_dest.buf, dtype=dest.dtype)

        sharr_Qi[:] = self.Q[i][:]
        sharr_Qim1[:] = self.Q[i-1][:]
        sharr_src[:] = src[:]
        sharr_dest[:] = dest[:]

        with Pool() as pool:
            pool.starmap(
                self._fdmt_iteration, 
                self._fdmt_iteration_get_iterator(
                    shm_src.name, shm_dest.name, shm_Qi.name, shm_Qim1.name, src.shape, self.Q[i].shape,
                    src.dtype, i, self.dF
                )
            )

        # Copy data from shared arrays back into main arrays and clean up.
        # Since we only actually write to dest, we only need to copy that array.
        dest[:] = sharr_dest[:]

        shm_Qi.close()
        shm_Qim1.close()
        shm_src.close()
        shm_dest.close()

        shm_Qi.unlink()
        shm_Qim1.unlink()
        shm_src.unlink()
        shm_dest.unlink()


    def _fdmt_iteration(self,
                        shm_src_name, shm_dest_name, shm_Qi_name, shm_Qim1_name,
                        srcdest_shape, Qi_shape,
                        dtype,
                        i, i_F, T, dF, f0, f1, f2, cor):
        shm_Qi = SharedMemory(name=shm_Qi_name, create=False)
        shm_Qim1 = SharedMemory(name=shm_Qim1_name, create=False)
        shm_src = SharedMemory(name=shm_src_name, create=False)
        shm_dest = SharedMemory(name=shm_dest_name, create=False)

        Qi = np.ndarray(Qi_shape, dtype=dtype,buffer=shm_Qi.buf)
        Qim1 = np.ndarray(Qi_shape, dtype=dtype,buffer=shm_Qim1.buf)
        src = np.ndarray(srcdest_shape, dtype=dtype,buffer=shm_src.buf)
        dest = np.ndarray(srcdest_shape, dtype=dtype,buffer=shm_dest.buf)
        
        C = (f1**-2 - f0**-2) / (f2**-2 - f0**-2)
        C01 = ((f1 - cor) ** -2 - f0**-2) / (f2**-2 - f0**-2)
        C12 = ((f1 + cor) ** -2 - f0**-2) / (f2**-2 - f0**-2)
        for i_dT in range(self.subDT(f0, dF)):
            dT_mid01 = round(i_dT * C01)
            dT_mid12 = round(i_dT * C12)
            dT_rest = i_dT - dT_mid12
            dest[Qi[i_F] + i_dT, :] = src[Qim1[2 * i_F] + dT_mid01, :]
            dest[Qi[i_F] + i_dT, dT_mid12:] += src[
                Qim1[2 * i_F + 1] + dT_rest, : T - dT_mid12
            ]

        shm_Qi.close()
        shm_Qim1.close()
        shm_src.close()
        shm_dest.close()


    def _fdmt_iteration_get_iterator(self,
            shm_src_name, shm_dest_name, shm_Qi_name, shm_Qim1_name, 
            srcdest_shape, Qi_shape, dtype, i, dF):
        '''Returns the iterator that supplies the arguments to be passed by starmap
        to _fdmt_iterator
        
        Parameters
        ----------
        shm_src_name: string
            The name of the SharedMemory instance holding the shared src array. 
        shm_dest_name: string
            The name of the SharedMemory instance holding the shared dest array. 
        shm_Qi_name: string
            The name of the SharedMemory instance holding the shared array representing Q[i]. 
        shm_Qim1_name: string
            The name of the SharedMemory instance holding the shared array representarray representing Q[i - 1]. 
        srcdest_shape: tuple of ints
            The shape of src and dest.
        Qi_shape: tuple of ints
            The shape of Q[i] and Q[i - 1].
        dtype: dtype
            The dtype of src and dest.
        '''

        T = srcdest_shape[1]
        dF = self.df * 2**i
        f_starts = self.fs[:: 2**i]
        f_ends = f_starts + dF
        f_mids = self.fs[2 ** (i - 1) :: 2**i]
        for i_F in range(self.nchan // 2**i):
            f0 = f_starts[i_F]
            f1 = f_mids[i_F]
            f2 = f_ends[i_F]
            # Using cor = df seems to give the best behaviour at high DMs, judging from
            # presto output. Nevertheless it may be worth adjusting this option
            # for a specific use case
            cor = self.df if i > 1 else 0

            yield (self, shm_src_name, shm_dest_name, shm_Qi_name, shm_Qim1_name,
                   srcdest_shape, Qi_shape, dtype, i, i_F, T, dF, f0, f1, f2, cor)



    def reset_ABQ(self):
        self.A = None
        self.B = None
        self.Q = []

    def recursive_fdmt(self, I, depth=0, curMax=0):
        """Performs FDMT, downsamples and repeats recursively, returning max sigma
        I should have shape (nchan, nsamp) where nsamp is the number of time samples"""
        curMax = max(curMax, fdmt(I))
        if depth <= 0:
            return curMax
        else:
            I2 = (
                I[:, ::2] + I[:, 1::2]
                if (I.shape[1] % 2 == 0)
                else I[:, :-1:2] + I[:, 1::2]
            )
            return self.recursive_fdmt(I2, depth - 1, curMax)
