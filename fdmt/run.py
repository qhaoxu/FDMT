'''Runs FDMT on filterbank data, yielding output in the format of the user's
choice.'''
from .cpu_fdmt import FDMT
import numpy as np
from astropy.coordinates import Angle


def write_inffile(basename, dm, dmprec=2, **infdict):
    '''Writes a presto .inf file with prefix basename.'''
    inf_template = (
""" Data file name without suffix          =  {}
 Telescope used                         =  {}
 Instrument used                        =  {}
 Object being observed                  =  {}
 J2000 Right Ascension (hh:mm:ss.ssss)  =  {}
 J2000 Declination     (dd:mm:ss.ssss)  =  {}
 Data observed by                       =  UNKNOWN
 Epoch of observation (MJD)             =  {:5.15f}
 Barycentered?           (1=yes, 0=no)  =  {}
 Number of bins in the time series      =  {}
 Width of each time series bin (sec)    =  {:.15g}
 Any breaks in the data? (1 yes, 0 no)  =  0
 Orbit removed?          (1=yes, 0=no)  =  {}
 Dispersion measure (cm-3 pc)           =  {:f}
 Central freq of low channel (Mhz)      =  {:f}
 Total bandwidth (Mhz)                  =  {:f}
 Number of channels                     =  {}
 Channel bandwidth (Mhz)                =  {}
 Data analyzed by                       =  {}
 Any additional notes:
     fdmt
""")

    filename = f"{basename}_DM{dm:.{dmprec}f}.inf"
    with open(filename, 'w') as ff:
        ff.write(inf_template.format(
            filename[:-4],  # data filename without suffix
            infdict.get('telescope', 'Other'),
            infdict.get('instrument', 'Other'),
            infdict.get('source_name', 'Object'),
            infdict.get('src_raj', Angle('0h')).to_string(sep=':'),
            infdict.get('src_dej', Angle('0d')).to_string(sep=':'),
            infdict.get('tstart', 0.0),
            1 if infdict.get('barycentric', False) else 0,
            1,  # num bins in time series
            infdict['nsamples'] * infdict['tsamp'],  # width of tseries bin
            1 if infdict.get('orbit_removed', False) else 0,
            dm,
            infdict['fch1'],
            infdict['foff'] * infdict['nchans'],   # total bandwidth
            infdict['nchans'],
            infdict['foff'],
            'user'
        ))



def write_dat_inf_file(basename, data, dm, dmprec=2, **infdict):
    '''Writes a presto .dat and a .inf file with prefix basename.'''
    # .dat file
    datname = f"{basename}_DM{dm:.{dmprec}f}.dat"
    data.tofile(datname)
    
    # .inf file
    write_inffile(basename, dm, dmprec, **infdict)


if __name__ == '__main__':
    import argparse
    
    try:
        import blimpy
    except ImportError as e:
        raise ImportError('Using FDMT from command line requires blimpy') \
                from e

    parser = argparse.ArgumentParser(
        description='Dedisperse filterbank files from command line.',
    )

    parser.add_argument(
        'infile',
        type=str,
        help='The filterbank file to be read in.'
    )

    parser.add_argument(
        'outfile',
        type=str,
        help='The file to be read written out,'
        ' with shape (maxDT, nsamples_fdmt). Its filetype is inferred from'
        ' the file extension, except presto-style output has no'
        ' extension.'
    )

    parser.add_argument(
        '--band-edges', '-e',
        type=float, 
        nargs=2,
        default=None,
        help='The lower and upper edges of the band, in MHz. If not given,'
        ' will attempt to read from the filterbank file.'
    )
    
    parser.add_argument(
        '--maxdt', '-d',
        type=int,
        default=2048,
        help='The number of time samples corresponding to the maximum delay'
        ' between the top and bottom of the band; this defines the maximum'
        ' DM of the search'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Whether to print info to stdout.'
    )

    parser.add_argument(
        '--padding', '-p',
        action='store_true',
        help='Whether to return additional, incompletely integrated time'
        ' samples at the end of the dedispersed time series.'
    )

    parser.add_argument(
        '--front-padding', '-f',
        action='store_true',
        help='Whether to return additional, incompletely integrated time'
        ' samples at the beginning of the dedispersed time series.'
    )
    
    parser.add_argument(
        '--dm-precision', '-n',
        type=int,
        default=2,
        help='When using presto-style dedispersion output, the precision'
        ' of the DM quoted in the filenames of the .dat and .inf'
        ' filenames. Unused when the output is in another format.'
    )

    args = parser.parse_args()

    fil = blimpy.Waterfall(args.infile)
    data = np.squeeze(fil.data.astype(np.float32))

    nchans = fil.header.get('nchans', data.shape[1])

    if args.band_edges is None:
        fch1 = fil.header['fch1']
        foff = fil.header['foff']
        tsamp = fil.header['tsamp']

        if foff < 0:
            # this is the middle of the highest frequency band
            fmax = fch1 - 0.5 * foff   # fmax is larger than fch1
            fmin = fmax + nchans * foff   # fmin is smaller than fmax
            data = data[:, ::-1]
        else:
            fmin = fch1 - 0.5 * foff   # fmin is smaller than fch1
            fmax = fmin + nchans * foff   # fmax is greater than fmin
    else:
        fmin, fmax = args.band_edges
        if fmin > fmax:
            fmin, fmax = fmax, fmin
            data = data[:, ::-1]

        print('Initializing dedispersion between',
              fmin, 'and', fmax, 'MHz')

    fdmt = FDMT(fmin=fmin, fmax=fmax,
                nchan=fil.header.get('nchans', data.shape[1]),
                maxDT=args.maxdt
               )

    out = fdmt.fdmt(data.T, retDMT=True, verbose=args.verbose,
                    padding=args.padding, frontpadding=args.front_padding)

    outfilename = args.outfile.rsplit('.', maxsplit=1)
    if len(outfilename) == 1:
        # Save .dat and .inf files like presto
        DM_CONSTANT = 1.0 / 2.41e-4    # Need to know DM for filenames
        dms = ((np.arange(0, fdmt.maxDT) /
            DM_CONSTANT / (1/fdmt.fmin**2 - 1/fdmt.fmax**2)) * tsamp)
        for i in range(args.maxdt):
            dm = dms[i]
            data = out[i]
            basename = outfilename[0]
            write_dat_inf_file(basename, data, dm, dmprec=args.dm_precision, **fil.header)

    elif outfilename[-1] == 'npy':
        np.save(args.outfile, out)
    elif outfilename[-1] == 'npz':
        np.savez(args.outfile, out)
    elif outfilename[-1] == 'dat':
        np.tofile(args.outfile, out)
    else:
        import sys
        print('Warning: did not recognize output file type; saving as npz',
              out=sys.err)
        np.savez(args.outfile + '.npz', out)

