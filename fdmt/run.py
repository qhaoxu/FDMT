if __name__ == '__main__':
    import argparse
    from .cpu_fdmt import FDMT
    import numpy as np
    
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
        help='The file to be read written out, in  npz format,'
        ' with shape (maxDT, nsamples_fdmt)'
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

    args = parser.parse_args()

    fil = blimpy.Waterfall(args.infile)
    data = np.squeeze(fil.data)

    nchans = fil.header.get('nchans', data.shape[1])

    if args.band_edges is None:
        fch1 = fil.header['fch1']
        foff = fil.header['foff']

        if foff < 0:
            # this is the middle of the highest frequency band
            fmax = fch1 - 0.5 * foff   # fmax is larger than fch1
            fmin = fmax + nchans * foff   # fmin is smaller than fmax
            data = data[:, ::-1]
        else:
            fmin = fch1 - 0.5 * foff   # fmin is smaller than fch1
            fmax = fmin + nchans * foff   # fmax is greater than fmin
    else:
        fmin, fmax = band_edges
        if fmin > fmax:
            fmin, fmax = fmax, fmin
            data = data[:, ::-1]

    if args.verbose:
        print('Initializing dedispersion between',
              fmin, 'and', fmax, 'MHz')

    fdmt = FDMT(fmin=fmin, fmax=fmax,
                nchan=fil.header.get('nchans', data.shape[1]),
                maxDT=args.maxdt
               )

    out = fdmt.fdmt(data.T, retDMT=True, verbose=args.verbose,
                    padding=args.padding, frontpadding=args.front_padding)

    outfiletype = args.outfile.rsplit('.', maxsplit=1)
    if outfiletype = 'npy':
        np.save(args.outfile, out)
    elif outfiletype = 'npz':
        np.savez(args.outfile, out)
    elif outfiletype = 'dat':
        np.tofile(args.outfile, out)
    else:
        import sys
        print('Warning: did not recognize output file type; saving as npz',
              out=sys.err)
        np.savez(args.outfile + '.npz', out)

