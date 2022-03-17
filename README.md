This is a Python implementation of FDMT, an algorithm to dedisperse pulsar and FRB signals
described by [Zackay \& Ofek (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...835...11Z/abstract).

# Installing
To install the core package, use `pip install git+ssh://git@github.com/chime-sps/FDMT.git`.

If you plan on calling FDMT directly from the command line on filterbank-format .fil files, 
use `pip install git+ssh://git@github.com/chime-sps/FDMT.git[cmdline]`.

# Using the code
The core package is `import`ed like any other Python package.

To run on filterbank files from command line, use `python -m fdmt.run`:
```
$ python -m fdmt.run infile.fil outfile.npy
$ python -m fdmt.run --help
```
The code can output raw binary `.dat` files, numpy `.npy` and `.npz` files,
and [presto](https://github.com/scottransom/presto/)-style time series defined by 
`.dat` and `.inf` file pairs. The output is recognized by the file extension, for instance:
`python -m fdmt.run infile.fil outfile.npy` writes a numpy array of dedispersed time
series to `outfile.npy`. When there is no file extension, then the output is
(intended to be) in the same format as the output of presto's `prepsubband` output:
`python -m fdmt.run infile.fil outfile` writes numerous files called `outfile_DMXX.XX.dat`
and `outfile_DMXX.XX.inf`, one pair of `.dat` and `.inf` files for each time series.

Happy dedispersing.

(more detailed documentation to come)
