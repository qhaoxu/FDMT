# Until PEP-517 can handle editable installs
# See: https://setuptools.readthedocs.io/en/latest/userguide/quickstart.html#development-mode

import setuptools

setuptools.setup(
    packages=setuptools.find_packages(),
)
