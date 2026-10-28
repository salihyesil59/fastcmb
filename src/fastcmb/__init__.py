"""
fastcmb: an online emulator of Boltzmann codes for CMB analyses.

:class:`CMBEmulator` wraps a Boltzmann code -- an *oracle*, such as
:class:`CAMBOracle` -- and answers with a Gaussian-process prediction
wherever its error, in units of the log-likelihood, is below a
tolerance; elsewhere the code runs, and the point is learned. It needs
only numpy and scipy, and fits in any analysis code:

* CosmoFit: ``theory: {background: ..., fastcmb: ...}`` in place of
  ``camb`` (built into CosmoFit, with ``fastcmb`` installed);
* cobaya: the theory ``fastcmb.cobaya.FastCAMB`` in place of ``camb``;
* anything else: call a :class:`CMBEmulator`.
"""

from .camb_oracle import CAMBOracle
from .emulator import CMBEmulator, Result
from .gp import GP
from .online import OnlineEmulator

__all__ = ["CAMBOracle", "CMBEmulator", "GP", "OnlineEmulator", "Result", "__version__"]

__version__ = "0.2.0"
