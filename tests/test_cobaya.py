import numpy as np
import pytest

pytest.importorskip("camb")
cobaya = pytest.importorskip("cobaya")

from cobaya.model import get_model

PARAMS = {"H0": 67.36, "ombh2": 0.02237, "omch2": 0.12, "logA": 3.044, "ns": 0.9649,
          "tau": 0.0544, "mnu": 0.06, "nnu": 3.044}


def _model(theory, extra=None):
    info = {
        "params": dict(PARAMS),
        "likelihood": {"cls": {"external": lambda _self=None: 0.0,
                               "requires": {"Cl": {"tt": 100, "te": 100, "ee": 100, "pp": 100}}}},
        "theory": {theory: extra or {}},
    }
    if theory == "fastcmb.cobaya.FastCAMB":
        info["params"]["sigma8"] = None
    else:
        # cobaya's camb takes As, not logA.
        info["params"]["As"] = 1e-10 * np.exp(info["params"].pop("logA"))
    return get_model(info)


@pytest.mark.parametrize("ell_factor", [False, True])
@pytest.mark.parametrize("units", ["FIRASmuK2", "1", "muK2"])
def test_cls_match_cobaya_camb(ell_factor, units):

    fast = _model("fastcmb.cobaya.FastCAMB", {"lens_potential_accuracy": 1})
    slow = _model("camb", {"extra_args": {"lens_potential_accuracy": 1, "num_massive_neutrinos": 1,
                                          "halofit_version": "mead2020"}})

    fast.logposterior({})
    slow.logposterior({})

    a = fast.provider.get_Cl(ell_factor=ell_factor, units=units)
    b = slow.provider.get_Cl(ell_factor=ell_factor, units=units)

    # The conventions -- units, ell factors -- would be off by orders of
    # magnitude. What is left is the two CAMB set-ups' accuracy: 1e-4 in
    # TT, 0.5% in EE at the reionization trough, TE where it crosses zero.
    for name in ("tt", "te", "ee", "pp"):
        x, y = a[name][2:101], b[name][2:101]
        assert x == pytest.approx(y, rel=1e-2, abs=1e-2 * np.abs(y).max())
        assert np.median(np.abs(x / y - 1)) < 2e-3


def test_sigma8_is_derived():

    m = _model("fastcmb.cobaya.FastCAMB")
    point = m.logposterior({})
    assert 0.7 < point.derived[0] < 0.9
