"""Where the debug cossim scripts find the PyTorch ACE-Step reference.

ACESTEP_REF, made absolute and moved to the front of sys.path, when it is
set; otherwise whatever PYTHONPATH gives. No fixed path: the old
../../ACE-Step-1.5 link went first on sys.path and won over both
(erichchampion/cadenza-audio#1146). Python puts a script's own directory on
sys.path, so debug-*-cossim.py import this from anywhere.
"""
import importlib.util
import os
import sys


HINT = ("set ACESTEP_REF to a clone of github.com/erichchampion/ace-step-1.5, the reference "
        "the goldens are anchored against (or put it on PYTHONPATH)")


def use_reference():
    ref = os.environ.get("ACESTEP_REF")
    if ref:
        ref = os.path.abspath(ref)
        # Named, so nothing else will do: not a copy found through PYTHONPATH.
        if not os.path.isfile(os.path.join(ref, "acestep", "handler.py")):
            sys.exit("ACESTEP_REF=%s holds no acestep/handler.py: %s" % (ref, HINT))
        # First, whatever PYTHONPATH also holds: ACESTEP_REF wins.
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != ref]
        sys.path.insert(0, ref)
    try:
        found = importlib.util.find_spec("acestep.handler") is not None
    except ImportError:
        found = False
    if not found:
        sys.exit("the ACE-Step reference (acestep.handler) isn't importable: " + HINT)
