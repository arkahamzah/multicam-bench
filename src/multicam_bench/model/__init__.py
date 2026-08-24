"""Capacity fit and calculator (PROJECT-CHARTER-v2.md §3 Layer 3 "TAHU BATASNYA").

`fit.py` fits the two-term cost model per subsystem; `machine_profile.py` stores
fitted coefficients and answers "does this fit, what's the limiting subsystem, how
much headroom" for a given (cameras, resolution, fps) request.
"""
