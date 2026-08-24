"""The production pipeline stages (PROJECT-CHARTER-v2.md §3 Layer 1: ingest ->
[gate] -> detect -> track -> analytics -> sinks). Only `detect` (v0.5) exists so
far — track/analytics/sinks are later milestones. `bench/` drives these stages
during measurement; it does not duplicate their logic.
"""
