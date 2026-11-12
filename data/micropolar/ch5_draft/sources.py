"""
sources.py - where every item of the pack reads its runs. One entry per data set; an item script names the entry it
uses, so replacing older runs by the runs on thesis-ch5-rc2 is one line here.

status: "final" = runs on thesis-ch5-rc2 (Stage 6); "provisional" = older runs, named in "note".
"""

# concentric cylinders: analysis file of micropolar_stage4.py analyse / micropolar_stage6.py concentric
CONCENTRIC_STAGE4 = {
    "analysis": "data/micropolar/stage4/couette_stage4.json", "status": "provisional", "tag": "thesis-ch5-rc1",
    "note": "Stage 4 runs on thesis-ch5-rc1 (thesis configuration, density diffusion towards the walls)"}
CONCENTRIC_STAGE6 = {
    "analysis": "data/micropolar/stage6/concentric.json", "status": "final", "tag": "thesis-ch5-rc2",
    "note": "Stage 6 runs on thesis-ch5-rc2 (numerics.delta_fluid_only: true)"}
CONCENTRIC = CONCENTRIC_STAGE4

# eccentric cylinders: benchmark file of micropolar_stage5.py benchmark / micropolar_stage6.py eccentric
ECCENTRIC_STAGE5 = {
    "analysis": "data/micropolar/stage5/benchmark.json", "status": "provisional", "tag": "thesis-ch5-rc1 + Stage 5",
    "note": "Stage 4 and 5 runs without the density diffusion term (numerics.use_density_diffusion: false)"}
ECCENTRIC_STAGE6 = {
    "analysis": "data/micropolar/stage6/eccentric.json", "status": "final", "tag": "thesis-ch5-rc2",
    "note": "Stage 6 runs on thesis-ch5-rc2 (numerics.delta_fluid_only: true)"}
ECCENTRIC = ECCENTRIC_STAGE5
