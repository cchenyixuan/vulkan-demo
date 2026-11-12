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
    "analysis": "data/micropolar/stage6/concentric.json", "tag": "thesis-ch5-rc2",
    "status": "final (rc2 runs); incomplete: rows at g = 200 dx still running, rows at g = 400 dx not reached",
    "note": "Stage 6 runs on thesis-ch5-rc2 (numerics.delta_fluid_only: true)"}
CONCENTRIC = CONCENTRIC_STAGE6

# eccentric cylinders: analysis file of micropolar_stage6.py eccentric (--set stage5 for the older runs)
ECCENTRIC_STAGE5 = {
    "analysis": "data/micropolar/stage6/eccentric_stage5_runs.json", "status": "provisional",
    "tag": "thesis-ch5-rc1 and the commits of Stage 5",
    "note": "Stage 4 and 5 runs without the density diffusion term (numerics.use_density_diffusion: false)"}
ECCENTRIC_STAGE6 = {
    "analysis": "data/micropolar/stage6/eccentric.json", "tag": "thesis-ch5-rc2",
    "status": "final (rc2 runs); incomplete: rows at g = 200 dx and the noise floors still running",
    "note": "Stage 6 runs on thesis-ch5-rc2 (numerics.delta_fluid_only: true)"}
ECCENTRIC = ECCENTRIC_STAGE6
# the Newtonian benchmark rows at Reynolds number 1 exist only in the setting of Stage 5 (they were not asked again)
ECCENTRIC_BENCHMARK_STAGE5 = ECCENTRIC_STAGE5

# fields of one eccentric run (micropolar_stage6.py fields): eps = 0.6, K = 3, l = 10 dx, g = 100 dx
ECCENTRIC_FIELD_STAGE5 = {
    "fields": "data/micropolar/stage6/fields/nd_e60_k3_m5", "status": "provisional",
    "tag": "thesis-ch5-rc1 and the commits of Stage 4",
    "note": "Stage 4 run without the density diffusion term"}
ECCENTRIC_FIELD_STAGE6 = {
    "fields": "data/micropolar/stage6/fields/e60_k3_m5", "status": "final", "tag": "thesis-ch5-rc2",
    "note": "Stage 6 run on thesis-ch5-rc2 (numerics.delta_fluid_only: true)"}
ECCENTRIC_FIELD = ECCENTRIC_FIELD_STAGE6
