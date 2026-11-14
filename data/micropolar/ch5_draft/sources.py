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
    "status": "final (rc2 runs) for g <= 200 dx; the rows at g = 400 dx are not reached",
    "note": "Stage 6 runs on thesis-ch5-rc2 (numerics.delta_fluid_only: true)"}
CONCENTRIC = CONCENTRIC_STAGE6

# eccentric cylinders: analysis file of micropolar_stage6.py eccentric (--set stage5 for the older runs)
ECCENTRIC_STAGE5 = {
    "analysis": "data/micropolar/stage6/eccentric_stage5_runs.json", "status": "provisional",
    "tag": "thesis-ch5-rc1 and the commits of Stage 5",
    "note": "Stage 4 and 5 runs without the density diffusion term (numerics.use_density_diffusion: false)"}
ECCENTRIC_STAGE6 = {
    "analysis": "data/micropolar/stage6/eccentric.json", "tag": "thesis-ch5-rc2",
    "status": "final",
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

# the dipole-wall collision of Stage 8 (micropolar_dipole_analysis.py analyse / fields)
DIPOLE = {
    "analysis": "data/micropolar/stage8/dipole.json", "fields": "data/micropolar/stage8/fields",
    "status": "final", "tag": "thesis-ch5-rc2 solver trees (runs on branch v10-showcase)",
    "note": "Stage 8 runs: the solver directories identical to thesis-ch5-rc2"}

# Stage 8b: periodic boundaries (micropolar_periodic.py / micropolar_turbulence.py analyse)
PERIODIC = {
    "analysis": "data/micropolar/stage8b/taylor_green.json", "turbulence": "data/micropolar/stage8b/turbulence.json",
    "fields": "data/micropolar/stage8b/fields", "status": "final",
    "tag": "v10-periodic (thesis-ch5-rc2 + the periodic domain of Stage 8b)",
    "note": "Stage 8b runs on branch v10-periodic"}
