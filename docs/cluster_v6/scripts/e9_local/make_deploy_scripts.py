"""E9 deploy scripts = the E8 ones with the E9 names (deployment ~/run/vulkan-demo-e9 on tag v7-rc4, the v7-single-rc4
deployment ~/run/vulkan-demo-v7single-rc4) and two extra checks (the defrag order default, the defrag_prefix module).
Exact-string replacements, each asserted."""
import pathlib

SCRIPTS = pathlib.Path("C:/Users/cchen/PycharmProjects/vulkan-demo-e8/docs/cluster_v6/scripts")


def derive(source, target, replacements):
    text = (SCRIPTS / source).read_text(encoding="utf-8")
    for old, new in replacements:
        count = text.count(old)
        assert count >= 1, f"{source}: missing {old[:80]!r}"
        text = text.replace(old, new)
    (SCRIPTS / target).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {target}")


derive("e8_deploy_remote.sh", "e9_deploy_remote.sh", [
    ("# e8_deploy_remote.sh — deploy the E8 tag into ~/run/vulkan-demo-e8 on the N56 login node (never on a compute node).",
     "# e9_deploy_remote.sh — deploy the E9 tag (v7-rc4: the prefix defrag order as the production path, kernels unchanged)\n"
     "# into ~/run/vulkan-demo-e9 on the N56 login node (never on a compute node). Derived from e8_deploy_remote.sh (E8)."),
    ("#   vd_e8_<SHA>.tar.gz   git archive --prefix=vulkan-demo-e8/ <tag>", "#   vd_e9_<SHA>.tar.gz   git archive --prefix=vulkan-demo-e9/ <tag>"),
    ("#   e8_sha.txt           the 40-hex SHA of the tag's commit", "#   e9_sha.txt           the 40-hex SHA of the tag's commit"),
    ("docs/cluster_v6/scripts docs/cluster_v6/e8_jobs", "docs/cluster_v6/scripts docs/cluster_v6/e8_jobs docs/cluster_v6/e9_jobs"),
    ("SHA=${SHA:-$(cat \"$HOME/run/tools/e8_sha.txt\" 2>/dev/null | tr -d '[:space:]')}",
     "SHA=${SHA:-$(cat \"$HOME/run/tools/e9_sha.txt\" 2>/dev/null | tr -d '[:space:]')}"),
    ("DEST=${DEST:-$HOME/run/vulkan-demo-e8}", "DEST=${DEST:-$HOME/run/vulkan-demo-e9}"),
    ("DEPLOY_TAG=${DEPLOY_TAG:-e8-3d}", "DEPLOY_TAG=${DEPLOY_TAG:-v7-rc4}"),
    ("TARBALL=$TOOLS/vd_e8_$SHA.tar.gz", "TARBALL=$TOOLS/vd_e9_$SHA.tar.gz"),
    ("STAGE=/dev/shm/scxm138/e8_stage_$SHA", "STAGE=/dev/shm/scxm138/e9_stage_$SHA"),
    ("echo \"=== e8 deploy $(date)", "echo \"=== e9 deploy $(date)"),
    ("solver_tag=$DEPLOY_TAG base=v7-rc3", "solver_tag=$DEPLOY_TAG base=v7-rc4"),
    ("python docs/cluster_v6/scripts/tree_hash.py experiment/v7 --expect \"$EXPECT_TREE\" && echo \"experiment/v7 tree OK ($EXPECT_TREE)\" || fail \"experiment/v7 tree\"\n",
     "python docs/cluster_v6/scripts/tree_hash.py experiment/v7 --expect \"$EXPECT_TREE\" && echo \"experiment/v7 tree OK ($EXPECT_TREE)\" || fail \"experiment/v7 tree\"\n"
     "# E9: the prefix defrag order is the code default and its host-scan module imports\n"
     "python -c \"import dataclasses; from experiment.v7.utils.case_v7 import NumericsConstants as N; from experiment.v7.utils import defrag_prefix; \\\n"
     "    f = {x.name: x.default for x in dataclasses.fields(N)}; assert f['use_prefix_sum_defrag'] is True, f; print('defrag order default: prefix (defrag_prefix imported)')\" \\\n"
     "    || fail \"defrag order default\"\n"
     "cmp -s experiment/v7/shaders/spv/defrag.comp.spv \"$HOME/run/vulkan-demo-e8/experiment/v7/shaders/spv/defrag.comp.spv\" && echo \"SPIR-V defrag.comp.spv = E8's\" || fail \"defrag.comp.spv differs from E8's\"\n"
     "for spv in correction_density.comp.spv force.comp.spv predict.comp.spv update_voxel.comp.spv; do\n"
     "    cmp -s \"experiment/v7/shaders/spv/$spv\" \"$HOME/run/vulkan-demo-e8/experiment/v7/shaders/spv/$spv\" && echo \"SPIR-V $spv = E8's\" || fail \"SPIR-V $spv differs from E8's\"\n"
     "done\n"),
    ("docs/cluster_v6/scripts/e8_field_compare.py docs/cluster_v6/scripts/e8_vram.py docs/cluster_v6/scripts/e8_plan.py \\\n",
     "docs/cluster_v6/scripts/e8_field_compare.py docs/cluster_v6/scripts/e8_vram.py docs/cluster_v6/scripts/e8_plan.py experiment/v7/utils/defrag_prefix.py \\\n"),
    ("for script in ${E8_DRY_JOBS:-docs/cluster_v6/e8_jobs/*.sbatch}; do", "for script in ${E8_DRY_JOBS:-docs/cluster_v6/e9_jobs/*.sbatch}; do"),
    ("DRY_LOG=$HOME/run/logs/e8_deploy_dry_", "DRY_LOG=$HOME/run/logs/e9_deploy_dry_"),
    ("E7_STATE=/dev/shm/scxm138/e8_state_dry E7_OBJ_CACHE=$HOME/run/e7_objcache E30_SHM_ROOT=/dev/shm/scxm138/e8_dry",
     "E7_STATE=/dev/shm/scxm138/e9_state_dry E7_OBJ_CACHE=$HOME/run/e7_objcache E30_SHM_ROOT=/dev/shm/scxm138/e9_dry"),
    ("E8_SINGLE_REPO=$HOME/run/vulkan-demo-v7single timeout 900", "E8_SINGLE_REPO=$HOME/run/vulkan-demo-v7single-rc4 timeout 900"),
    ("rm -rf /dev/shm/scxm138/e8_dry /dev/shm/scxm138/e8_state_dry", "rm -rf /dev/shm/scxm138/e9_dry /dev/shm/scxm138/e9_state_dry"),
    ("for other in \"$HOME/run/vulkan-demo-v7rc2\" \"$HOME/run/vulkan-demo-v7rc3\"; do",
     "for other in \"$HOME/run/vulkan-demo-v7rc2\" \"$HOME/run/vulkan-demo-v7rc3\" \"$HOME/run/vulkan-demo-e8\"; do"),
    ("docs/cluster_v6/e8_jobs/<job>.sbatch\"", "docs/cluster_v6/e9_jobs/<job>.sbatch\""),
])

derive("e8_deploy_single_remote.sh", "e9_deploy_single_remote.sh", [
    ("# e8_deploy_single_remote.sh — deploy the v7_single solver (branch v7-single, experiment/v7_single on v7-rc1's\n"
     "# experiment/v7 and SPIR-V) into ~/run/vulkan-demo-v7single on the N56 login node, for the E8 single-GPU-solver\n",
     "# e9_deploy_single_remote.sh — deploy the v7_single solver of tag v7-single-rc4 (branch v7-single: experiment/v7_single on\n"
     "# v7-rc1's experiment/v7 + the E9 prefix defrag host scan; SPIR-V unchanged) into ~/run/vulkan-demo-v7single-rc4 on the\n"
     "# N56 login node, for the E9 single-GPU-solver\n"),
    ("#   v7single_sha.txt           the 40-hex SHA of that commit", "#   v7single_rc4_sha.txt       the 40-hex SHA of that commit (tag v7-single-rc4)"),
    ("# Usage: [SHA=<40 hex>] [UPDATE=1] [DEST=<dir>] bash e8_deploy_single_remote.sh",
     "# Usage: EXPECT_TREE=<sha> [SHA=<40 hex>] [UPDATE=1] [DEST=<dir>] bash e9_deploy_single_remote.sh"),
    ("SHA=${SHA:-$(cat \"$HOME/run/tools/v7single_sha.txt\" 2>/dev/null | tr -d '[:space:]')}",
     "SHA=${SHA:-$(cat \"$HOME/run/tools/v7single_rc4_sha.txt\" 2>/dev/null | tr -d '[:space:]')}"),
    ("EXPECT_TREE=82dd6a740fa2a97a44e6a31d75392f75e486163b       # v7-rc1's experiment/v7 (the branch v7-single never edits it)",
     "EXPECT_TREE=${EXPECT_TREE:?\"the experiment/v7 tree hash of tag v7-single-rc4 (git rev-parse v7-single-rc4:experiment/v7)\"}"),
    ("DEST=${DEST:-$HOME/run/vulkan-demo-v7single}", "DEST=${DEST:-$HOME/run/vulkan-demo-v7single-rc4}"),
    ("STAGE=/dev/shm/scxm138/v7single_stage_$SHA", "STAGE=/dev/shm/scxm138/v7single_rc4_stage_$SHA"),
    ("echo \"=== v7single deploy $(date)", "echo \"=== v7single-rc4 deploy $(date)"),
    ("if [ -e \"$DEST\" ] && echo \"$QUEUE\" | grep -F \"vulkan-demo-e8\" | grep -vF \"vulkan-demo-e8d\"; then echo \"ABORT: an E8 job that may read $DEST is queued or running\"; exit 1; fi",
     "if [ -e \"$DEST\" ] && echo \"$QUEUE\" | grep -F \"vulkan-demo-e9\"; then echo \"ABORT: an E9 job that may read $DEST is queued or running\"; exit 1; fi"),
    ("solver_tag=v7-single base=v7-rc1", "solver_tag=v7-single-rc4 base=v7-rc1+E9-defrag-prefix"),
    ("python \"$HOME/run/vulkan-demo-e8/docs/cluster_v6/scripts/tree_hash.py\" experiment/v7 --expect \"$EXPECT_TREE\" && echo \"experiment/v7 tree OK (v7-rc1)\" || fail \"experiment/v7 tree\"",
     "python \"$HOME/run/vulkan-demo-e9/docs/cluster_v6/scripts/tree_hash.py\" experiment/v7 --expect \"$EXPECT_TREE\" && echo \"experiment/v7 tree OK (v7-single-rc4)\" || fail \"experiment/v7 tree\"\n"
     "python -c \"import dataclasses; from experiment.v7.utils.case_v7 import NumericsConstants as N; from experiment.v7.utils import defrag_prefix; \\\n"
     "    f = {x.name: x.default for x in dataclasses.fields(N)}; assert f['use_prefix_sum_defrag'] is True, f; print('defrag order default: prefix (defrag_prefix imported)')\" \\\n"
     "    || fail \"defrag order default\""),
    ("python -m py_compile experiment/v7_single/_run_single_bench.py experiment/v7_single/utils/simulator_single.py && echo \"py_compile OK\" || fail \"py_compile\"",
     "python -m py_compile experiment/v7_single/_run_single_bench.py experiment/v7_single/utils/simulator_single.py experiment/v7/utils/defrag_prefix.py && echo \"py_compile OK\" || fail \"py_compile\""),
    ("cmp -s \"experiment/v7/shaders/spv/$spv\" \"$HOME/run/vulkan-demo-e8/experiment/v7/shaders/spv/$spv\" && echo \"SPIR-V $spv = e8's\" || fail \"SPIR-V $spv differs from e8's\"",
     "cmp -s \"experiment/v7/shaders/spv/$spv\" \"$HOME/run/vulkan-demo-e9/experiment/v7/shaders/spv/$spv\" && echo \"SPIR-V $spv = e9's\" || fail \"SPIR-V $spv differs from e9's\""),
])
