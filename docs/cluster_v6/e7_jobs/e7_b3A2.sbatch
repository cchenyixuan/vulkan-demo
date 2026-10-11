#!/bin/bash
#SBATCH -p hp_5090
#SBATCH -A hp5090
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH --time=10:00:00
#SBATCH -J e7_A
#SBATCH --dependency=singleton
#SBATCH --exclude=wqd10nah10g2
#SBATCH -o /data/run01/scxm138/logs/e7_b3A2_%j.out
# E7 full campaign, batch 3, line A, remainder: the five points job 1683993 (e7_b3A.sbatch) did not reach. That job ran
# 3d_n160_k2_K2 and 3d_n160_k4_K4 (34 runs, all pass) and stopped at cal_3d_n200_k2_K2 when GPU 1 of wqd10nah10g2 was
# lost (nvidia-smi "[GPU is lost]", node drained "Kill task failed", 2026-10-11). The e7_point lines are e7_b3A.sbatch's,
# unchanged; the case list is the same; same deployment (v7-rc1). Estimate 4.96 h for the five points (e7_plan.py model;
# the first two points took 0.70 x their estimate). --exclude keeps the job off that node.
source "${E30_REPO:-$HOME/run/vulkan-demo-v7rc1}/docs/cluster_v6/scripts/e7_lib.sh" || { echo "ABORT: no e7_lib.sh"; exit 2; }
e7_job_begin b3A2 A cavity2d_n1000 cavity3d_n160 cavity3d_n160_k2 cavity3d_n160_k4 cavity3d_n160_k8 cavity3d_n200 cavity3d_n200_k2 cavity3d_n200_k4 cavity3d_n200_k8
e7_point F4 cavity3d_n200_k2 2 3 1500 500 900 900 900 2 cavity3d_n200:900
e7_point F4 cavity3d_n200_k4 4 3 1500 500 900 900 960 2 cavity3d_n200:900
e7_point F5 cavity3d_n160_k8 2 3 1500 500 1020 1080 1140 2 same:1440
e7_point F5 cavity3d_n200_k8 2 3 1500 500 1380 1500 1560 2 same:2100
e7_point F5 cavity3d_n200_k8 4 3 1500 500 1200 1140 1200 2 same:2100
e7_job_end
