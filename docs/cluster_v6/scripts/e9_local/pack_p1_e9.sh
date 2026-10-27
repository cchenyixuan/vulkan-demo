set -u
JOB=$(cat ~/run/tools/e9_p1_jobid.txt)
D=$HOME/run/logs/e7_p1_$JOB
[ -d "$D" ] || { echo "ABORT: no $D"; ls ~/run/logs | tail -5; exit 1; }
cp /data/run01/scxm138/logs/e9_p1_$JOB.out "$D/" 2>/dev/null
cp ~/run/vulkan-demo-e9/docs/cluster_v6/e9_jobs/e9_p1.sbatch "$D/" 2>/dev/null
cp ~/run/vulkan-demo-e9/COMMIT "$D/COMMIT_e9" 2>/dev/null
sacct -j $JOB -o JobID,JobName,Partition,State,Submit,Start,End,Elapsed,NodeList,ExitCode,AllocTRES%60 -P > "$D/sacct.txt" 2>&1
cat "$D/sacct.txt" | head -3
du -sh "$D"; ls "$D/dumps" "$D/compare" "$D/traces" 2>/dev/null | tr '\n' ' '; echo
cd $HOME/run/logs && tar czf e9_p1_$JOB.tar.gz --exclude='*.npz' e7_p1_$JOB && ls -la e9_p1_$JOB.tar.gz && sha256sum e9_p1_$JOB.tar.gz
