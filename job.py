import os
import time
import subprocess
from datetime import datetime
import cudaq

# ==== Job & System Info ====
job_id = os.environ.get("SLURM_JOB_ID", "manual")
container_name = os.environ.get("USER", "unknown")
node = os.uname()[1]
start_time = int(time.time())

# ==== Simulate Job Work =====

notebook_path = "vqe.py"
with open(notebook_path) as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
try:
    ep.preprocess(nb, {'metadata': {'path': './'}})
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
except Exception as e:
    print("Error executing notebook:", e)

# ==== GPU Memory Usage Tracking ====
pid = os.getpid()
mem_used = "0"

nvsmi_out = subprocess.getoutput(
    "nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits"
)

for line in nvsmi_out.strip().splitlines():
    parts = [x.strip() for x in line.split(',')]
    if len(parts) == 2 and int(parts[0]) == pid:
        mem_used = parts[1]
        break

# ==== Log Execution Info ====
end_time = int(time.time())
total_duration = end_time - start_time

log_file = f"/workspace/gpu_usage_{container_name}.csv"
log_line = (
    f"{container_name},{job_id},"
    f"{datetime.fromtimestamp(start_time)},"
    f"{datetime.fromtimestamp(end_time)},"
    f"{total_duration},{node},{mem_used}\n"
)
header = "ContainerName,JobID,StartTime,EndTime,Duration,Node,MemUsedMB\n"

if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write(header)

with open(log_file, "a") as f:
    f.write(log_line)

print(" ^|^e Execution log saved to:", log_file)
