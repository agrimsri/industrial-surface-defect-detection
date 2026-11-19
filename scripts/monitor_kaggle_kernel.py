import time
import subprocess
import sys
from datetime import datetime

KERNEL_NAME = "agrim26/industrial-defect-training"
POLL_INTERVAL_SEC = 45
MAX_WAIT_MINUTES = 240

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitoring Kaggle kernel: {KERNEL_NAME}")
start_time = time.time()

while True:
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "status", KERNEL_NAME],
            capture_output=True,
            text=True,
            check=False
        )
        output = res.stdout.strip()
        elapsed_min = (time.time() - start_time) / 60.0
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed_min:.1f}m elapsed) {output}", flush=True)
        
        if "KernelWorkerStatus.COMPLETE" in output:
            print(f"✅ Kaggle kernel {KERNEL_NAME} finished successfully in {elapsed_min:.1f} minutes!")
            sys.exit(0)
        elif any(err in output for err in ["KernelWorkerStatus.ERROR", "KernelWorkerStatus.FAILED", "KernelWorkerStatus.CANCELLED"]):
            print(f"❌ Kaggle kernel failed with status: {output}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Warning during status check: {e}")

    if (time.time() - start_time) > (MAX_WAIT_MINUTES * 60):
        print(f"❌ Timeout reached ({MAX_WAIT_MINUTES} minutes). Exiting.")
        sys.exit(1)

    time.sleep(POLL_INTERVAL_SEC)
