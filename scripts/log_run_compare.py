import re
from datetime import datetime
import matplotlib.pyplot as plt

# ---- CONFIG ----
log_files = {
    "run_1_small": "files/experiments/t001/t001_small/training.log",
    "run_1_embedding" : "files/experiments/t001/t001_embedding/training.log",
    "run_1_patch" : "files/experiments/t001/t001_patch/training.log"
}

pattern = re.compile(r"Epoch \[(\d+)/\d+\] - Loss: ([0-9.]+)")
time_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

def parse_log(filepath):
    epochs = []
    losses = []
    timestamps = []

    with open(filepath, "r") as f:
        for line in f:
            # extract timestamp
            t_match = time_pattern.match(line)
            if t_match:
                timestamps.append(datetime.strptime(t_match.group(1), "%Y-%m-%d %H:%M:%S"))

            # extract epoch + loss
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    # compute runtime
    runtime = None
    if timestamps:
        runtime = timestamps[-1] - timestamps[0]

    return epochs, losses, runtime


# ---- PARSE ALL LOGS ----
results = {}

for name, path in log_files.items():
    epochs, losses, runtime = parse_log(path)
    results[name] = (epochs, losses, runtime)


# ---- PLOT ----
plt.figure()

title_parts = []

for name, (epochs, losses, runtime) in results.items():
    plt.plot(epochs, losses, marker='o', label=name)

    if runtime:
        # format runtime nicely
        total_seconds = int(runtime.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        title_parts.append(f"{name}: {hours}h {minutes}m {seconds}s")

plt.xlabel("Epoch")
plt.ylabel("Loss")

# combine runtimes into title
plt.title("Training Loss Comparison\n" + " | ".join(title_parts))

plt.legend()
plt.grid()

plt.show()