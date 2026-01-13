import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics.csv")

# Average metrics by mode
avg = df.groupby("mode").mean(numeric_only=True)

# Plot latency breakdown
plt.figure()
avg[["cpu_time_ms", "inference_time_ms"]].plot(kind="bar")
plt.title("CPU vs DSP Inference Time")
plt.ylabel("Milliseconds")
plt.xlabel("Execution Mode")
plt.tight_layout()
plt.savefig("results/latency_breakdown.png")
plt.close()

# Plot total latency comparison
plt.figure()
avg["total_latency_ms"].plot(kind="bar")
plt.title("Total Latency: Sequential vs Pipelined")
plt.ylabel("Milliseconds")
plt.xlabel("Execution Mode")
plt.tight_layout()
plt.savefig("results/total_latency.png")
plt.close()

print("Plots saved to results/")
