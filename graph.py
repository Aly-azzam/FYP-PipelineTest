import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
INPUT_CSV = "hamer_gpu_log.csv"
OUTPUT_MEMORY_PNG = "hamer_gpu_memory_over_time.png"
OUTPUT_UTIL_PNG = "hamer_gpu_utilization_over_time.png"

# =========================
# LOAD CSV
# =========================
# Your nvidia-smi file is UTF-16
df = pd.read_csv(INPUT_CSV, encoding="utf-16")
df.columns = [c.strip() for c in df.columns]

# =========================
# CLEAN COLUMNS
# =========================
# Remove units and convert to numbers
df["utilization.gpu [%]"] = (
    df["utilization.gpu [%]"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)
df["utilization.memory [%]"] = (
    df["utilization.memory [%]"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)
df["memory.used [MiB]"] = (
    df["memory.used [MiB]"]
    .astype(str)
    .str.replace("MiB", "", regex=False)
    .str.strip()
)
df["memory.total [MiB]"] = (
    df["memory.total [MiB]"]
    .astype(str)
    .str.replace("MiB", "", regex=False)
    .str.strip()
)

# Convert invalid values to NaN safely
df["utilization.gpu [%]"] = pd.to_numeric(df["utilization.gpu [%]"], errors="coerce")
df["utilization.memory [%]"] = pd.to_numeric(df["utilization.memory [%]"], errors="coerce")
df["memory.used [MiB]"] = pd.to_numeric(df["memory.used [MiB]"], errors="coerce")
df["memory.total [MiB]"] = pd.to_numeric(df["memory.total [MiB]"], errors="coerce")

# Drop rows where main columns are missing
df = df.dropna(subset=["memory.used [MiB]"]).copy()

# =========================
# TIME TO ELAPSED SECONDS
# =========================
# Example format: 2026/04/11 22:44:44.788
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
df = df.dropna(subset=["timestamp"]).copy()

start_time = df["timestamp"].iloc[0]
df["elapsed_seconds"] = (df["timestamp"] - start_time).dt.total_seconds()

# =========================
# SUMMARY STATS
# =========================
peak_mem = df["memory.used [MiB]"].max()
avg_gpu_util = df["utilization.gpu [%]"].mean(skipna=True)
avg_mem_util = df["utilization.memory [%]"].mean(skipna=True)

print("Fast-HaMeR GPU stats")
print(f"Peak GPU memory: {peak_mem:.0f} MB")
print(f"Average GPU utilization: {avg_gpu_util:.1f}%")
print(f"Average GPU memory utilization: {avg_mem_util:.1f}%")

# =========================
# PLOT 1: GPU MEMORY OVER TIME
# =========================
plt.figure(figsize=(12, 6))
plt.plot(df["elapsed_seconds"], df["memory.used [MiB]"], linewidth=2)
plt.title("Fast-HaMeR GPU (RTX 3060) Memory Usage Over Time")
plt.xlabel("Elapsed seconds")
plt.ylabel("GPU memory used (MB)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_MEMORY_PNG, dpi=200)
plt.close()

# =========================
# PLOT 2: GPU UTILIZATION OVER TIME
# =========================
# Keep only rows where utilization is valid
df_util = df.dropna(subset=["utilization.gpu [%]"]).copy()

plt.figure(figsize=(12, 6))
plt.plot(df_util["elapsed_seconds"], df_util["utilization.gpu [%]"], linewidth=2)
plt.title("Fast-HaMeR GPU (RTX 3060) Utilization Over Time")
plt.xlabel("Elapsed seconds")
plt.ylabel("GPU utilization (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_UTIL_PNG, dpi=200)
plt.close()

print(f"Saved: {OUTPUT_MEMORY_PNG}")
print(f"Saved: {OUTPUT_UTIL_PNG}")