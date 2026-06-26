import subprocess
import time
import os
import sys
import statistics
import random
import csv

# Configuration
P_ROWS = 2_000_000
Q_ROWS = 1_500_000
UNIQUE_KEYS = 1_000_000
P_FILE = "build/P_bench.csv"
Q_FILE = "build/Q_bench.csv"
OUT_FILE = "build/output_bench.csv"
BINARY_PATH = "./build/hash_join"
ITERATIONS = 5

def generate_data():
    print(f"Generating data for benchmarking: P ({P_ROWS} rows), Q ({Q_ROWS} rows)...")
    random.seed(42)

    if not os.path.exists("build"):
        os.makedirs("build")

    # Generate P
    with open(P_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["k1", "k2", "p_val1", "p_val2"])
        for _ in range(P_ROWS):
            writer.writerow([
                random.randint(1000, 1000 + UNIQUE_KEYS - 1),
                random.randint(5000, 5000 + UNIQUE_KEYS - 1),
                round(random.random(), 4),
                random.choice(["A", "B", "C", "D"])
            ])

    # Generate Q
    with open(Q_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["k1", "k2", "q_val1", "q_val2"])
        for _ in range(Q_ROWS):
            writer.writerow([
                random.randint(1000, 1000 + UNIQUE_KEYS - 1),
                random.randint(5000, 5000 + UNIQUE_KEYS - 1),
                round(random.random(), 4),
                random.choice(["X", "Y", "Z"])
            ])
    print("Data generation complete.")

def run_benchmark():
    if not os.path.exists(BINARY_PATH):
        print(f"FATAL: Binary not found at {BINARY_PATH}")
        sys.exit(1)

    times = []
    print(f"\nStarting benchmark with {ITERATIONS} iterations...")
    
    for i in range(ITERATIONS):
        print(f"Iteration {i+1}/{ITERATIONS}...", end="", flush=True)
        start_time = time.perf_counter()
        
        with open(OUT_FILE, "w") as f_out:
            result = subprocess.run(
                [BINARY_PATH, P_FILE, Q_FILE], stdout=f_out, stderr=subprocess.PIPE
            )
            
        end_time = time.perf_counter()
        
        if result.returncode != 0:
            print(f" Execution failed!\nSTDERR:\n{result.stderr.decode()}")
            sys.exit(result.returncode)
            
        exec_time = end_time - start_time
        times.append(exec_time)
        print(f" {exec_time:.4f} seconds")

    print("\n--- Benchmark Results ---")
    print(f"Iterations: {ITERATIONS}")
    print(f"Min:    {min(times):.4f} s")
    print(f"Max:    {max(times):.4f} s")
    print(f"Mean:   {statistics.mean(times):.4f} s")
    if ITERATIONS > 1:
        print(f"Stdev:  {statistics.stdev(times):.4f} s")

def cleanup():
    print("\nCleaning up temporary files...")
    for f in [P_FILE, Q_FILE, OUT_FILE]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    try:
        generate_data()
        run_benchmark()
    finally:
        cleanup()
