import pandas as pd
import numpy as np
import subprocess
import time
import os
import sys


# Configuration
P_ROWS = 1000_000
Q_ROWS = 800_000
UNIQUE_KEYS = 500_000
P_FILE = "build/P.csv"
Q_FILE = "build/Q.csv"
OUT_FILE = "build/output.csv"
BINARY_PATH = "./build/hash_join"


# Data Generation
def generate_data():
    print(f"Generating data: P ({P_ROWS} rows), Q ({Q_ROWS} rows)...")
    np.random.seed(42)

    # Generate K1, K2 pairs
    k1_pool = np.random.randint(1000, 1000 + UNIQUE_KEYS, size=max(P_ROWS, Q_ROWS))
    k2_pool = np.random.randint(5000, 5000 + UNIQUE_KEYS, size=max(P_ROWS, Q_ROWS))

    # Create DataFrames
    df_p = pd.DataFrame(
        {
            "k1": k1_pool[:P_ROWS],
            "k2": k2_pool[:P_ROWS],
            "p_val1": np.random.rand(P_ROWS).round(4),
            "p_val2": np.random.choice(["A", "B", "C", "D"], P_ROWS),
        }
    )

    df_q = pd.DataFrame(
        {
            "k1": k1_pool[:Q_ROWS],  # Force overlap
            "k2": k2_pool[:Q_ROWS],
            "q_val1": np.random.rand(Q_ROWS).round(4),
            "q_val2": np.random.choice(["X", "Y", "Z"], Q_ROWS),
        }
    )

    # Shuffle to ensure non-trivial ordering
    df_p = df_p.sample(frac=1).reset_index(drop=True)
    df_q = df_q.sample(frac=1).reset_index(drop=True)

    df_p.to_csv(P_FILE, index=False)
    df_q.to_csv(Q_FILE, index=False)
    print("Data generation complete.")
    return df_p, df_q


# Execution & Benchmarking
def run_binary():
    if not os.path.exists(BINARY_PATH):
        print(f"FATAL: Binary not found at {BINARY_PATH}")
        sys.exit(1)

    print(f"\nExecuting C binary: {BINARY_PATH} {P_FILE} {Q_FILE}")

    start_time = time.perf_counter()
    with open(OUT_FILE, "w") as f_out:
        result = subprocess.run(
            [BINARY_PATH, P_FILE, Q_FILE], stdout=f_out, stderr=subprocess.PIPE
        )
    end_time = time.perf_counter()

    if result.returncode != 0:
        print("Execution failed!")
        print("STDERR:\n", result.stderr.decode())
        sys.exit(result.returncode)

    exec_time = end_time - start_time
    print(f"Binary execution time: {exec_time:.4f} seconds")
    return exec_time


# Pandas Ground Truth & Verification
def verify_correctness(df_p, df_q):
    print("\nCalculating ground truth using Pandas inner merge...")

    start_time = time.perf_counter()
    # Pandas inner join
    expected_df = pd.merge(df_p, df_q, on=["k1", "k2"], how="inner")
    pd_time = time.perf_counter() - start_time
    print(f"Pandas merge time:     {pd_time:.4f} seconds")

    # The C code outputs columns in a specific order:
    # p_c1, p_c2, (p_rest), (q_rest)
    c_output_columns = ["k1", "k2", "p_val1", "p_val2", "q_val1", "q_val2"]
    expected_df = expected_df[c_output_columns]

    print("\nLoading binary output...")
    try:
        actual_df = pd.read_csv(OUT_FILE)
    except Exception as e:
        print(f"Failed to read binary output: {e}")
        sys.exit(1)

    # Check column counts
    if len(actual_df.columns) != len(expected_df.columns):
        print(
            f"FAIL: Column count mismatch. Expected {len(expected_df.columns)}, got {len(actual_df.columns)}"
        )
        sys.exit(1)

    # Rename actual columns to match expected for easy comparison
    # (since the C code headers might just match the raw strings)
    actual_df.columns = c_output_columns

    # Check row counts
    if len(actual_df) != len(expected_df):
        print(
            f"FAIL: Row count mismatch. Expected {len(expected_df)} rows, got {len(actual_df)} rows."
        )
        sys.exit(1)

    print(f"Row counts match: {len(expected_df)} rows.")
    print("Verifying data integrity (sorting both datasets)...")

    # Sort both dataframes to handle arbitrary hash map ordering
    expected_sorted = expected_df.sort_values(by=c_output_columns).reset_index(
        drop=True
    )
    actual_sorted = actual_df.sort_values(by=c_output_columns).reset_index(drop=True)

    # Type alignment for robust checking (pandas might cast floats slightly differently than C parsing)
    # Since we are printing exact strings from the source CSVs in C, we should compare as strings.
    expected_sorted = expected_sorted.astype(str)
    actual_sorted = actual_sorted.astype(str)

    # Compare
    differences = (expected_sorted != actual_sorted).sum().sum()

    if differences == 0:
        print("\nSUCCESS: Binary output perfectly matches Pandas ground truth!")
    else:
        print(f"\nFAIL: Found {differences} mismatched cells.")
        # Print a sample of mismatched rows
        mask = (expected_sorted != actual_sorted).any(axis=1)
        print("Expected Sample:\n", expected_sorted[mask].head())
        print("Actual Sample:\n", actual_sorted[mask].head())


if __name__ == "__main__":
    df_p, df_q = generate_data()
    run_binary()
    verify_correctness(df_p, df_q)

    # Cleanup
    if os.path.exists(P_FILE):
        os.remove(P_FILE)
    if os.path.exists(Q_FILE):
        os.remove(Q_FILE)
    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)
