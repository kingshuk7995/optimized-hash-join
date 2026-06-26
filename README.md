# optimized-hash-join
A optiimized hash join implementation in C, written for the DBMS course

Implemented a optimized disk based hash join utilizing io-uring for disk-IO and SIMD CPU ops.

# Benchmark

On **WSL2**

Output:
kp ~/development/optimized-hash-join$ make benchmark
python3 benchmark.py
Generating data for benchmarking: P (2000000 rows), Q (1500000 rows)...
Data generation complete.

Starting benchmark with 5 iterations...
Iteration 1/5... 0.4921 seconds
Iteration 2/5... 0.3890 seconds
Iteration 3/5... 0.3747 seconds
Iteration 4/5... 0.3798 seconds
Iteration 5/5... 0.3834 seconds

--- Benchmark Results ---
Iterations: 5
Min:    0.3747 s
Max:    0.4921 s
Mean:   0.4038 s
Stdev:  0.0497 s

Cleaning up temporary files...
