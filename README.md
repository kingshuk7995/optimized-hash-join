# optimized-hash-join
A optiimized hash join implementation in C, written for the DBMS course

Implemented a optimized disk based hash join utilizing io-uring for disk-IO and SIMD CPU ops.

# Benchmark

On **WSL2**

| Implementation      |    Min (s) |   Mean (s) |    Max (s) | Std Dev (s) | Speedup vs `mmap + linear` |
| :------------------ | ---------: | ---------: | ---------: | ----------: | -------------------------: |
| `mmap + linear`     | **0.5848** | **0.6084** | **0.6319** |  **0.0194** |                  **1.00×** |
| `mmap + SIMD`       |     0.3770 |     0.3971 |     0.4308 |      0.0217 |                  **1.53×** |
| `io_uring + linear` |     0.5618 |     0.5671 |     0.5742 |      0.0054 |                  **1.07×** |
| `io_uring + SIMD`   | **0.3640** | **0.3691** | **0.3803** |  **0.0068** |                  **1.65×** |
