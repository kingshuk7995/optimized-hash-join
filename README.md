# optimized-hash-join
A optiimized hash join implementation in C, written for the DBMS course

Implemented a optimized disk based hash join utilizing io-uring for disk-IO and SIMD CPU ops.

# Benchmark


## Benchmark Environment

| Component | Specification |
|:----------|:--------------|
| CPU | Intel Core i7-13700HX |
| RAM | 16 GB |
| Storage | NVMe SSD (PCIe Gen 4) |
| OS | WSL2 Archlinux Kernel Version: 6.18.33.1-microsoft-standard-WSL2 |

| Implementation      |    Min (s) |   Mean (s) |    Max (s) | Std Dev (s) | Speedup vs `mmap + linear` |
| :------------------ | ---------: | ---------: | ---------: | ----------: | -------------------------: |
| `mmap + linear`     | **0.5848** | **0.6084** | **0.6319** |  **0.0194** |                  **1.00×** |
| `mmap + SIMD`       |     0.3770 |     0.3971 |     0.4308 |      0.0217 |                  **1.53×** |
| `io_uring + linear` |     0.5618 |     0.5671 |     0.5742 |      0.0054 |                  **1.07×** |
| `io_uring + SIMD`   | **0.3640** | **0.3691** | **0.3803** |  **0.0068** |                  **1.65×** |


## Benchmark Environment

| Component | Specification |
|:----------|:--------------|
| CPU | AMD Ryzen 7 8845HS |
| RAM | 16 GB |
| Storage | NVMe SSD (PCIe Gen 4) |
| OS | Garuda Linux Kernel Version: 7.0.12-zen1-1-zen (64-bit) |

| Implementation      |    Min (s) |   Mean (s) |    Max (s) | Std Dev (s) | Speedup vs `mmap + linear` |
| :------------------ | ---------: | ---------: | ---------: | ----------: | -------------------------: |
| `mmap + linear`     | **0.6703** | **0.6838** | **0.7146** |  **0.0179** |                  **1.00×** |
| `mmap + SIMD`       |     0.4222 |     0.4304 |     0.4349 |      0.0053 |                  **1.59×** |
| `io_uring + linear` |     0.6578 |     0.6663 |     0.6793 |      0.0085 |                  **1.03×** |
| `io_uring + SIMD`   | **0.4113** | **0.4148** | **0.4175** |  **0.0028** |                  **1.65×** |
