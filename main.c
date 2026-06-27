#define _GNU_SOURCE
#include "types.h"
#include "mmap_io.h"
#include "io_uring_io.h"
#include "csv_linear_parse.h"
#include "csv_simd_parse.h"
#include "hash_join.h"

char *out_buf;
size_t out_pos = 0;

void verify_cpu_features() {
  __builtin_cpu_init();
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("sse4.2") ||
      !__builtin_cpu_supports("popcnt")) {
    fprintf(stderr,
            "FATAL: CPU lacks required AVX2/SSE4.2/POPCNT instructions.\n");
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv) {
  verify_cpu_features();
  if (argc != 5) {
    fprintf(stderr, "usage: %s [mmap|uring] [linear|simd] <P.csv> <Q.csv>\n", argv[0]);
    return 1;
  }

  IoBackend *io = NULL;
  ParseBackend *pb = NULL;

  if (strcmp(argv[1], "mmap") == 0) io = &io_backend_mmap;
  else if (strcmp(argv[1], "uring") == 0) io = &io_backend_uring;
  else { fprintf(stderr, "Invalid io backend\n"); return 1; }

  if (strcmp(argv[2], "linear") == 0) pb = &parse_backend_linear;
  else if (strcmp(argv[2], "simd") == 0) pb = &parse_backend_simd;
  else { fprintf(stderr, "Invalid parse backend\n"); return 1; }

  execute_join(io, pb, argv[3], argv[4]);
  return 0;
}
