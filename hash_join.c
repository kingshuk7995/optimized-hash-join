#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <liburing.h>
#include <nmmintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_COLS 64
#define OUT_BUF_SIZE (16 * 1024 * 1024)
#define NUM_PRTNS 256
#define CHUNK_SIZE (2 * 1024 * 1024)
#define ALIGNMENT 4096
#define QUEUE_DEPTH 16

typedef struct {
  const char *data;
  size_t len;
} StringView;

typedef struct {
  StringView cols[MAX_COLS];
  int col_count;
} CsvRow;

// CPU Verification
void verify_cpu_features() {
  __builtin_cpu_init();
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("sse4.2") ||
      !__builtin_cpu_supports("popcnt")) {
    fprintf(stderr,
            "FATAL: CPU lacks required AVX2/SSE4.2/POPCNT instructions.\n");
    exit(EXIT_FAILURE);
  }
}

// Fast Buffered Output
char *out_buf;
size_t out_pos = 0;

static inline void flush_out() {
  if (out_pos > 0) {
    if (write(STDOUT_FILENO, out_buf, out_pos) == -1) {
      perror("write");
      exit(EXIT_FAILURE);
    }
    out_pos = 0;
  }
}

static inline void write_out(const char *data, size_t len) {
  if (out_pos + len > OUT_BUF_SIZE)
    flush_out();
  memcpy(out_buf + out_pos, data, len);
  out_pos += len;
}

static inline void write_char(char c) {
  if (out_pos + 1 > OUT_BUF_SIZE)
    flush_out();
  out_buf[out_pos++] = c;
}

// --- SIMD CSV Parsing & Hashing ---
__attribute__((target("avx2,popcnt"))) size_t count_rows_simd(const char *ptr,
                                                              size_t size) {
  size_t count = 0;
  const char *end = ptr + size;
  __m256i newlines = _mm256_set1_epi8('\n');

  while (ptr + 32 <= end) {
    __m256i chunk = _mm256_loadu_si256((const __m256i *)ptr);
    uint32_t mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, newlines));
    count += __builtin_popcount(mask);
    ptr += 32;
  }
  while (ptr < end) {
    if (*ptr == '\n')
      count++;
    ptr++;
  }
  if (size > 0 && *(end - 1) != '\n')
    count++;
  return count;
}

__attribute__((target("avx2"))) bool csv_next(const char **current,
                                              const char *end, CsvRow *row) {
  if (*current >= end)
    return false;
  row->col_count = 0;
  const char *ptr = *current;
  const char *tok_st = ptr;
  __m256i commas = _mm256_set1_epi8(',');
  __m256i newlines = _mm256_set1_epi8('\n');

  while (ptr < end) {
    if (end - ptr >= 32) {
      __m256i chunk = _mm256_loadu_si256((const __m256i *)ptr);
      uint32_t msk = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, commas)) |
                     _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, newlines));

      while (msk != 0) {
        uint32_t ofst = __builtin_ctz(msk);
        char c = ptr[ofst];
        size_t len = (ptr + ofst) - tok_st;
        if (len > 0 && tok_st[len - 1] == '\r')
          len--;

        if (row->col_count < MAX_COLS)
          row->cols[row->col_count++] = (StringView){tok_st, len};

        tok_st = ptr + ofst + 1;

        if (c == '\n') {
          *current = ptr + ofst + 1;
          return true;
        }
        msk &= msk - 1;
      }
      ptr += 32;
    } else {
      char c = *ptr;
      if (c == ',' || c == '\n') {
        size_t len = ptr - tok_st;
        if (len > 0 && tok_st[len - 1] == '\r')
          len--;
        if (row->col_count < MAX_COLS)
          row->cols[row->col_count++] = (StringView){tok_st, len};
        tok_st = ptr + 1;
        if (c == '\n') {
          *current = ptr + 1;
          return true;
        }
      }
      ptr++;
    }
  }
  if (tok_st < end && row->col_count < MAX_COLS) {
    size_t len = end - tok_st;
    if (len > 0 && tok_st[len - 1] == '\r')
      len--;
    row->cols[row->col_count++] = (StringView){tok_st, len};
  }
  *current = end;
  return row->col_count > 0;
}

__attribute__((target("sse4.2"))) static inline uint64_t
hash_view(StringView v) {
  uint64_t hash = 0;
  const char *ptr = v.data;
  size_t len = v.len;
  while (len >= 8) {
    uint64_t chunk;
    memcpy(&chunk, ptr, 8);
    hash = _mm_crc32_u64(hash, chunk);
    ptr += 8;
    len -= 8;
  }
  while (len > 0) {
    hash = _mm_crc32_u8((uint32_t)hash, *ptr);
    ptr++;
    len--;
  }
  return hash;
}

static inline uint64_t fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

static inline uint64_t combine_hashes(uint64_t h1, uint64_t h2) {
  uint64_t combined = h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  return fmix64(
      combined); // Distribute bits to fix the key >> 56 partition clustering
}

static inline bool view_eq(StringView a, StringView b) {
  return a.len == b.len && memcmp(a.data, b.data, a.len) == 0;
}

// Partitioned Radix Hash Map
typedef struct {
  uint64_t key;
  uint32_t row_idx;
  bool occupied;
} HashEntry;

typedef struct {
  HashEntry *entries;
  size_t capacity;
  size_t mask;
} Partition;

Partition prtns[NUM_PRTNS];
StringView *p_cell_arena;
int p_cols_per_row = 0;
size_t p_pool_size = 0;

void init_prtns(size_t exact_rows, int cols_per_row) {
  size_t cap_per_part = 1;
  size_t target = (exact_rows / NUM_PRTNS) * 4 + 1024;
  while (cap_per_part < target)
    cap_per_part <<= 1;

  for (int i = 0; i < NUM_PRTNS; ++i) {
    prtns[i].capacity = cap_per_part;
    prtns[i].mask = cap_per_part - 1;
    prtns[i].entries = calloc(cap_per_part, sizeof(HashEntry));
  }

  p_cols_per_row = cols_per_row;
  p_cell_arena = malloc(exact_rows * p_cols_per_row * sizeof(StringView));
}

void insert_partitioned(uint64_t key, CsvRow *row) {
  uint32_t idx = p_pool_size++;
  uint32_t arena_ofst = idx * p_cols_per_row;
  memcpy(&p_cell_arena[arena_ofst], row->cols,
         p_cols_per_row * sizeof(StringView));

  uint32_t part_idx = key >> 56;
  Partition *p = &prtns[part_idx];

  size_t slot = key & p->mask;
  while (p->entries[slot].occupied)
    slot = (slot + 1) & p->mask;

  p->entries[slot].key = key;
  p->entries[slot].row_idx = idx;
  p->entries[slot].occupied = true;
}

// Output Formatting
static inline void print_joined_row(StringView *p_cols, int p_col_count,
                                    CsvRow *q_row, int p_k1, int p_k2, int q_k1,
                                    int q_k2) {
  write_out(p_cols[p_k1].data, p_cols[p_k1].len);
  write_char(',');
  write_out(p_cols[p_k2].data, p_cols[p_k2].len);

  // Prepend commas to avoid tracking trailing elements and buffer wrap-arounds
  for (int i = 0; i < p_col_count; ++i) {
    if (i == p_k1 || i == p_k2)
      continue;
    write_char(',');
    write_out(p_cols[i].data, p_cols[i].len);
  }

  for (int i = 0; i < q_row->col_count; ++i) {
    if (i == q_k1 || i == q_k2)
      continue;
    write_char(',');
    write_out(q_row->cols[i].data, q_row->cols[i].len);
  }

  write_char('\n');
}

// io_uring Helpers
char *bulk_load_file_uring(const char *filename, size_t *out_size) {
  int fd = open(filename, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    fd = open(filename, O_RDONLY); // Fallback for tmpfs/ZFS
    if (fd < 0) {
      perror("open P");
      exit(1);
    }
  }

  struct stat sb;
  fstat(fd, &sb);
  *out_size = sb.st_size;

  size_t alloc_size = (sb.st_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
  char *buf;
  if (posix_memalign((void **)&buf, ALIGNMENT, alloc_size)) {
    perror("memalign");
    exit(1);
  }

  struct io_uring ring;
  io_uring_queue_init(QUEUE_DEPTH, &ring, 0);

  off_t offset = 0;
  int submissions = 0;

  while (offset < sb.st_size) {
    while (submissions < QUEUE_DEPTH && offset < sb.st_size) {
      size_t read_bytes =
          alloc_size - offset > CHUNK_SIZE ? CHUNK_SIZE : alloc_size - offset;
      struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
      io_uring_prep_read(sqe, fd, buf + offset, read_bytes, offset);
      io_uring_sqe_set_data(sqe, (void *)1);
      io_uring_submit(&ring);
      offset += read_bytes;
      submissions++;
    }

    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&ring, &cqe);
    if (cqe->res < 0) {
      fprintf(stderr, "Async read failed\n");
      exit(1);
    }
    io_uring_cqe_seen(&ring, cqe);
    submissions--;
  }

  while (submissions > 0) {
    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&ring, &cqe);
    io_uring_cqe_seen(&ring, cqe);
    submissions--;
  }

  io_uring_queue_exit(&ring);
  close(fd);
  return buf;
}

// Join Execution
__attribute__((target("avx2,sse4.2,popcnt"))) void
execute_join(const char *p_file, const char *q_file) {
  out_buf = malloc(OUT_BUF_SIZE);

  // Bulk Load P completely to memory
  size_t p_size;
  char *p_buf = bulk_load_file_uring(p_file, &p_size);
  const char *p_ptr = p_buf;
  const char *p_end = p_buf + p_size;

  CsvRow header_p;
  csv_next(&p_ptr, p_end, &header_p);

  size_t exact_p_rows = count_rows_simd(p_ptr, p_end - p_ptr);
  init_prtns(exact_p_rows, header_p.col_count);

  // Setup Async Streaming for Q
  int q_fd = open(q_file, O_RDONLY | O_DIRECT);
  if (q_fd < 0) {
    q_fd = open(q_file, O_RDONLY); // Fallback for tmpfs/ZFS
    if (q_fd < 0) {
      perror("open Q");
      exit(1);
    }
  }

  struct stat q_sb;
  fstat(q_fd, &q_sb);
  size_t q_file_size = q_sb.st_size;

  struct io_uring ring;
  io_uring_queue_init(QUEUE_DEPTH, &ring, 0);

  char *q_io_bufs[QUEUE_DEPTH];
  for (int i = 0; i < QUEUE_DEPTH; i++) {
    posix_memalign((void **)&q_io_bufs[i], ALIGNMENT, CHUNK_SIZE);
  }
  char *q_work_buf = malloc(CHUNK_SIZE * 2);

  // Read first chunk of Q to get header
  struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
  io_uring_prep_read(sqe, q_fd, q_io_bufs[0], CHUNK_SIZE, 0);
  io_uring_sqe_set_data(sqe, (void *)0);
  io_uring_submit(&ring);

  struct io_uring_cqe *cqe;
  io_uring_wait_cqe(&ring, &cqe);
  int first_read = cqe->res;
  io_uring_cqe_seen(&ring, cqe);

  if (q_file_size < first_read) {
    first_read = q_file_size; // Trim EOF padding
  }
  memcpy(q_work_buf, q_io_bufs[0], first_read);

  CsvRow header_q;
  const char *q_ptr_initial = q_work_buf;
  const char *q_end_initial = q_work_buf + first_read;
  csv_next(&q_ptr_initial, q_end_initial, &header_q);

  // Match keys
  int p_keys[2] = {-1, -1};
  int q_keys[2] = {-1, -1};
  int keys_found = 0;

  for (int i = 0; i < header_p.col_count && keys_found < 2; ++i) {
    for (int j = 0; j < header_q.col_count && keys_found < 2; ++j) {
      if (view_eq(header_p.cols[i], header_q.cols[j])) {
        p_keys[keys_found] = i;
        q_keys[keys_found] = j;
        keys_found++;
      }
    }
  }
  if (keys_found != 2) {
    fprintf(stderr, "FATAL: Need exactly 2 common attributes.\n");
    exit(1);
  }

  int p_c1 = p_keys[0], p_c2 = p_keys[1], q_c1 = q_keys[0], q_c2 = q_keys[1];
  print_joined_row(header_p.cols, header_p.col_count, &header_q, p_c1, p_c2,
                   q_c1, q_c2);

  // Build Hash Map for P
  CsvRow row;
  while (csv_next(&p_ptr, p_end, &row)) {
    if (row.col_count != header_p.col_count)
      continue;
    uint64_t key =
        combine_hashes(hash_view(row.cols[p_c1]), hash_view(row.cols[p_c2]));
    insert_partitioned(key, &row);
  }

  // Stream and Probe Q (Deadlock Fixed)
  off_t submit_offset = CHUNK_SIZE;
  off_t completed_offset = CHUNK_SIZE;
  int pending_reads = 0;
  size_t tail_len = q_end_initial - q_ptr_initial;

  if (tail_len > 0)
    memmove(q_work_buf, q_ptr_initial, tail_len);

  // Kickstart pipelined read
  if (submit_offset < q_file_size) {
    size_t next_sz =
        (q_file_size - submit_offset < CHUNK_SIZE)
            ? ((q_file_size - submit_offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1))
            : CHUNK_SIZE;
    sqe = io_uring_get_sqe(&ring);
    io_uring_prep_read(sqe, q_fd, q_io_bufs[1], next_sz, submit_offset);
    io_uring_sqe_set_data(sqe, (void *)1);
    io_uring_submit(&ring);
    pending_reads++;
    submit_offset += next_sz;
  }

  while (pending_reads > 0 || tail_len > 0) {
    int bytes_read = 0;
    long buf_idx = -1;

    if (pending_reads > 0) {
      io_uring_wait_cqe(&ring, &cqe);
      bytes_read = cqe->res;
      buf_idx = (long)io_uring_cqe_get_data(cqe);
      io_uring_cqe_seen(&ring, cqe);
      pending_reads--;

      if (bytes_read > 0) {
        if (submit_offset < q_file_size) {
          long next_idx = (buf_idx + 1) % QUEUE_DEPTH;
          size_t next_sz =
              (q_file_size - submit_offset < CHUNK_SIZE)
                  ? ((q_file_size - submit_offset + ALIGNMENT - 1) &
                     ~(ALIGNMENT - 1))
                  : CHUNK_SIZE;
          sqe = io_uring_get_sqe(&ring);
          io_uring_prep_read(sqe, q_fd, q_io_bufs[next_idx], next_sz,
                             submit_offset);
          io_uring_sqe_set_data(sqe, (void *)next_idx);
          io_uring_submit(&ring);
          pending_reads++;
          submit_offset += next_sz;
        }

        if (completed_offset + bytes_read > q_file_size) {
          bytes_read = q_file_size - completed_offset;
        }
        completed_offset += bytes_read;
        memcpy(q_work_buf + tail_len, q_io_bufs[buf_idx], bytes_read);
      }
    }

    const char *work_ptr = q_work_buf;
    const char *work_end = q_work_buf + tail_len + bytes_read;

    // Handle EOF
    if (pending_reads == 0 && submit_offset >= q_file_size) {
      while (work_ptr < work_end) {
        if (csv_next(&work_ptr, work_end, &row)) {
          if (row.col_count != header_q.col_count)
            continue;
          StringView qc1_v = row.cols[q_c1], qc2_v = row.cols[q_c2];
          uint64_t key = combine_hashes(hash_view(qc1_v), hash_view(qc2_v));
          uint32_t part_idx = key >> 56;
          Partition *p = &prtns[part_idx];

          size_t slot = key & p->mask;
          while (p->entries[slot].occupied) {
            if (p->entries[slot].key == key) {
              StringView *matched =
                  &p_cell_arena[p->entries[slot].row_idx * p_cols_per_row];
              if (view_eq(matched[p_c1], qc1_v) &&
                  view_eq(matched[p_c2], qc2_v)) {
                print_joined_row(matched, p_cols_per_row, &row, p_c1, p_c2,
                                 q_c1, q_c2);
              }
            }
            slot = (slot + 1) & p->mask;
          }
        } else {
          break;
        }
      }
      tail_len = 0;
    } else {
      const char *last_newline = work_end - 1;
      while (last_newline >= work_ptr && *last_newline != '\n')
        last_newline--;

      while (work_ptr <= last_newline) {
        if (csv_next(&work_ptr, last_newline + 1, &row)) {
          if (row.col_count != header_q.col_count)
            continue;
          StringView qc1_v = row.cols[q_c1], qc2_v = row.cols[q_c2];
          uint64_t key = combine_hashes(hash_view(qc1_v), hash_view(qc2_v));
          uint32_t part_idx = key >> 56;
          Partition *p = &prtns[part_idx];

          size_t slot = key & p->mask;
          while (p->entries[slot].occupied) {
            if (p->entries[slot].key == key) {
              StringView *matched =
                  &p_cell_arena[p->entries[slot].row_idx * p_cols_per_row];
              if (view_eq(matched[p_c1], qc1_v) &&
                  view_eq(matched[p_c2], qc2_v)) {
                print_joined_row(matched, p_cols_per_row, &row, p_c1, p_c2,
                                 q_c1, q_c2);
              }
            }
            slot = (slot + 1) & p->mask;
          }
        }
      }
      tail_len = work_end - work_ptr;
      if (tail_len > 0)
        memmove(q_work_buf, work_ptr, tail_len);
    }
  }

  flush_out();

  for (int i = 0; i < NUM_PRTNS; ++i)
    free(prtns[i].entries);
  free(p_cell_arena);
  free(p_buf);
  free(q_work_buf);
  for (int i = 0; i < QUEUE_DEPTH; i++)
    free(q_io_bufs[i]);
  free(out_buf);
  io_uring_queue_exit(&ring);
  close(q_fd);
}

int main(int argc, char **argv) {
  verify_cpu_features();
  if (argc != 3) {
    fprintf(stderr, "usage: %s <P.csv> <Q.csv>\n", argv[0]);
    return 1;
  }
  execute_join(argv[1], argv[2]);
  return 0;
}
