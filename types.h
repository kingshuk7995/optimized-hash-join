#ifndef TYPES_H
#define TYPES_H

#define _GNU_SOURCE

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <liburing.h>
#include <sys/types.h>

#define MAX_COLS 64
#define NUM_PRTNS 256
#define OUT_BUF_SIZE (16 * 1024 * 1024)

#define CHUNK_SIZE (2 * 1024 * 1024)
#define QUEUE_DEPTH 16
#define ALIGNMENT 4096

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

typedef struct {
  const char *data;
  size_t len;
} StringView;

typedef struct {
  StringView cols[MAX_COLS];
  int col_count;
} CsvRow;

typedef struct {
  uint64_t key;
  uint32_t row_idx;
  bool occupied;
} HashEntry;

typedef struct {
  HashEntry *entries;
  size_t capacity;
  size_t mask;
  size_t count;
} Partition;

typedef struct {
  int fd;
  size_t file_size;
  // io_uring
  off_t submit_offset;
  off_t completed_offset;
  int pending_reads;
  size_t tail_len;
  char *tail_ptr;
  char *q_io_bufs[QUEUE_DEPTH];
  char *q_work_buf;
  size_t q_work_buf_size;
  struct io_uring ring;
  // mmap
  char *mmap_ptr;
  off_t current_offset;
} ProbeStream;

typedef struct {
  void *(*load_build)(const char *path, size_t *out_size);
  void  (*stream_probe_init)(const char *path, ProbeStream *s);
  bool  (*stream_probe_next)(ProbeStream *s, char **buf, size_t *len);
  void  (*stream_probe_close)(ProbeStream *s);
} IoBackend;

typedef struct {
  size_t (*count_rows)(const char *p, size_t n);
  bool   (*csv_next)(const char **cur, const char *end, CsvRow *row);
} ParseBackend;

#endif
