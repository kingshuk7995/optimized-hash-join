#ifndef HASH_JOIN_H
#define HASH_JOIN_H

#include "types.h"
#include <immintrin.h>
#include <nmmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern char *out_buf;
extern size_t out_pos;

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
  if (unlikely(len >= OUT_BUF_SIZE)) {
    flush_out();
    if (write(STDOUT_FILENO, data, len) == -1) {
      perror("write large chunk");
      exit(EXIT_FAILURE);
    }
    return;
  }
  if (unlikely(out_pos + len > OUT_BUF_SIZE))
    flush_out();
  memcpy(out_buf + out_pos, data, len);
  out_pos += len;
}

static inline void write_char(char c) {
  if (unlikely(out_pos + 1 > OUT_BUF_SIZE))
    flush_out();
  out_buf[out_pos++] = c;
}

__attribute__((target("sse4.2"))) static inline uint64_t
hash_view(StringView v) {
  uint64_t hash = 0;
  if (v.len >= 2 && v.data[0] == '"' && v.data[v.len - 1] == '"') {
    v.data++; v.len -= 2;
  }
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
  return fmix64(combined);
}

static inline bool view_eq(StringView a, StringView b) {
  if (a.len >= 2 && a.data[0] == '"' && a.data[a.len - 1] == '"') {
    a.data++; a.len -= 2;
  }
  if (b.len >= 2 && b.data[0] == '"' && b.data[b.len - 1] == '"') {
    b.data++; b.len -= 2;
  }
  return a.len == b.len && memcmp(a.data, b.data, a.len) == 0;
}

static inline void print_joined_row(StringView *p_cols, int p_col_count,
                                    CsvRow *q_row, int p_k1, int p_k2, int q_k1,
                                    int q_k2) {
  write_out(p_cols[p_k1].data, p_cols[p_k1].len); write_char(',');
  write_out(p_cols[p_k2].data, p_cols[p_k2].len);
  for (int i = 0; i < p_col_count; ++i) {
    if (unlikely(i == p_k1 || i == p_k2)) continue;
    write_char(','); write_out(p_cols[i].data, p_cols[i].len);
  }
  for (int i = 0; i < q_row->col_count; ++i) {
    if (unlikely(i == q_k1 || i == q_k2)) continue;
    write_char(','); write_out(q_row->cols[i].data, q_row->cols[i].len);
  }
  write_char('\n');
}

__attribute__((target("avx2,sse4.2,popcnt"))) 
static inline void execute_join(IoBackend *io, ParseBackend *pb, const char *p_file, const char *q_file) {
  out_buf = malloc(OUT_BUF_SIZE);
  out_pos = 0;

  size_t p_size;
  char *p_buf = io->load_build(p_file, &p_size);
  const char *p_ptr = p_buf;
  const char *p_end = p_buf + p_size;

  CsvRow header_p;
  pb->csv_next(&p_ptr, p_end, &header_p);

  size_t exact_p_rows = pb->count_rows(p_ptr, p_end - p_ptr);
  
  Partition *prtns = malloc(NUM_PRTNS * sizeof(Partition));
  int p_cols_per_row = header_p.col_count;
  StringView *p_cell_arena = malloc(exact_p_rows * p_cols_per_row * sizeof(StringView));
  
  size_t cap_per_part = 1;
  size_t target = (exact_p_rows / NUM_PRTNS) * 4 + 1024;
  while (cap_per_part < target) cap_per_part <<= 1;

  for (int i = 0; i < NUM_PRTNS; ++i) {
    prtns[i].capacity = cap_per_part;
    prtns[i].mask = cap_per_part - 1;
    prtns[i].entries = calloc(cap_per_part, sizeof(HashEntry));
    prtns[i].count = 0;
  }
  size_t p_pool_size = 0;

  int p_keys[2] = {-1, -1};
  int q_keys[2] = {-1, -1};
  int keys_found = 0;
  int p_c1 = -1, p_c2 = -1, q_c1 = -1, q_c2 = -1;

  ProbeStream qs;
  io->stream_probe_init(q_file, &qs);

  char *buf; size_t len;
  bool header_matched = false;
  CsvRow header_q;

  while (io->stream_probe_next(&qs, &buf, &len)) {
    const char *q_ptr = buf;
    const char *q_end = buf + len;
    
    if (!header_matched) {
      if (pb->csv_next(&q_ptr, q_end, &header_q)) {
        for (int i = 0; i < header_p.col_count && keys_found < 2; ++i) {
          for (int j = 0; j < header_q.col_count && keys_found < 2; ++j) {
            if (view_eq(header_p.cols[i], header_q.cols[j])) {
              p_keys[keys_found] = i; q_keys[keys_found] = j; keys_found++;
            }
          }
        }
        if (keys_found != 2) {
          fprintf(stderr, "FATAL: Need exactly 2 common attributes.\n");
          exit(1);
        }
        p_c1 = p_keys[0]; p_c2 = p_keys[1]; q_c1 = q_keys[0]; q_c2 = q_keys[1];
        print_joined_row(header_p.cols, header_p.col_count, &header_q, p_c1, p_c2, q_c1, q_c2);
        
        // Build Hash Map for P
        CsvRow row;
        while (pb->csv_next(&p_ptr, p_end, &row)) {
          if (unlikely(row.col_count != header_p.col_count)) continue;
          uint64_t key = combine_hashes(hash_view(row.cols[p_c1]), hash_view(row.cols[p_c2]));
          
          uint32_t idx = p_pool_size++;
          uint32_t arena_ofst = idx * p_cols_per_row;
          memcpy(&p_cell_arena[arena_ofst], row.cols, p_cols_per_row * sizeof(StringView));

          uint32_t part_idx = key >> 56;
          Partition *p = &prtns[part_idx];

          if (unlikely(p->count + 1 > p->capacity * 7 / 10)) {
            size_t new_cap = p->capacity * 2;
            HashEntry *ne = calloc(new_cap, sizeof(HashEntry));
            size_t new_mask = new_cap - 1;
            for (size_t i = 0; i < p->capacity; i++) {
              if (!p->entries[i].occupied) continue;
              size_t slot = p->entries[i].key & new_mask;
              while (ne[slot].occupied) slot = (slot + 1) & new_mask;
              ne[slot] = p->entries[i];
            }
            free(p->entries);
            p->entries = ne; p->capacity = new_cap; p->mask = new_mask;
          }

          size_t slot = key & p->mask;
          while (unlikely(p->entries[slot].occupied)) slot = (slot + 1) & p->mask;

          p->entries[slot].key = key;
          p->entries[slot].row_idx = idx;
          p->entries[slot].occupied = true;
          p->count++;
        }
        header_matched = true;
      }
    }
    
    CsvRow row;
    while (pb->csv_next(&q_ptr, q_end, &row)) {
      if (unlikely(row.col_count != header_q.col_count)) continue;
      StringView qc1_v = row.cols[q_c1], qc2_v = row.cols[q_c2];
      uint64_t key = combine_hashes(hash_view(qc1_v), hash_view(qc2_v));
      uint32_t part_idx = key >> 56;
      Partition *p = &prtns[part_idx];
      size_t slot = key & p->mask;
      while (likely(p->entries[slot].occupied)) {
        if (likely(p->entries[slot].key == key)) {
          StringView *matched = &p_cell_arena[p->entries[slot].row_idx * p_cols_per_row];
          if (view_eq(matched[p_c1], qc1_v) && view_eq(matched[p_c2], qc2_v)) {
            print_joined_row(matched, p_cols_per_row, &row, p_c1, p_c2, q_c1, q_c2);
          }
        }
        slot = (slot + 1) & p->mask;
      }
    }
  }

  flush_out();

  for (int i = 0; i < NUM_PRTNS; ++i) free(prtns[i].entries);
  free(prtns);
  free(p_cell_arena);
  io->stream_probe_close(&qs);
  free(out_buf);
}

#endif
