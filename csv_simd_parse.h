#ifndef CSV_SIMD_PARSE_H
#define CSV_SIMD_PARSE_H

#include "types.h"
#include <immintrin.h>

__attribute__((target("avx2"))) static inline size_t count_rows_simd(const char *p, size_t n) {
  size_t count = 0;
  size_t i = 0;
  __m256i nl = _mm256_set1_epi8('\n');
  for (; i + 31 < n; i += 32) {
    __m256i chunk = _mm256_loadu_si256((const __m256i *)(p + i));
    __m256i cmp = _mm256_cmpeq_epi8(chunk, nl);
    unsigned int mask = _mm256_movemask_epi8(cmp);
    count += __builtin_popcount(mask);
  }
  for (; i < n; i++) {
    if (p[i] == '\n') count++;
  }
  return count;
}

__attribute__((target("avx2,popcnt"))) static inline bool
csv_next_simd(const char **current, const char *end, CsvRow *row) {
  const char *p = *current;
  if (unlikely(p >= end)) return false;

  row->col_count = 0;
  const char *field_start = p;

  __m256i comma = _mm256_set1_epi8(',');
  __m256i newline = _mm256_set1_epi8('\n');
  __m256i cr = _mm256_set1_epi8('\r');

  while (p + 31 < end) {
    __m256i chunk = _mm256_loadu_si256((const __m256i *)p);
    __m256i cmp_comma = _mm256_cmpeq_epi8(chunk, comma);
    __m256i cmp_nl = _mm256_cmpeq_epi8(chunk, newline);
    __m256i cmp_cr = _mm256_cmpeq_epi8(chunk, cr);

    unsigned int mask_comma = _mm256_movemask_epi8(cmp_comma);
    unsigned int mask_nl = _mm256_movemask_epi8(cmp_nl);
    unsigned int mask_cr = _mm256_movemask_epi8(cmp_cr);

    unsigned int special = mask_comma | mask_nl;

    while (special) {
      int tz = __builtin_ctz(special);
      if (mask_comma & (1 << tz)) {
        if (row->col_count < MAX_COLS) {
          row->cols[row->col_count].data = field_start;
          row->cols[row->col_count].len = p + tz - field_start;
          row->col_count++;
        }
        field_start = p + tz + 1;
      } else {
        if (row->col_count < MAX_COLS) {
          row->cols[row->col_count].data = field_start;
          size_t len = p + tz - field_start;
          if (len > 0 && (mask_cr & (1 << (tz - 1)))) {
            len--;
          }
          row->cols[row->col_count].len = len;
          row->col_count++;
        }
        *current = p + tz + 1;
        return true;
      }
      special &= special - 1;
    }
    p += 32;
  }

  while (p < end) {
    char c = *p;
    if (c == ',') {
      if (row->col_count < MAX_COLS) {
        row->cols[row->col_count].data = field_start;
        row->cols[row->col_count].len = p - field_start;
        row->col_count++;
      }
      field_start = p + 1;
    } else if (c == '\n') {
      if (row->col_count < MAX_COLS) {
        row->cols[row->col_count].data = field_start;
        row->cols[row->col_count].len = p - field_start;
        if (row->cols[row->col_count].len > 0 && row->cols[row->col_count].data[row->cols[row->col_count].len - 1] == '\r') {
          row->cols[row->col_count].len--;
        }
        row->col_count++;
      }
      *current = p + 1;
      return true;
    }
    p++;
  }

  if (p > field_start) {
    if (row->col_count < MAX_COLS) {
      row->cols[row->col_count].data = field_start;
      row->cols[row->col_count].len = p - field_start;
      if (row->cols[row->col_count].len > 0 && row->cols[row->col_count].data[row->cols[row->col_count].len - 1] == '\r') {
        row->cols[row->col_count].len--;
      }
      row->col_count++;
    }
  }

  *current = end;
  return row->col_count > 0;
}

static ParseBackend parse_backend_simd = { count_rows_simd, csv_next_simd };

#endif
