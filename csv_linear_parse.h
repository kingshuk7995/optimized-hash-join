#ifndef CSV_LINEAR_PARSE_H
#define CSV_LINEAR_PARSE_H

#include "types.h"

static inline size_t count_rows_linear(const char *p, size_t n) {
  size_t count = 0;
  for (size_t i = 0; i < n; i++) {
    if (p[i] == '\n') count++;
  }
  return count;
}

static inline bool csv_next_linear(const char **current, const char *end, CsvRow *row) {
  const char *p = *current;
  if (unlikely(p >= end)) return false;

  row->col_count = 0;
  bool in_quotes = false;
  const char *field_start = p;

  while (p < end) {
    char c = *p;
    if (c == '"') {
      in_quotes = !in_quotes;
    } else if (c == ',' && !in_quotes) {
      if (row->col_count < MAX_COLS) {
        row->cols[row->col_count].data = field_start;
        row->cols[row->col_count].len = p - field_start;
        row->col_count++;
      }
      field_start = p + 1;
    } else if (c == '\n' && !in_quotes) {
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

static ParseBackend parse_backend_linear = { count_rows_linear, csv_next_linear };

#endif
