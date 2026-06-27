#ifndef IO_URING_IO_H
#define IO_URING_IO_H

#include "types.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static inline void *uring_load_build(const char *filename, size_t *out_size) {
  int fd = open(filename, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    fd = open(filename, O_RDONLY);
    if (fd < 0) { perror("open P"); exit(1); }
  }
  struct stat sb; fstat(fd, &sb); *out_size = sb.st_size;
  size_t alloc_size = (sb.st_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
  char *buf;
  if (posix_memalign((void **)&buf, ALIGNMENT, alloc_size)) { perror("memalign"); exit(1); }
  struct io_uring ring; io_uring_queue_init(QUEUE_DEPTH, &ring, 0);
  off_t offset = 0; int submissions = 0;
  while (offset < sb.st_size) {
    while (submissions < QUEUE_DEPTH && offset < sb.st_size) {
      size_t read_bytes = alloc_size - offset > CHUNK_SIZE ? CHUNK_SIZE : alloc_size - offset;
      struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
      io_uring_prep_read(sqe, fd, buf + offset, read_bytes, offset);
      io_uring_sqe_set_data(sqe, (void *)1);
      io_uring_submit(&ring);
      offset += read_bytes;
      submissions++;
    }
    struct io_uring_cqe *cqe; io_uring_wait_cqe(&ring, &cqe);
    io_uring_cqe_seen(&ring, cqe); submissions--;
  }
  while (submissions > 0) {
    struct io_uring_cqe *cqe; io_uring_wait_cqe(&ring, &cqe);
    io_uring_cqe_seen(&ring, cqe); submissions--;
  }
  io_uring_queue_exit(&ring); close(fd);
  return buf;
}

static inline void uring_stream_probe_init(const char *filename, ProbeStream *s) {
  s->fd = open(filename, O_RDONLY | O_DIRECT);
  if (s->fd < 0) { s->fd = open(filename, O_RDONLY); }
  if (s->fd < 0) { perror("open Q"); exit(1); }
  struct stat sb; fstat(s->fd, &sb); s->file_size = sb.st_size;
  io_uring_queue_init(QUEUE_DEPTH, &s->ring, 0);
  for (int i = 0; i < QUEUE_DEPTH; i++) posix_memalign((void **)&s->q_io_bufs[i], ALIGNMENT, CHUNK_SIZE);
  s->q_work_buf_size = CHUNK_SIZE * 2;
  s->q_work_buf = malloc(s->q_work_buf_size);
  s->submit_offset = 0;
  s->completed_offset = 0;
  s->pending_reads = 0;
  s->tail_len = 0;
  s->tail_ptr = NULL;

  if (s->submit_offset < (off_t)s->file_size) {
    size_t next_sz = (s->file_size - s->submit_offset < CHUNK_SIZE) ? ((s->file_size - s->submit_offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1)) : CHUNK_SIZE;
    struct io_uring_sqe *sqe = io_uring_get_sqe(&s->ring);
    io_uring_prep_read(sqe, s->fd, s->q_io_bufs[0], next_sz, s->submit_offset);
    io_uring_sqe_set_data(sqe, (void *)0);
    io_uring_submit(&s->ring);
    s->pending_reads++;
    s->submit_offset += next_sz;
  }
}

static inline bool uring_stream_probe_next(ProbeStream *s, char **buf, size_t *len) {
  if (s->pending_reads == 0 && s->tail_len == 0) return false;

  if (s->tail_len > 0 && s->tail_ptr != NULL) {
    memmove(s->q_work_buf, s->tail_ptr, s->tail_len);
    s->tail_ptr = NULL;
  }

  int bytes_read = 0;
  long buf_idx = -1;
  if (s->pending_reads > 0) {
    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&s->ring, &cqe);
    bytes_read = cqe->res;
    if (bytes_read < 0) bytes_read = 0;
    buf_idx = (long)io_uring_cqe_get_data(cqe);
    io_uring_cqe_seen(&s->ring, cqe);
    s->pending_reads--;

    if (bytes_read > 0) {
      if (s->submit_offset < (off_t)s->file_size) {
        long next_idx = (buf_idx + 1) % QUEUE_DEPTH;
        size_t next_sz = (s->file_size - s->submit_offset < CHUNK_SIZE) ? ((s->file_size - s->submit_offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1)) : CHUNK_SIZE;
        struct io_uring_sqe *sqe = io_uring_get_sqe(&s->ring);
        io_uring_prep_read(sqe, s->fd, s->q_io_bufs[next_idx], next_sz, s->submit_offset);
        io_uring_sqe_set_data(sqe, (void *)next_idx);
        io_uring_submit(&s->ring);
        s->pending_reads++;
        s->submit_offset += next_sz;
      }
      if (s->completed_offset + bytes_read > s->file_size) {
        bytes_read = s->file_size - s->completed_offset;
      }
      s->completed_offset += bytes_read;

      if (s->tail_len + bytes_read > s->q_work_buf_size) {
        s->q_work_buf_size = s->tail_len + bytes_read + CHUNK_SIZE;
        s->q_work_buf = realloc(s->q_work_buf, s->q_work_buf_size);
      }
      memcpy(s->q_work_buf + s->tail_len, s->q_io_bufs[buf_idx], bytes_read);
    }
  }

  const char *work_ptr = s->q_work_buf;
  const char *work_end = s->q_work_buf + s->tail_len + bytes_read;
  
  if (s->pending_reads == 0 && s->submit_offset >= (off_t)s->file_size) {
    *buf = (char *)work_ptr;
    *len = work_end - work_ptr;
    s->tail_len = 0;
    return *len > 0;
  }

  const char *last_newline = work_end - 1;
  while (last_newline >= work_ptr && *last_newline != '\n') last_newline--;
  
  size_t valid_len = last_newline >= work_ptr ? (size_t)(last_newline + 1 - work_ptr) : 0;
  
  *buf = (char *)work_ptr;
  *len = valid_len;
  
  s->tail_len = work_end - (work_ptr + valid_len);
  s->tail_ptr = (char *)(work_ptr + valid_len);
  
  return valid_len > 0 || s->tail_len > 0;
}

static inline void uring_stream_probe_close(ProbeStream *s) {
  for (int i = 0; i < QUEUE_DEPTH; i++) free(s->q_io_bufs[i]);
  free(s->q_work_buf);
  io_uring_queue_exit(&s->ring);
  close(s->fd);
}

static IoBackend io_backend_uring = { uring_load_build, uring_stream_probe_init, uring_stream_probe_next, uring_stream_probe_close };

#endif
