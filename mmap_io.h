#ifndef MMAP_IO_H
#define MMAP_IO_H

#include "types.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

static inline void *mmap_load_build(const char *filename, size_t *out_size) {
  int fd = open(filename, O_RDONLY);
  if (fd < 0) { perror("open P"); exit(1); }
  struct stat sb; fstat(fd, &sb); *out_size = sb.st_size;
  void *ptr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  if (ptr == MAP_FAILED) { perror("mmap load"); exit(1); }
  madvise(ptr, sb.st_size, MADV_SEQUENTIAL);
  close(fd);
  return ptr;
}

static inline void mmap_stream_probe_init(const char *filename, ProbeStream *s) {
  s->fd = open(filename, O_RDONLY);
  if (s->fd < 0) { perror("open Q"); exit(1); }
  struct stat sb; fstat(s->fd, &sb); s->file_size = sb.st_size;
  s->mmap_ptr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, s->fd, 0);
  if (s->mmap_ptr == MAP_FAILED) { perror("mmap probe"); exit(1); }
  madvise(s->mmap_ptr, s->file_size, MADV_SEQUENTIAL);
  s->current_offset = 0;
}

static inline bool mmap_stream_probe_next(ProbeStream *s, char **buf, size_t *len) {
  if (s->current_offset >= (off_t)s->file_size) return false;
  size_t chunk = s->file_size - s->current_offset;
  if (chunk > CHUNK_SIZE) chunk = CHUNK_SIZE;
  const char *ptr = s->mmap_ptr + s->current_offset;
  if (s->current_offset + chunk < s->file_size) {
    const char *nl = ptr + chunk - 1;
    while (nl >= ptr && *nl != '\n') nl--;
    if (nl >= ptr) {
      chunk = nl - ptr + 1;
    } else {
      nl = ptr + chunk;
      while (s->current_offset + (nl - ptr) < s->file_size && *nl != '\n') nl++;
      if (s->current_offset + (nl - ptr) < s->file_size) chunk = nl - ptr + 1;
      else chunk = s->file_size - s->current_offset;
    }
  }
  *buf = (char *)ptr;
  *len = chunk;
  s->current_offset += chunk;
  return chunk > 0;
}

static inline void mmap_stream_probe_close(ProbeStream *s) {
  munmap(s->mmap_ptr, s->file_size);
  close(s->fd);
}

static IoBackend io_backend_mmap = { mmap_load_build, mmap_stream_probe_init, mmap_stream_probe_next, mmap_stream_probe_close };

#endif
