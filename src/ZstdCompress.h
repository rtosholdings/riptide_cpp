#pragma once

#include <stdlib.h>

void ZstdSetChecksumming(bool enable);
size_t ZstdCompressData(void * dst, size_t dstCapacity, const void * src, size_t srcSize, int32_t compressionLevel);
