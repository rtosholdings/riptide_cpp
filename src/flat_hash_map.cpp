#include "CommonInc.h"
#include "RipTide.h"
#include "MathWorker.h"
#include "Recycler.h"
#include "flat_hash_map.h"

#include "absl/container/flat_hash_map.h"

template struct fhm_hasher<uint64_t, int8_t>;
template struct fhm_hasher<int64_t, int8_t>;
template struct fhm_hasher<uint32_t, int8_t>;
template struct fhm_hasher<int32_t, int8_t>;
template struct fhm_hasher<uint16_t, int8_t>;
template struct fhm_hasher<int16_t, int8_t>;
template struct fhm_hasher<uint8_t, int8_t>;
template struct fhm_hasher<int8_t, int8_t>;
template struct fhm_hasher<float, int8_t>;
template struct fhm_hasher<double, int8_t>;
template struct fhm_hasher<uint64_t, int16_t>;
template struct fhm_hasher<int64_t, int16_t>;
template struct fhm_hasher<uint32_t, int16_t>;
template struct fhm_hasher<int32_t, int16_t>;
template struct fhm_hasher<uint16_t, int16_t>;
template struct fhm_hasher<int16_t, int16_t>;
template struct fhm_hasher<uint8_t, int16_t>;
template struct fhm_hasher<int8_t, int16_t>;
template struct fhm_hasher<float, int16_t>;
template struct fhm_hasher<double, int16_t>;
template struct fhm_hasher<uint64_t, int32_t>;
template struct fhm_hasher<int64_t, int32_t>;
template struct fhm_hasher<uint32_t, int32_t>;
template struct fhm_hasher<int32_t, int32_t>;
template struct fhm_hasher<uint16_t, int32_t>;
template struct fhm_hasher<int16_t, int32_t>;
template struct fhm_hasher<uint8_t, int32_t>;
template struct fhm_hasher<int8_t, int32_t>;
template struct fhm_hasher<float, int32_t>;
template struct fhm_hasher<double, int32_t>;
template struct fhm_hasher<uint64_t, int64_t>;
template struct fhm_hasher<int64_t, int64_t>;
template struct fhm_hasher<uint32_t, int64_t>;
template struct fhm_hasher<int32_t, int64_t>;
template struct fhm_hasher<uint16_t, int64_t>;
template struct fhm_hasher<int16_t, int64_t>;
template struct fhm_hasher<uint8_t, int64_t>;
template struct fhm_hasher<int8_t, int64_t>;
template struct fhm_hasher<float, int64_t>;
template struct fhm_hasher<double, int64_t>;
