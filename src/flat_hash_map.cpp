#include "CommonInc.h"
#include "RipTide.h"
#include "MathWorker.h"
#include "Recycler.h"
#include "flat_hash_map.h"

#include "absl/container/flat_hash_map.h"

template struct fhm_hasher<uint64_t>;
template struct fhm_hasher<int64_t>;
template struct fhm_hasher<uint32_t>;
template struct fhm_hasher<int32_t>;
template struct fhm_hasher<uint16_t>;
template struct fhm_hasher<int16_t>;
template struct fhm_hasher<uint8_t>;
template struct fhm_hasher<int8_t>;
template struct fhm_hasher<float>;
template struct fhm_hasher<double>;
