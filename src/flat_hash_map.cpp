#include "CommonInc.h"
#include "RipTide.h"
#include "MathWorker.h"
#include "Recycler.h"
#include "flat_hash_map.h"

#include "absl/container/flat_hash_map.h"

DllExport hash_choice_t runtime_hash_choice{ hash_choice_t::hash_linear };

// template struct fhm_hasher<uint64_t, int8_t>;
// template struct fhm_hasher<int64_t, int8_t>;
// template struct fhm_hasher<uint32_t, int8_t>;
// template struct fhm_hasher<int32_t, int8_t>;
// template struct fhm_hasher<uint16_t, int8_t>;
// template struct fhm_hasher<int16_t, int8_t>;
// template struct fhm_hasher<uint8_t, int8_t>;
// template struct fhm_hasher<int8_t, int8_t>;
// template struct fhm_hasher<float, int8_t>;
// template struct fhm_hasher<double, int8_t>;
template struct fhm_hasher<char, char>;
template struct fhm_hasher<unsigned char, char>;
template struct fhm_hasher<uint64_t, char>;
template struct fhm_hasher<int64_t, char>;
template struct fhm_hasher<uint32_t, char>;
template struct fhm_hasher<int32_t, char>;
template struct fhm_hasher<uint16_t, char>;
template struct fhm_hasher<int16_t, char>;
// template struct fhm_hasher<uint8_t, char>;
// template struct fhm_hasher<int8_t, char>;
template struct fhm_hasher<float, char>;
template struct fhm_hasher<double, char>;
template struct fhm_hasher<uint64_t, unsigned char>;
template struct fhm_hasher<int64_t, unsigned char>;
template struct fhm_hasher<uint32_t, unsigned char>;
template struct fhm_hasher<int32_t, unsigned char>;
template struct fhm_hasher<uint16_t, unsigned char>;
template struct fhm_hasher<int16_t, unsigned char>;
// template struct fhm_hasher<uint8_t, unsigned char>;
// template struct fhm_hasher<int8_t, unsigned char>;
template struct fhm_hasher<float, unsigned char>;
template struct fhm_hasher<double, unsigned char>;
template struct fhm_hasher<char, unsigned char>;
template struct fhm_hasher<unsigned char, unsigned char>;
template struct fhm_hasher<uint64_t, int16_t>;
template struct fhm_hasher<int64_t, int16_t>;
template struct fhm_hasher<uint32_t, int16_t>;
template struct fhm_hasher<int32_t, int16_t>;
template struct fhm_hasher<uint16_t, int16_t>;
template struct fhm_hasher<int16_t, int16_t>;
// template struct fhm_hasher<uint8_t, int16_t>;
// template struct fhm_hasher<int8_t, int16_t>;
template struct fhm_hasher<float, int16_t>;
template struct fhm_hasher<double, int16_t>;
template struct fhm_hasher<char, int16_t>;
template struct fhm_hasher<unsigned char, int16_t>;
template struct fhm_hasher<uint64_t, int32_t>;
template struct fhm_hasher<int64_t, int32_t>;
template struct fhm_hasher<uint32_t, int32_t>;
template struct fhm_hasher<int32_t, int32_t>;
template struct fhm_hasher<uint16_t, int32_t>;
template struct fhm_hasher<int16_t, int32_t>;
// template struct fhm_hasher<uint8_t, int32_t>;
// template struct fhm_hasher<int8_t, int32_t>;
template struct fhm_hasher<float, int32_t>;
template struct fhm_hasher<double, int32_t>;
template struct fhm_hasher<char, int32_t>;
template struct fhm_hasher<unsigned char, int32_t>;
template struct fhm_hasher<uint64_t, int64_t>;
template struct fhm_hasher<int64_t, int64_t>;
template struct fhm_hasher<uint32_t, int64_t>;
template struct fhm_hasher<int32_t, int64_t>;
template struct fhm_hasher<uint16_t, int64_t>;
template struct fhm_hasher<int16_t, int64_t>;
// template struct fhm_hasher<uint8_t, int64_t>;
// template struct fhm_hasher<int8_t, int64_t>;
template struct fhm_hasher<float, int64_t>;
template struct fhm_hasher<double, int64_t>;
template struct fhm_hasher<char, int64_t>;
template struct fhm_hasher<unsigned char, int64_t>;
