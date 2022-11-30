#include "is_member_tg.h"

DllExport hash_choice_t runtime_hash_choice{ hash_choice_t::hash_linear };

static_assert(std::is_trivial_v<typename riptide_cpp::simple_span<char>>, "simple_span must be trivial to work with TBB");
static_assert(std::is_trivial_v<typename riptide_cpp::simple_span<wchar_t>>, "simple_span must be trivial to work with TBB");

static_assert(std::is_aggregate_v<typename riptide_cpp::simple_span<char>>,
              "simple_span must be capable of aggregate initialization");
static_assert(std::is_aggregate_v<typename riptide_cpp::simple_span<wchar_t>>,
              "simple_span must be capable of aggregate initialization");
