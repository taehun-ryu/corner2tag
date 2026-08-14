#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace corner2tag::core {

struct FixedWindowRangeUs {
  uint64_t index = 0;
  uint64_t begin_us = 0;
  uint64_t end_us = 0;
};

inline FixedWindowRangeUs
fixedWindowForRelativeTimestampUs(uint64_t relative_timestamp_us,
                                  uint64_t window_us) {
  if (window_us == 0) {
    throw std::invalid_argument("fixed_window_us must be positive");
  }

  FixedWindowRangeUs range;
  range.index = relative_timestamp_us / window_us;
  range.begin_us = range.index * window_us;
  if (range.begin_us >
      std::numeric_limits<uint64_t>::max() - window_us) {
    throw std::overflow_error("fixed window timestamp overflow");
  }
  range.end_us = range.begin_us + window_us;
  return range;
}

} // namespace corner2tag::core
