#pragma once

#include <cstdint>

namespace corner2tag {
namespace core {

/**
 * @brief CM/IWE computation-friendly event (time in seconds, typically relative
 * time).
 */
struct SimpleEvent {
  uint16_t x = 0;
  uint16_t y = 0;
  bool polarity = false;
  double t = 0.0; // seconds (usually relative, e.g., t - t0)
};

/**
 * @brief ROS time domain event for offline slicing / multi-sensor alignment.
 *
 * Design:
 *  - t_ns is integer nanoseconds in a common time domain (typically ROS time).
 *  - Use this type for windowing against RGB timestamps (header.stamp) and for
 * dumping.
 */
struct TimedEventNs {
  uint16_t x = 0;
  uint16_t y = 0;
  bool polarity = false;
  int64_t t_ns = 0; // nanoseconds
};

} // namespace core
} // namespace corner2tag
