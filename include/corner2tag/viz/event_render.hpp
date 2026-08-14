#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "corner2tag/core/simple_event.hpp"

namespace corner2tag::viz {

cv::Mat eventsToImage(const std::vector<corner2tag::core::TimedEventNs> &events,
                      int width, int height, float zoom_factor = 1.0f,
                      bool non_max_suppression = false, int add_value = 50);

} // namespace corner2tag::viz
