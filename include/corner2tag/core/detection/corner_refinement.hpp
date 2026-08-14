#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "corner2tag/core/detection/corner_init.hpp"

namespace corner2tag::core {

struct CornerRefineOptions {
  bool enable = true;
  float lr = 0.25f;
  int max_iter = 200;
  float gtol = 1e-6f;
  float armijo_c = 1e-4f;
  float min_step = 1e-6f;
  // half-width of central strips (pixels)
  float strip_half_width0 = 0.5f;
  float strip_half_width1 = 0.5f;
  bool keep_path = false;
};

struct CornerRefineResult {
  bool success = false;
  cv::Point2f refined_xy = cv::Point2f(0.0f, 0.0f);
  float f = 0.0f;
  std::vector<cv::Point2f> path;
};

CornerRefineResult refineCornerInIwePatch(const cv::Mat &iwe_f32,
                                          const cv::Point2f &init_xy,
                                          const Line2D &line0,
                                          const Line2D &line1,
                                          const CornerRefineOptions &opt);

} // namespace corner2tag::core
