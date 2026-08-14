#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace corner2tag::core {

struct CheckerboardFilterResult {
  bool success = false;
  std::vector<cv::Point2f> filtered; // selected corners (rows*cols)
  float score = 0.0f;                // lower is better
};

struct CheckerboardOrderResult {
  bool success = false;
  std::vector<cv::Point2f> ordered; // row-major (rows*cols)
  float score = 0.0f;               // lower is better
};

// Filter false positives using checkerboard grid prior.
// Input may contain extra points; output contains exactly rows*cols points.
CheckerboardFilterResult
filterCheckerboardCorners(const std::vector<cv::Point2f> &points, int rows,
                          int cols);

// Order corners into row-major layout.
// This function does not filter extras and expects exactly rows*cols points.
CheckerboardOrderResult
orderCheckerboardCorners(const std::vector<cv::Point2f> &points, int rows,
                         int cols);

bool isCheckerboardValid(const std::vector<cv::Point2f> &ordered, int rows,
                         int cols, float tor_spacing = 0.25f,
                         float tor_orth = 0.2f);

std::vector<cv::Point3f> buildObjectPoints(int rows, int cols,
                                           float square_size);

} // namespace corner2tag::core
