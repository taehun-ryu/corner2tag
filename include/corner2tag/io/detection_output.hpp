#pragma once

#include <cstddef>
#include <string>

#include <opencv2/core.hpp>

namespace corner2tag::io {

struct DetectionOutputLayout {
  std::string detection_dir;
  std::string detection_prefix;
};

inline const DetectionOutputLayout kCheckerboardDetectionOutputLayout{
    "corner_detection", "corners"};
inline const DetectionOutputLayout kAprilTagDetectionOutputLayout{
    "tag_detection", "tags"};

void ensureDetectionOutputDirs(const std::string &out_dir,
                               const DetectionOutputLayout &layout);

void saveDetectionOutputs(const std::string &out_dir, size_t window_idx,
                          const cv::Mat &raw_vis, const cv::Mat &iwe_vis,
                          const cv::Mat &detection_vis,
                          const DetectionOutputLayout &layout);

} // namespace corner2tag::io
