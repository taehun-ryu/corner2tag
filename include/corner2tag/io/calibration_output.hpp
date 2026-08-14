#pragma once

#include <cstddef>
#include <string>

#include <opencv2/core.hpp>

#include "corner2tag/io/detection_output.hpp"

namespace corner2tag::io {

std::string
prepareCalibrationRunOutputDir(const std::string &base_out_dir,
                               const DetectionOutputLayout &layout =
                                   kCheckerboardDetectionOutputLayout);

void saveCheckerboardCalibrationYaml(const std::string &out_dir,
                                     size_t used_windows, size_t total_windows,
                                     int board_rows, int board_cols,
                                     float square_size,
                                     const cv::Mat &camera_matrix,
                                     const cv::Mat &dist_coeffs,
                                     double reprojection_error);

} // namespace corner2tag::io
