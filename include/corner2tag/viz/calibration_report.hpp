#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace corner2tag::viz {

void saveCalibrationReportImages(
    const std::string &out_dir, const cv::Size &image_size, const cv::Mat &K,
    const cv::Mat &dist, const std::vector<cv::Mat> &rvecs,
    const std::vector<cv::Mat> &tvecs,
    const std::vector<std::vector<cv::Point3f>> &objpoints,
    const std::vector<std::vector<cv::Point2f>> &imgpoints);

} // namespace corner2tag::viz
