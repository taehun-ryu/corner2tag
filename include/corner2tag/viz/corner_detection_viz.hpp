#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "corner2tag/core/simple_event.hpp"

namespace corner2tag::viz {

struct WindowVis {
  cv::Mat raw_vis;
  cv::Mat iwe_vis;
  cv::Mat detection_vis;
};

cv::Mat buildRawEventsWindowVis(
    const std::vector<corner2tag::core::TimedEventNs> &events, int width,
    int height, float vx, float vy, double window_dt_s,
    float zoom_factor = 2.0f);

WindowVis buildCheckerboardWindowVis(
    const std::vector<corner2tag::core::TimedEventNs> &events, int width,
    int height, const cv::Mat &iwe,
    const std::vector<cv::Point2f> &init_corners,
    const std::vector<cv::Point2f> &refined_corners,
    const std::vector<cv::Point2f> &filter_corners,
    const std::vector<cv::Point2f> &ordered_corners, bool order_ok,
    bool validity_ok, float vx, float vy, double window_dt_s, int board_rows,
    int board_cols, float zoom_factor = 2.0f);

void showWindowVis(const WindowVis &vis,
                   const std::string &detection_window_name);

void showCheckerboardWindowVis(const WindowVis &vis);

} // namespace corner2tag::viz
