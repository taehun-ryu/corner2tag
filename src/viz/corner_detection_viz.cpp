#include "corner2tag/viz/corner_detection_viz.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "corner2tag/viz/event_render.hpp"

namespace corner2tag::viz {

static cv::Mat normalizeToU8(const cv::Mat &m) {
  CV_Assert(m.type() == CV_32F);
  cv::Mat norm;
  cv::normalize(m, norm, 0, 255, cv::NORM_MINMAX);
  cv::Mat u8;
  norm.convertTo(u8, CV_8U);
  return u8;
}

static cv::Mat toBgr(const cv::Mat &m) {
  if (m.empty()) {
    return cv::Mat();
  }
  cv::Mat out;
  if (m.channels() == 1) {
    cv::cvtColor(m, out, cv::COLOR_GRAY2BGR);
  } else {
    out = m.clone();
  }
  return out;
}

static int scaledSize(int base, float vis_zoom) {
  return std::max(1, static_cast<int>(std::lround(base * vis_zoom)));
}

static int scaledRadius(float base, float vis_zoom) {
  return std::max(1, static_cast<int>(std::lround(base * vis_zoom)));
}

static int gridIndex(int r, int c, int cols) { return r * cols + c; }

static std::vector<cv::Point2f>
reshapeGrid(const std::vector<cv::Point2f> &ordered, int rows, int cols) {
  std::vector<cv::Point2f> out = ordered;
  if (rows <= 0 || cols <= 0 || out.empty()) {
    return out;
  }
  if (out[0].x > out[cols - 1].x) {
    for (int r = 0; r < rows; ++r) {
      std::reverse(out.begin() + r * cols, out.begin() + (r + 1) * cols);
    }
  }
  if (out[0].y > out[(rows - 1) * cols].y) {
    for (int r = 0; r < rows / 2; ++r) {
      for (int c = 0; c < cols; ++c) {
        std::swap(out[r * cols + c], out[(rows - 1 - r) * cols + c]);
      }
    }
  }
  return out;
}

static cv::Mat drawCheckerboardRowSnake(const cv::Mat &gray_or_bgr,
                                        const std::vector<cv::Point2f> &ordered,
                                        int rows, int cols, int radius,
                                        bool draw_points, int thickness) {
  cv::Mat vis;
  if (gray_or_bgr.channels() == 1) {
    cv::cvtColor(gray_or_bgr, vis, cv::COLOR_GRAY2BGR);
  } else {
    vis = gray_or_bgr.clone();
  }

  const int line_thickness = std::max(1, thickness);
  const int point_radius = std::max(1, radius);

  const cv::Scalar colors[] = {cv::Scalar(0, 0, 255),   cv::Scalar(0, 165, 255),
                               cv::Scalar(0, 255, 255), cv::Scalar(0, 255, 0),
                               cv::Scalar(255, 0, 0),   cv::Scalar(130, 0, 75),
                               cv::Scalar(211, 0, 148)};

  const std::vector<cv::Point2f> grid = reshapeGrid(ordered, rows, cols);

  for (int r = 0; r < rows; ++r) {
    const cv::Scalar color = colors[r % (sizeof(colors) / sizeof(colors[0]))];
    std::vector<cv::Point> row_pts;
    row_pts.reserve(static_cast<size_t>(cols));
    for (int c = 0; c < cols; ++c) {
      const cv::Point2f p = grid[gridIndex(r, c, cols)];
      row_pts.emplace_back(static_cast<int>(std::round(p.x)),
                           static_cast<int>(std::round(p.y)));
    }

    cv::polylines(vis, row_pts, false, color, line_thickness, cv::LINE_AA);
    if (draw_points) {
      for (const auto &p : row_pts) {
        cv::circle(vis, p, point_radius, color, -1, cv::LINE_AA);
        cv::circle(vis, p, point_radius, cv::Scalar(0, 0, 0), line_thickness,
                   cv::LINE_AA);
      }
    }

    if (r < rows - 1) {
      const cv::Point p_end = row_pts.back();
      const cv::Point p_next(
          static_cast<int>(std::round(grid[gridIndex(r + 1, 0, cols)].x)),
          static_cast<int>(std::round(grid[gridIndex(r + 1, 0, cols)].y)));
      cv::line(vis, p_end, p_next, color, line_thickness, cv::LINE_AA);
    }
  }

  return vis;
}

static void putDebugText(cv::Mat &img, const std::string &txt, int y) {
  cv::putText(img, txt, cv::Point(11, y + 1), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
  cv::putText(img, txt, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

cv::Mat buildRawEventsWindowVis(
    const std::vector<corner2tag::core::TimedEventNs> &events, int width,
    int height, float vx, float vy, double window_dt_s, float zoom_factor) {
  const float vis_zoom = (zoom_factor > 0.0f) ? zoom_factor : 1.0f;
  cv::Mat raw_vis =
      corner2tag::viz::eventsToImage(events, width, height, vis_zoom, true, 50);
  if (!raw_vis.empty()) {
    const cv::Point center(raw_vis.cols / 2, raw_vis.rows / 2);
    const double dx = static_cast<double>(vx) * window_dt_s * vis_zoom;
    const double dy = static_cast<double>(vy) * window_dt_s * vis_zoom;
    const double norm = std::sqrt(dx * dx + dy * dy);
    if (std::isfinite(norm) && norm > 1e-6) {
      const cv::Point tip(static_cast<int>(std::round(center.x + dx)),
                          static_cast<int>(std::round(center.y + dy)));
      cv::arrowedLine(raw_vis, center, tip, cv::Scalar(0, 255, 0), 2,
                      cv::LINE_AA, 0, 0.25);
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << "v=(" << vx << "," << vy
          << ") px/s";
      const std::string label = oss.str();
      cv::putText(raw_vis, label, cv::Point(10, 20),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1,
                  cv::LINE_AA);
    }
  }
  return raw_vis;
}

static cv::Mat buildScaledIweBase(const cv::Mat &iwe_u8, float vis_zoom) {
  if (iwe_u8.empty()) {
    return cv::Mat();
  }
  cv::Mat iwe_vis;
  cv::resize(iwe_u8, iwe_vis,
             cv::Size(scaledSize(iwe_u8.cols, vis_zoom),
                      scaledSize(iwe_u8.rows, vis_zoom)),
             0, 0,
             cv::INTER_NEAREST);
  return toBgr(iwe_vis);
}

static cv::Mat buildCheckerboardIweVis(
    const cv::Mat &iwe_u8, float vis_zoom,
    const std::vector<cv::Point2f> &init_corners,
    const std::vector<cv::Point2f> &refined_corners,
    const std::vector<cv::Point2f> &filter_corners,
    const std::vector<cv::Point2f> &ordered_corners, bool order_ok,
    bool validity_ok) {
  cv::Mat iwe_vis = buildScaledIweBase(iwe_u8, vis_zoom);
  if (iwe_vis.empty()) {
    return iwe_vis;
  }

  const int corner_outline = std::max(1, static_cast<int>(std::lround(vis_zoom)));
  const int fp_radius = scaledRadius(3.0f, vis_zoom);
  const int tp_radius = scaledRadius(4.0f, vis_zoom);

  const std::vector<cv::Point2f> &filter_input_corners =
      refined_corners.empty() ? init_corners : refined_corners;
  const bool filtering_ok = !filter_corners.empty();

  // Base: all refined candidates are shown in red.
  for (const auto &c : filter_input_corners) {
    const cv::Point cc(static_cast<int>(std::lround(c.x * vis_zoom)),
                       static_cast<int>(std::lround(c.y * vis_zoom)));
    cv::circle(iwe_vis, cc, fp_radius, cv::Scalar(0, 0, 255), -1,
               cv::LINE_AA);
  }

  const std::vector<cv::Point2f> *stage_points = nullptr;
  cv::Scalar stage_color;
  if (filtering_ok && order_ok && !validity_ok) {
    stage_points = &ordered_corners;       // filtering+ordering pass, validity no
    stage_color = cv::Scalar(255, 0, 255); // pink
  } else if (filtering_ok && order_ok && validity_ok) {
    stage_points = &ordered_corners;       // filtering+ordering+validity pass
    stage_color = cv::Scalar(255, 255, 0); // cyan
  }

  if (stage_points != nullptr) {
    for (const auto &c : *stage_points) {
      const cv::Point cc(static_cast<int>(std::lround(c.x * vis_zoom)),
                         static_cast<int>(std::lround(c.y * vis_zoom)));
      cv::circle(iwe_vis, cc, tp_radius, stage_color, -1, cv::LINE_AA);
      cv::circle(iwe_vis, cc, tp_radius, cv::Scalar(0, 0, 0),
                 corner_outline, cv::LINE_AA);
    }
  }

  putDebugText(iwe_vis,
               "cand=" + std::to_string(filter_input_corners.size()) +
                   " filtered=" + std::to_string(filter_corners.size()) +
                   " validity=" + std::string(validity_ok ? "ok" : "no"),
               20);
  return iwe_vis;
}

static cv::Mat buildCheckerboardDetectionVis(
    const cv::Mat &iwe_u8, float vis_zoom,
    const std::vector<cv::Point2f> &ordered_corners, bool order_ok,
    bool validity_ok, int board_rows, int board_cols) {
  if (!(order_ok && validity_ok && !ordered_corners.empty() && board_rows > 0 &&
        board_cols > 0)) {
    return cv::Mat();
  }
  cv::Mat detection_base;
  cv::resize(iwe_u8, detection_base,
             cv::Size(scaledSize(iwe_u8.cols, vis_zoom),
                      scaledSize(iwe_u8.rows, vis_zoom)),
             0, 0, cv::INTER_NEAREST);
  cv::Mat detection_vis = toBgr(detection_base);
  const int corner_radius = scaledRadius(3.0f, vis_zoom);
  const int line_thickness =
      std::max(1, static_cast<int>(std::lround(vis_zoom)));
  std::vector<cv::Point2f> scaled;
  scaled.reserve(ordered_corners.size());
  for (const auto &p : ordered_corners) {
    scaled.emplace_back(p.x * vis_zoom, p.y * vis_zoom);
  }
  return drawCheckerboardRowSnake(detection_vis, scaled, board_rows, board_cols,
                                  corner_radius, true, line_thickness);
}

WindowVis buildCheckerboardWindowVis(
    const std::vector<corner2tag::core::TimedEventNs> &events, int width,
    int height, const cv::Mat &iwe,
    const std::vector<cv::Point2f> &init_corners,
    const std::vector<cv::Point2f> &refined_corners,
    const std::vector<cv::Point2f> &filter_corners,
    const std::vector<cv::Point2f> &ordered_corners, bool order_ok,
    bool validity_ok, float vx, float vy, double window_dt_s, int board_rows,
    int board_cols, float zoom_factor) {
  WindowVis out;
  const float vis_zoom = (zoom_factor > 0.0f) ? zoom_factor : 1.0f;
  out.raw_vis =
      buildRawEventsWindowVis(events, width, height, vx, vy, window_dt_s, vis_zoom);
  if (iwe.empty()) {
    return out;
  }
  const cv::Mat iwe_u8 = normalizeToU8(iwe);
  out.iwe_vis = buildCheckerboardIweVis(iwe_u8, vis_zoom, init_corners,
                                        refined_corners, filter_corners,
                                        ordered_corners, order_ok, validity_ok);
  out.detection_vis = buildCheckerboardDetectionVis(
      iwe_u8, vis_zoom, ordered_corners, order_ok, validity_ok, board_rows,
      board_cols);
  return out;
}

void showWindowVis(const WindowVis &vis,
                   const std::string &detection_window_name) {
  if (!vis.raw_vis.empty()) {
    cv::imshow("raw_events", vis.raw_vis);
  }
  if (!vis.iwe_vis.empty()) {
    cv::imshow("IWE", vis.iwe_vis);
  }
  if (!vis.detection_vis.empty()) {
    cv::imshow(detection_window_name, vis.detection_vis);
  }
}

void showCheckerboardWindowVis(const WindowVis &vis) {
  showWindowVis(vis, "corner_detection");
}

} // namespace corner2tag::viz
