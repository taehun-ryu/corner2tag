#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

namespace corner2tag::core {

struct CornerInitOptions {
  // Perimeter seed peak threshold (on raw IWE values)
  float seed_thr = 5.0f;
  // If > 0, use percentile of perimeter IWE values as seed threshold
  float seed_percentile = 0.0f;
  // Propagation threshold (neighbors must satisfy >= tau)
  float tau = 1.0f;
  // If > 0, use percentile of all IWE values as propagation threshold
  float tau_percentile = 0.0f;
  // Max propagation iterations (depth)
  int max_grow_iters = 10;
  // Reject tiny components (noise)
  int min_component_area = 4;
  // If true, use weights = |IWE| in line fitting (recommended)
  bool weighted_line_fit = true;
};

struct Line2D {
  cv::Point2f c;  // centroid
  cv::Point2f v;  // unit direction (max variance)
  cv::Point2f n;  // unit normal
  float d = 0.0f; // n.x * x + n.y * y + d = 0

  float rms = 0.0f; // RMS distance of points to line
};

struct CornerInitResult {
  bool success = false;

  // initialization intersection
  cv::Point2f init_corner_xy = cv::Point2f(0.0f, 0.0f);
  // final (same as init in this stage)
  cv::Point2f corner_xy = cv::Point2f(0.0f, 0.0f);
  Line2D line0;
  Line2D line1;

  // Debug
  float chosen_tau = 0.0f;
  int num_components = 0;

  // CV_8U binary
  cv::Mat mask_u8;
  // CV_8U seed mask
  cv::Mat seeds_u8;
  // CV_32S depth map (-1 for background)
  cv::Mat depth_i32;

  // CV_32S label map (background=-1, otherwise 0..3 after remap)
  cv::Mat labels_i32;

  // BGR visualization for quick debug (CV_8UC3)
  cv::Mat labels_bgr_u8;
};

CornerInitResult initializeCornerFromIwePatch(const cv::Mat &iwe_f32,
                                              const CornerInitOptions &opt,
                                              float seed_thr_override = -1.0f,
                                              float tau_thr_override = -1.0f);

// Compute global thresholds from full IWE (percentiles if enabled).
// Ensures tau_thr < seed_thr when both are set.
void computeGlobalIweThresholds(const cv::Mat &iwe_f32,
                                const CornerInitOptions &opt,
                                float *seed_thr_out, float *tau_thr_out);

// Helper: convert a label map (0..3, bg=-1) into BGR image
cv::Mat colorizeLabels4(const cv::Mat &labels_i32);

} // namespace corner2tag::core
