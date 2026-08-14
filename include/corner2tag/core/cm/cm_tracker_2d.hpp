#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "corner2tag/core/cm/cm_velocity_2d.hpp"
#include "corner2tag/core/simple_event.hpp"

namespace corner2tag::core {

struct CmTracker2DOptions {
  int max_iterations = 200;
  bool verbose = false;

  // If true, compute final IWE/pIWE at the solved velocity.
  bool compute_final_iwe = true;

  // Final IWE uses full events_frame (not the subsampled set used inside
  // Ceres).
  bool final_use_full_events = true;
};

struct CmTracker2DResult {
  double vx = 0.0;
  double vy = 0.0;

  double cx = 0.0;
  double cy = 0.0;

  bool success = false;
  std::string brief_report;

  cv::Mat iwe;  // CV_32F
  cv::Mat piwe; // CV_32F
};

class CmTracker2D {
public:
  CmTracker2D(int width, int height, int num_threads, CmIweOptions iwe_opt);

  CmTracker2DResult track(const std::vector<SimpleEvent> &events_frame,
                          const CmTracker2DOptions &opt, double v_init_x,
                          double v_init_y) const;

private:
  void computeFinalIweAndCentroid(const std::vector<SimpleEvent> &events_frame,
                                  double vx, double vy, cv::Mat &iwe_out,
                                  cv::Mat &piwe_out, double &cx_out,
                                  double &cy_out) const;

private:
  int width_ = 0;
  int height_ = 0;
  int num_threads_ = 1;
  CmIweOptions iwe_opt_;
  GaussianKernel2D kernel_;
};

} // namespace corner2tag::core
