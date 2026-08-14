#include "corner2tag/core/cm/cm_tracker_2d.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ceres/ceres.h>
#include <opencv2/core.hpp>

#include "corner2tag/core/cm/cm_velocity_2d.hpp"
#include "corner2tag/core/cm/iwe_gaussian.hpp"
#include "corner2tag/core/simple_event.hpp"

namespace corner2tag::core {

CmTracker2D::CmTracker2D(int width, int height, int num_threads,
                         CmIweOptions iwe_opt)
    : width_(width), height_(height), num_threads_(num_threads),
      iwe_opt_(iwe_opt) {
  if (width_ <= 0 || height_ <= 0) {
    throw std::runtime_error("CmTracker2D: invalid image size");
  }
  if (num_threads_ <= 0) {
    num_threads_ = 1;
  }

  kernel_ = makeGaussianKernel2D(iwe_opt_.sigma, iwe_opt_.cutoff_factor);
  if (iwe_opt_.patch_radius_override >= 0) {
    const int r = iwe_opt_.patch_radius_override;
    const float forced_cutoff = static_cast<float>(r) / iwe_opt_.sigma;
    kernel_ = makeGaussianKernel2D(iwe_opt_.sigma, forced_cutoff);
  }
}

static inline uint8_t polTo01(bool p) {
  if (p) {
    return static_cast<uint8_t>(1);
  }
  return static_cast<uint8_t>(0);
}

void CmTracker2D::computeFinalIweAndCentroid(
    const std::vector<SimpleEvent> &events_frame, double vx, double vy,
    cv::Mat &iwe_out, cv::Mat &piwe_out, double &cx_out, double &cy_out) const {
  if (events_frame.empty()) {
    iwe_out = cv::Mat::zeros(height_, width_, CV_32F);
    piwe_out = cv::Mat::zeros(height_, width_, CV_32F);
    cx_out = 0.0;
    cy_out = 0.0;
    return;
  }

  const double t0 = events_frame.front().t;

  std::vector<float> xw;
  std::vector<float> yw;
  std::vector<uint8_t> pol01;

  xw.reserve(events_frame.size());
  yw.reserve(events_frame.size());
  pol01.reserve(events_frame.size());

  double sum_x = 0.0;
  double sum_y = 0.0;
  double cnt = 0.0;

  for (const auto &e : events_frame) {
    const double dt = e.t - t0;
    const double wx = static_cast<double>(e.x) - dt * vx;
    const double wy = static_cast<double>(e.y) - dt * vy;

    xw.push_back(static_cast<float>(wx));
    yw.push_back(static_cast<float>(wy));
    pol01.push_back(polTo01(e.polarity));

    // centroid: use in-range points only (avoid drifting by out-of-bounds)
    if (wx >= 0.0 && wy >= 0.0) {
      if (wx < static_cast<double>(width_) &&
          wy < static_cast<double>(height_)) {
        sum_x += wx;
        sum_y += wy;
        cnt += 1.0;
      }
    }
  }

  if (cnt > 0.0) {
    cx_out = sum_x / cnt;
    cy_out = sum_y / cnt;
  } else {
    cx_out = 0.0;
    cy_out = 0.0;
  }

  accumulateIweGaussian(xw, yw, pol01, width_, height_, kernel_, piwe_out,
                        iwe_out, num_threads_);
}

CmTracker2DResult
CmTracker2D::track(const std::vector<SimpleEvent> &events_frame,
                   const CmTracker2DOptions &opt, double v_init_x,
                   double v_init_y) const {
  CmTracker2DResult out;
  out.vx = v_init_x;
  out.vy = v_init_y;

  if (events_frame.empty()) {
    out.success = false;
    out.brief_report = "empty window";
    return out;
  }

  double v[2];
  v[0] = v_init_x;
  v[1] = v_init_y;

  ceres::Problem problem;
  ceres::NumericDiffOptions nd_opt;
  nd_opt.relative_step_size = 1e-3;
  nd_opt.ridders_relative_initial_step_size = 1e-2;

  auto *cost =
      new ceres::NumericDiffCostFunction<EventVelocity2D, ceres::CENTRAL, 1, 2>(
          new EventVelocity2D(events_frame, width_, height_, num_threads_,
                              iwe_opt_));

  problem.AddResidualBlock(cost, nullptr, v);

  ceres::Solver::Options ceres_opt;
  ceres_opt.minimizer_progress_to_stdout = opt.verbose;
  ceres_opt.max_num_iterations = opt.max_iterations;
  ceres_opt.num_threads = 1; // deterministic; our IWE uses threads

  ceres::Solver::Summary summary;
  ceres::Solve(ceres_opt, &problem, &summary);

  out.vx = v[0];
  out.vy = v[1];
  out.success = summary.IsSolutionUsable();
  out.brief_report = summary.BriefReport();

  if (opt.compute_final_iwe) {
    computeFinalIweAndCentroid(events_frame, out.vx, out.vy, out.iwe, out.piwe,
                               out.cx, out.cy);
  }

  return out;
}

} // namespace corner2tag::core
