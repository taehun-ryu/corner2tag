#include "corner2tag/core/cm/cm_velocity_2d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>

#include "corner2tag/core/cm/iwe_gaussian.hpp"
#include "corner2tag/core/simple_event.hpp"

namespace corner2tag::core {

EventVelocity2D::EventVelocity2D(const std::vector<SimpleEvent> &events,
                                 int width, int height, int num_threads,
                                 CmIweOptions iwe_opt)
    : width_(width), height_(height), num_threads_(num_threads),
      iwe_opt_(iwe_opt) {
  if (width_ <= 0 || height_ <= 0) {
    throw std::runtime_error("EventVelocity2D: invalid image size");
  }
  if (events.empty()) {
    throw std::runtime_error("EventVelocity2D: empty events");
  }
  if (num_threads_ <= 0) {
    num_threads_ = 1;
  }

  // Kernel
  kernel_ = makeGaussianKernel2D(iwe_opt_.sigma, iwe_opt_.cutoff_factor);
  if (iwe_opt_.patch_radius_override >= 0) {
    const int r = iwe_opt_.patch_radius_override;
    const float forced_cutoff = static_cast<float>(r) / iwe_opt_.sigma;
    kernel_ = makeGaussianKernel2D(iwe_opt_.sigma, forced_cutoff);
  }

  selected_.reserve(events.size());

  for (const auto &e : events) {
    selected_.push_back(e);
  }

  if (selected_.empty()) {
    throw std::runtime_error("EventVelocity2D: selected events empty");
  }

  t0_ = selected_.front().t;
}

bool EventVelocity2D::operator()(const double *const v,
                                 double *const res) const {
  const double vx = v[0];
  const double vy = v[1];

  const size_t n = selected_.size();

  std::vector<float> xw;
  std::vector<float> yw;
  std::vector<uint8_t> pol01;

  xw.reserve(n);
  yw.reserve(n);
  pol01.reserve(n);

  const int r = kernel_.radius;

  for (const auto &e : selected_) {
    const double dt = e.t - t0_;
    const double wx = static_cast<double>(e.x) - dt * vx;
    const double wy = static_cast<double>(e.y) - dt * vy;

    if (wx < -r || wy < -r) {
      continue;
    }
    if (wx >= static_cast<double>(width_ + r)) {
      continue;
    }
    if (wy >= static_cast<double>(height_ + r)) {
      continue;
    }

    xw.push_back(static_cast<float>(wx));
    yw.push_back(static_cast<float>(wy));
    pol01.push_back(e.polarity ? static_cast<uint8_t>(1)
                               : static_cast<uint8_t>(0));
  }

  cv::Mat piwe;
  cv::Mat iwe;

  accumulateIweGaussian(xw, yw, pol01, width_, height_, kernel_, piwe, iwe,
                        num_threads_);

  // FIXED: objective is ALWAYS based on IWE
  const ObjectiveStats st = computeObjectiveStats(iwe, iwe_opt_.use_variance);
  const double objective = iwe_opt_.use_variance ? st.variance : st.l2;

  res[0] = -objective + 1000.0;

  return true;
}

} // namespace corner2tag::core
