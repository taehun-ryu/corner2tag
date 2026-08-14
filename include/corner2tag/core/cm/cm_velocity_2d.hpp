#pragma once

#include <vector>

#include "corner2tag/core/cm/iwe_gaussian.hpp"
#include "corner2tag/core/simple_event.hpp"

namespace corner2tag::core {

struct CmIweOptions {
  float sigma = 1.0f;
  float cutoff_factor = 3.0f;

  // If >=0: radius override (radius = patch_radius_override)
  int patch_radius_override = -1;

  // objective on IWE only
  bool use_variance = false;
};

class EventVelocity2D {
public:
  EventVelocity2D(const std::vector<SimpleEvent> &events, int width, int height,
                  int num_threads, CmIweOptions iwe_opt);

  bool operator()(const double *const v, double *const res) const;

private:
  std::vector<SimpleEvent> selected_;

  int width_ = 0;
  int height_ = 0;
  int num_threads_ = 4;

  double t0_ = 0.0;

  CmIweOptions iwe_opt_;
  GaussianKernel2D kernel_;
};

} // namespace corner2tag::core
