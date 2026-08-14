#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

namespace corner2tag::core {

struct GaussianKernel2D {
  int radius = 0;
  float sigma = 1.0f;
  float inv_2s2 = 0.5f; // 1/(2*sigma^2)
};

GaussianKernel2D makeGaussianKernel2D(float sigma, float cutoff_factor);

// Accumulate both IWE and pIWE.
// xw,yw: warped coords (float, pixel units)
// pol01: 0/1 polarity (uint8)
// width,height: image size
// num_threads: >0
void accumulateIweGaussian(const std::vector<float> &xw,
                           const std::vector<float> &yw,
                           const std::vector<uint8_t> &pol01, int width,
                           int height, const GaussianKernel2D &kernel,
                           cv::Mat &piwe_out, cv::Mat &iwe_out,
                           int num_threads);

struct ObjectiveStats {
  double l2 = 0.0;
  double variance = 0.0;
};

// Compute objective statistics for a CV_32F image.
// If use_variance=false: fills l2 (sqrt(mean(img^2)))
// If use_variance=true:  fills variance
ObjectiveStats computeObjectiveStats(const cv::Mat &img, bool use_variance);

} // namespace corner2tag::core
