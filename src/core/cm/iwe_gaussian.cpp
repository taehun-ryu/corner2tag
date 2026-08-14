#include "corner2tag/core/cm/iwe_gaussian.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

namespace corner2tag::core {

GaussianKernel2D makeGaussianKernel2D(float sigma, float cutoff_factor) {
  if (sigma <= 0.0f) {
    throw std::runtime_error("sigma must be > 0");
  }
  if (cutoff_factor <= 0.0f) {
    throw std::runtime_error("cutoff_factor must be > 0");
  }
  GaussianKernel2D k;
  k.sigma = sigma;
  k.inv_2s2 = 1.0f / (2.0f * sigma * sigma);
  k.radius = static_cast<int>(std::ceil(cutoff_factor * sigma));
  return k;
}

static inline float polaritySign(uint8_t p01) {
  // FIXED mapping:
  // ON=1 -> +1
  // OFF=0 -> -1
  if (p01 != 0) {
    return 1.0f;
  }
  return -1.0f;
}

static void accumulateRange(const std::vector<float> &xw,
                            const std::vector<float> &yw,
                            const std::vector<uint8_t> &pol01, size_t i0,
                            size_t i1, int width, int height,
                            const GaussianKernel2D &kernel, cv::Mat &piwe_local,
                            cv::Mat &iwe_local) {
  const int r = kernel.radius;
  const float inv_2s2 = kernel.inv_2s2;

  for (size_t i = i0; i < i1; ++i) {
    const float x = xw[i];
    const float y = yw[i];

    const int cx = static_cast<int>(std::floor(x));
    const int cy = static_cast<int>(std::floor(y));

    // Early reject with footprint
    if (cx + r < 0 || cy + r < 0 || cx - r >= width || cy - r >= height) {
      continue;
    }

    const float sgn = (pol01[i] != 0) ? 1.0f : -1.0f;

    // Optional: precompute x distances for speed
    float dx2[128]; // radius small (<= 20-ish). adjust if needed.
    const int span = 2 * r + 1;
    if (span > 128) {
      throw std::runtime_error("kernel radius too large for stack buffer");
    }

    for (int dx = -r; dx <= r; ++dx) {
      const float xx = static_cast<float>(cx + dx);
      const float d = xx - x;
      dx2[dx + r] = d * d;
    }

    for (int dy = -r; dy <= r; ++dy) {
      const int yy = cy + dy;
      if (yy < 0 || yy >= height) {
        continue;
      }

      const float yyf = static_cast<float>(yy);
      const float dyf = yyf - y;
      const float dy2 = dyf * dyf;

      float *piwe_row = piwe_local.ptr<float>(yy);
      float *iwe_row = iwe_local.ptr<float>(yy);

      for (int dx = -r; dx <= r; ++dx) {
        const int xx = cx + dx;
        if (xx < 0 || xx >= width) {
          continue;
        }

        const float r2 = dx2[dx + r] + dy2;
        const float w = std::exp(-r2 * inv_2s2);

        iwe_row[xx] += w;
        piwe_row[xx] += sgn * w;
      }
    }
  }
}

static void accumulateRangeWorker(const std::vector<float> &xw,
                                  const std::vector<float> &yw,
                                  const std::vector<uint8_t> &pol01, size_t i0,
                                  size_t i1, int width, int height,
                                  const GaussianKernel2D &kernel,
                                  cv::Mat &piwe_local, cv::Mat &iwe_local) {
  accumulateRange(xw, yw, pol01, i0, i1, width, height, kernel, piwe_local,
                  iwe_local);
}

void accumulateIweGaussian(const std::vector<float> &xw,
                           const std::vector<float> &yw,
                           const std::vector<uint8_t> &pol01, int width,
                           int height, const GaussianKernel2D &kernel,
                           cv::Mat &piwe_out, cv::Mat &iwe_out,
                           int num_threads) {
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("accumulateIweGaussian: invalid image size");
  }
  if (xw.size() != yw.size() || xw.size() != pol01.size()) {
    throw std::runtime_error(
        "accumulateIweGaussian: input vector size mismatch");
  }
  if (num_threads <= 0) {
    num_threads = 1;
  }

  piwe_out = cv::Mat::zeros(height, width, CV_32F);
  iwe_out = cv::Mat::zeros(height, width, CV_32F);

  const size_t n = xw.size();
  if (n == 0) {
    return;
  }

  const int tcount = std::min<int>(num_threads, static_cast<int>(n));
  std::vector<cv::Mat> piwe_locals;
  std::vector<cv::Mat> iwe_locals;
  piwe_locals.reserve(static_cast<size_t>(tcount));
  iwe_locals.reserve(static_cast<size_t>(tcount));

  for (int t = 0; t < tcount; ++t) {
    piwe_locals.push_back(cv::Mat::zeros(height, width, CV_32F));
    iwe_locals.push_back(cv::Mat::zeros(height, width, CV_32F));
  }

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(tcount));

  for (int t = 0; t < tcount; ++t) {
    const size_t i0 =
        (n * static_cast<size_t>(t)) / static_cast<size_t>(tcount);
    const size_t i1 =
        (n * static_cast<size_t>(t + 1)) / static_cast<size_t>(tcount);

    threads.emplace_back(
        accumulateRangeWorker, std::cref(xw), std::cref(yw), std::cref(pol01),
        i0, i1, width, height, std::cref(kernel),
        std::ref(piwe_locals[static_cast<size_t>(t)]),
        std::ref(iwe_locals[static_cast<size_t>(t)]));
  }

  for (auto &th : threads) {
    th.join();
  }

  for (int t = 0; t < tcount; ++t) {
    piwe_out += piwe_locals[static_cast<size_t>(t)];
    iwe_out += iwe_locals[static_cast<size_t>(t)];
  }
}

ObjectiveStats computeObjectiveStats(const cv::Mat &img, bool use_variance) {
  if (img.empty()) {
    throw std::runtime_error("computeObjectiveStats: empty image");
  }
  if (img.type() != CV_32F) {
    throw std::runtime_error("computeObjectiveStats: expected CV_32F");
  }

  ObjectiveStats st;

  const int rows = img.rows;
  const int cols = img.cols;
  const double inv_n = 1.0 / static_cast<double>(rows * cols);

  if (!use_variance) {
    double sum_sq = 0.0;
    for (int y = 0; y < rows; ++y) {
      const float *row = img.ptr<float>(y);
      for (int x = 0; x < cols; ++x) {
        const double v = static_cast<double>(row[x]);
        sum_sq += v * v;
      }
    }
    st.l2 = std::sqrt(sum_sq * inv_n);
    st.variance = 0.0;
    return st;
  }

  double mean = 0.0;
  double mean_sq = 0.0;
  for (int y = 0; y < rows; ++y) {
    const float *row = img.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      const double v = static_cast<double>(row[x]);
      mean += v;
      mean_sq += v * v;
    }
  }

  mean *= inv_n;
  mean_sq *= inv_n;

  st.variance = mean_sq - mean * mean;
  st.l2 = 0.0;

  return st;
}

} // namespace corner2tag::core
