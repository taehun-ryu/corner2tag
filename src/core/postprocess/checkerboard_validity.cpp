#include "corner2tag/core/postprocess/checkerboard_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace corner2tag::core {

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

static int gridIndex(int r, int c, int cols) { return r * cols + c; }

static bool monotonicGridOk(const std::vector<cv::Point2f> &grid, int rows,
                            int cols) {
  constexpr float eps_mono = 1e-3f;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols - 1; ++c) {
      if (grid[gridIndex(r, c + 1, cols)].x <
          grid[gridIndex(r, c, cols)].x - eps_mono) {
        return false;
      }
    }
  }
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows - 1; ++r) {
      if (grid[gridIndex(r + 1, c, cols)].y <
          grid[gridIndex(r, c, cols)].y - eps_mono) {
        return false;
      }
    }
  }
  return true;
}

static std::pair<double, double> computeMeanStd(const std::vector<float> &v) {
  double mean = 0.0;
  for (float x : v) {
    mean += x;
  }
  mean /= std::max<size_t>(1, v.size());

  double var = 0.0;
  for (float x : v) {
    const double d = x - mean;
    var += d * d;
  }
  var /= std::max<size_t>(1, v.size());
  return std::pair<double, double>(mean, std::sqrt(var));
}

static float computeQuantile(std::vector<float> v, float q) {
  if (v.empty()) {
    return 0.0f;
  }
  q = std::clamp(q, 0.0f, 1.0f);
  const size_t k =
      static_cast<size_t>(std::round(q * static_cast<float>(v.size() - 1)));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

static std::pair<float, float> computeRelativeDeviationStats(
    const std::vector<float> &vals, float center) {
  std::vector<float> rel;
  rel.reserve(vals.size());
  float mx = 0.0f;
  for (float x : vals) {
    const float d = std::abs(x - center) / (center + 1e-6f);
    rel.push_back(d);
    mx = std::max(mx, d);
  }
  return std::pair<float, float>(computeQuantile(rel, 0.9f), mx);
}

static float computeMad(const std::vector<float> &v) {
  if (v.empty()) {
    return 0.0f;
  }
  std::vector<float> tmp = v;
  std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
  const float med = tmp[tmp.size() / 2];
  for (float &x : tmp) {
    x = std::abs(x - med);
  }
  std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
  return tmp[tmp.size() / 2];
}

static float computeSignedArea(const cv::Point2f &a, const cv::Point2f &b,
                               const cv::Point2f &c) {
  return 0.5f * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

bool isCheckerboardValid(const std::vector<cv::Point2f> &ordered, int rows,
                         int cols, float tor_spacing, float tor_orth) {
  const int N = rows * cols;
  if (static_cast<int>(ordered.size()) != N) {
    return false;
  }
  if (rows < 2 || cols < 2) {
    return false;
  }

  std::vector<cv::Point2f> grid = ordered;

  std::vector<cv::Point2f> G;
  G.reserve(static_cast<size_t>(N));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      G.emplace_back(static_cast<float>(c), static_cast<float>(r));
    }
  }

  cv::Mat H = cv::findHomography(G, grid, 0);
  if (H.empty()) {
    return false;
  }
  cv::Mat Hinv = H.inv();

  std::vector<cv::Point2f> canon;
  canon.reserve(static_cast<size_t>(N));
  for (const auto &p : grid) {
    cv::Matx31d v(p.x, p.y, 1.0);
    cv::Matx31d q = cv::Matx33d(Hinv) * v;
    const double w = q(2, 0);
    if (std::abs(w) < 1e-9) {
      return false;
    }
    canon.emplace_back(static_cast<float>(q(0, 0) / w),
                       static_cast<float>(q(1, 0) / w));
  }

  std::vector<cv::Point2f> canon_grid = canon;

  if (!monotonicGridOk(canon_grid, rows, cols)) {
    canon_grid = reshapeGrid(canon_grid, rows, cols);
    if (!monotonicGridOk(canon_grid, rows, cols)) {
      return false;
    }
  }

  std::vector<float> row_d;
  std::vector<float> col_d;
  row_d.reserve(static_cast<size_t>(rows * (cols - 1)));
  col_d.reserve(static_cast<size_t>((rows - 1) * cols));

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols - 1; ++c) {
      cv::Point2f v = canon_grid[gridIndex(r, c + 1, cols)] -
                      canon_grid[gridIndex(r, c, cols)];
      row_d.push_back(std::sqrt(v.dot(v)));
    }
  }
  for (int r = 0; r < rows - 1; ++r) {
    for (int c = 0; c < cols; ++c) {
      cv::Point2f v = canon_grid[gridIndex(r + 1, c, cols)] -
                      canon_grid[gridIndex(r, c, cols)];
      col_d.push_back(std::sqrt(v.dot(v)));
    }
  }

  const auto [rm, rs] = computeMeanStd(row_d);
  const auto [cm, cs] = computeMeanStd(col_d);
  const float row_cv = static_cast<float>(rs / (rm + 1e-6));
  const float col_cv = static_cast<float>(cs / (cm + 1e-6));

  const float spacing_tol = std::max(tor_spacing, 0.01f);
  const float row_med = computeQuantile(row_d, 0.5f);
  const float col_med = computeQuantile(col_d, 0.5f);
  const auto [row_rel_p90, row_rel_max] =
      computeRelativeDeviationStats(row_d, row_med);
  const auto [col_rel_p90, col_rel_max] =
      computeRelativeDeviationStats(col_d, col_med);
  const bool row_ok = row_cv < spacing_tol && row_rel_p90 < spacing_tol * 1.25f &&
                      row_rel_max < spacing_tol * 2.0f;
  const bool col_ok = col_cv < spacing_tol && col_rel_p90 < spacing_tol * 1.25f &&
                      col_rel_max < spacing_tol * 2.0f;

  std::vector<float> orth;
  orth.reserve(static_cast<size_t>((rows - 1) * (cols - 1)));
  for (int r = 0; r < rows - 1; ++r) {
    for (int c = 0; c < cols - 1; ++c) {
      cv::Point2f dx = canon_grid[gridIndex(r, c + 1, cols)] -
                       canon_grid[gridIndex(r, c, cols)];
      cv::Point2f dy = canon_grid[gridIndex(r + 1, c, cols)] -
                       canon_grid[gridIndex(r, c, cols)];
      const float ndx = std::sqrt(dx.dot(dx));
      const float ndy = std::sqrt(dy.dot(dy));
      if (ndx < 1e-8f || ndy < 1e-8f) {
        continue;
      }
      const float cosang = std::abs(dx.dot(dy) / (ndx * ndy));
      orth.push_back(cosang);
    }
  }

  float orth_mean = 1.0f;
  if (!orth.empty()) {
    double sum = 0.0;
    for (float v : orth) {
      sum += v;
    }
    orth_mean = static_cast<float>(sum / orth.size());
  }
  const float orth_tol = std::max(tor_orth, 0.01f);
  const float orth_p90 = computeQuantile(orth, 0.9f);
  const float orth_max =
      orth.empty() ? 1.0f : *std::max_element(orth.begin(), orth.end());
  const float orth_p90_lim = std::max(0.12f, orth_tol * 1.35f);
  const float orth_max_lim = std::max(0.18f, orth_tol * 1.8f);
  const bool orth_ok = orth_mean < orth_tol && orth_p90 < orth_p90_lim &&
                       orth_max < orth_max_lim;

  float xmin = canon_grid[0].x;
  float xmax = canon_grid[0].x;
  float ymin = canon_grid[0].y;
  float ymax = canon_grid[0].y;
  for (const auto &p : canon_grid) {
    xmin = std::min(xmin, p.x);
    xmax = std::max(xmax, p.x);
    ymin = std::min(ymin, p.y);
    ymax = std::max(ymax, p.y);
  }
  const float eps_range = 0.35f;
  const bool range_ok =
      (xmin >= -eps_range && ymin >= -eps_range &&
       xmax <= (cols - 1) + eps_range && ymax <= (rows - 1) + eps_range);

  float dx_mean = 0.0f;
  float dy_mean = 0.0f;
  float dx_max = 0.0f;
  float dy_max = 0.0f;
  for (const auto &p : canon_grid) {
    const float dx = std::abs(p.x - std::round(p.x));
    const float dy = std::abs(p.y - std::round(p.y));
    dx_mean += dx;
    dy_mean += dy;
    dx_max = std::max(dx_max, dx);
    dy_max = std::max(dy_max, dy);
  }
  dx_mean /= static_cast<float>(N);
  dy_mean /= static_cast<float>(N);
  const bool near_mean_ok = (dx_mean < 0.18f) && (dy_mean < 0.18f);
  const bool near_max_ok = (dx_max < 0.45f) && (dy_max < 0.45f);

  float max_row_mad = 0.0f;
  float max_col_mad = 0.0f;
  float mean_row_mad = 0.0f;
  float mean_col_mad = 0.0f;
  for (int r = 0; r < rows; ++r) {
    std::vector<float> ys;
    ys.reserve(static_cast<size_t>(cols));
    for (int c = 0; c < cols; ++c) {
      ys.push_back(canon_grid[gridIndex(r, c, cols)].y);
    }
    const float row_mad = computeMad(ys);
    max_row_mad = std::max(max_row_mad, row_mad);
    mean_row_mad += row_mad;
  }
  for (int c = 0; c < cols; ++c) {
    std::vector<float> xs;
    xs.reserve(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) {
      xs.push_back(canon_grid[gridIndex(r, c, cols)].x);
    }
    const float col_mad = computeMad(xs);
    max_col_mad = std::max(max_col_mad, col_mad);
    mean_col_mad += col_mad;
  }
  mean_row_mad /= static_cast<float>(rows);
  mean_col_mad /= static_cast<float>(cols);
  const bool linearity_ok = (max_row_mad < 0.18f) && (max_col_mad < 0.18f) &&
                            (mean_row_mad < 0.10f) && (mean_col_mad < 0.10f);

  float sign0 = 0.0f;
  bool area_ok = true;
  bool sign_set = false;
  std::vector<float> abs_areas;
  abs_areas.reserve(static_cast<size_t>((rows - 1) * (cols - 1)));
  for (int r = 0; r < rows - 1; ++r) {
    for (int c = 0; c < cols - 1; ++c) {
      const cv::Point2f a = canon_grid[gridIndex(r, c, cols)];
      const cv::Point2f b = canon_grid[gridIndex(r, c + 1, cols)];
      const cv::Point2f cpt = canon_grid[gridIndex(r + 1, c, cols)];
      const float s = computeSignedArea(a, b, cpt);
      abs_areas.push_back(std::abs(s));
      const float sign = (s >= 0.0f) ? 1.0f : -1.0f;
      if (!sign_set) {
        sign0 = sign;
        sign_set = true;
      } else if (sign != sign0) {
        area_ok = false;
      }
    }
  }
  if (abs_areas.empty()) {
    area_ok = false;
  } else {
    const auto [am, as] = computeMeanStd(abs_areas);
    const float area_min = *std::min_element(abs_areas.begin(), abs_areas.end());
    const float area_cv = static_cast<float>(as / (am + 1e-6));
    const bool area_min_ok = area_min > 0.12f;
    const bool area_spread_ok = area_cv < 0.45f;
    area_ok = area_ok && area_min_ok && area_spread_ok;
  }

  return (row_ok && col_ok && orth_ok && range_ok && near_mean_ok &&
          near_max_ok && linearity_ok && area_ok);
}

std::vector<cv::Point3f> buildObjectPoints(int rows, int cols,
                                           float square_size) {
  std::vector<cv::Point3f> obj;
  obj.reserve(static_cast<size_t>(rows * cols));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      obj.emplace_back(static_cast<float>(c) * square_size,
                       static_cast<float>(r) * square_size, 0.0f);
    }
  }
  return obj;
}

} // namespace corner2tag::core
