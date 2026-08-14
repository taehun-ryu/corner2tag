#include "corner2tag/core/postprocess/checkerboard_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

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

static std::pair<double, double>
computeMeanStdFloat(const std::vector<float> &v) {
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

struct IndexByCoordAscending {
  const std::vector<double> *coord = nullptr;

  bool operator()(int a, int b) const {
    return (*coord)[static_cast<size_t>(a)] <
           (*coord)[static_cast<size_t>(b)];
  }
};

static float scoreGrid(const std::vector<cv::Point2f> &ordered, int rows,
                       int cols) {
  const int N = rows * cols;
  if (static_cast<int>(ordered.size()) != N) {
    return std::numeric_limits<float>::infinity();
  }
  std::vector<float> row_d;
  std::vector<float> col_d;
  row_d.reserve(rows * (cols - 1));
  col_d.reserve((rows - 1) * cols);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols - 1; ++c) {
      cv::Point2f v = ordered[gridIndex(r, c + 1, cols)] -
                      ordered[gridIndex(r, c, cols)];
      row_d.push_back(std::sqrt(v.dot(v)));
    }
  }
  for (int r = 0; r < rows - 1; ++r) {
    for (int c = 0; c < cols; ++c) {
      cv::Point2f v = ordered[gridIndex(r + 1, c, cols)] -
                      ordered[gridIndex(r, c, cols)];
      col_d.push_back(std::sqrt(v.dot(v)));
    }
  }

  const auto [rm, rs] = computeMeanStdFloat(row_d);
  const auto [cm, cs] = computeMeanStdFloat(col_d);
  const float rcv = static_cast<float>(rs / (rm + 1e-9));
  const float ccv = static_cast<float>(cs / (cm + 1e-9));

  std::vector<float> ortho;
  ortho.reserve(static_cast<size_t>((rows - 1) * (cols - 1)));
  for (int r = 0; r < rows - 1; ++r) {
    for (int c = 0; c < cols - 1; ++c) {
      cv::Point2f dx = ordered[gridIndex(r, c + 1, cols)] -
                       ordered[gridIndex(r, c, cols)];
      cv::Point2f dy = ordered[gridIndex(r + 1, c, cols)] -
                       ordered[gridIndex(r, c, cols)];
      const float ndx = std::sqrt(dx.dot(dx));
      const float ndy = std::sqrt(dy.dot(dy));
      if (ndx < 1e-9f || ndy < 1e-9f) {
        continue;
      }
      ortho.push_back(std::abs(dx.dot(dy) / (ndx * ndy)));
    }
  }

  float oerr = 1.0f;
  if (!ortho.empty()) {
    double sum = 0.0;
    for (float v : ortho) {
      sum += v;
    }
    oerr = static_cast<float>(sum / ortho.size());
  }

  return rcv + ccv + oerr;
}

static bool solveGridFromProjectionExact(const std::vector<cv::Point2f> &points,
                                         int rows, int cols, int axis_rows,
                                         std::vector<cv::Point2f> *ordered,
                                         float *score) {
  const int K = rows * cols;
  if (static_cast<int>(points.size()) != K) {
    return false;
  }

  cv::Point2d mean(0.0, 0.0);
  for (const auto &p : points) {
    mean.x += p.x;
    mean.y += p.y;
  }
  mean.x /= static_cast<double>(K);
  mean.y /= static_cast<double>(K);

  cv::Mat X(K, 2, CV_64F);
  for (int i = 0; i < K; ++i) {
    X.at<double>(i, 0) = points[static_cast<size_t>(i)].x - mean.x;
    X.at<double>(i, 1) = points[static_cast<size_t>(i)].y - mean.y;
  }

  cv::SVD svd(X, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  cv::Vec2d u(svd.vt.at<double>(0, 0), svd.vt.at<double>(0, 1));
  cv::Vec2d v(svd.vt.at<double>(1, 0), svd.vt.at<double>(1, 1));
  if (u.dot(cv::Vec2d(1.0, 0.0)) < 0.0) {
    u = -u;
  }
  if (v.dot(cv::Vec2d(0.0, 1.0)) < 0.0) {
    v = -v;
  }

  std::vector<double> proj_u(static_cast<size_t>(K));
  std::vector<double> proj_v(static_cast<size_t>(K));
  for (int i = 0; i < K; ++i) {
    const double x = X.at<double>(i, 0);
    const double y = X.at<double>(i, 1);
    proj_u[static_cast<size_t>(i)] = x * u[0] + y * u[1];
    proj_v[static_cast<size_t>(i)] = x * v[0] + y * v[1];
  }

  const std::vector<double> &row_coord = (axis_rows == 1) ? proj_v : proj_u;
  const std::vector<double> &col_coord = (axis_rows == 1) ? proj_u : proj_v;

  std::vector<int> by_row(static_cast<size_t>(K));
  std::iota(by_row.begin(), by_row.end(), 0);
  std::sort(by_row.begin(), by_row.end(),
            IndexByCoordAscending{&row_coord});

  ordered->clear();
  ordered->reserve(static_cast<size_t>(K));

  for (int r = 0; r < rows; ++r) {
    const int start = r * cols;
    const int end = start + cols;
    std::vector<int> row_idxs(by_row.begin() + start, by_row.begin() + end);
    std::sort(row_idxs.begin(), row_idxs.end(),
              IndexByCoordAscending{&col_coord});

    for (int idx : row_idxs) {
      ordered->push_back(points[static_cast<size_t>(idx)]);
    }
  }

  *ordered = reshapeGrid(*ordered, rows, cols);
  *score = scoreGrid(*ordered, rows, cols);
  return std::isfinite(*score);
}

CheckerboardOrderResult
orderCheckerboardCorners(const std::vector<cv::Point2f> &points, int rows,
                         int cols) {
  CheckerboardOrderResult out;
  if (rows <= 0 || cols <= 0) {
    return out;
  }

  const int K = rows * cols;
  if (static_cast<int>(points.size()) != K) {
    return out;
  }

  std::vector<cv::Point2f> cand1;
  std::vector<cv::Point2f> cand2;
  float s1 = std::numeric_limits<float>::infinity();
  float s2 = std::numeric_limits<float>::infinity();

  const bool ok1 =
      solveGridFromProjectionExact(points, rows, cols, 1, &cand1, &s1);
  const bool ok2 =
      solveGridFromProjectionExact(points, rows, cols, 0, &cand2, &s2);

  if (!ok1 && !ok2) {
    return out;
  }
  if (!ok2 || (ok1 && s1 <= s2)) {
    out.success = true;
    out.ordered = std::move(cand1);
    out.score = s1;
    return out;
  }

  out.success = true;
  out.ordered = std::move(cand2);
  out.score = s2;
  return out;
}

} // namespace corner2tag::core
