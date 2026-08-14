#include "corner2tag/core/postprocess/checkerboard_postprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace corner2tag::core {

static double percentileCopy(std::vector<double> vals, double q01) {
  if (vals.empty()) {
    return 0.0;
  }
  if (q01 < 0.0) {
    q01 = 0.0;
  }
  if (q01 > 1.0) {
    q01 = 1.0;
  }
  const size_t k = static_cast<size_t>(
      std::round(q01 * static_cast<double>(vals.size() - 1)));
  std::nth_element(vals.begin(), vals.begin() + static_cast<long>(k),
                   vals.end());
  return vals[k];
}

static std::vector<cv::Point2f> buildIdealGridPoints(int rows, int cols) {
  std::vector<cv::Point2f> ideal;
  ideal.reserve(static_cast<size_t>(rows * cols));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      ideal.emplace_back(static_cast<float>(c), static_cast<float>(r));
    }
  }
  return ideal;
}

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

static cv::Point2f uvToXy(const cv::Point2d &mean, const cv::Vec2d &u,
                          const cv::Vec2d &v, double a, double b) {
  return cv::Point2f(static_cast<float>(mean.x + a * u[0] + b * v[0]),
                     static_cast<float>(mean.y + a * u[1] + b * v[1]));
}

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

static void kmeans1d(const std::vector<double> &x, int k,
                     std::vector<int> *labels_out,
                     std::vector<double> *centers_out) {
  const int n = static_cast<int>(x.size());
  labels_out->assign(n, 0);
  centers_out->assign(k, 0.0);
  if (k <= 1) {
    double mean = 0.0;
    for (double v : x) {
      mean += v;
    }
    mean /= std::max(1, n);
    (*centers_out)[0] = mean;
    return;
  }

  std::vector<double> centers(static_cast<size_t>(k), 0.0);
  for (int i = 0; i < k; ++i) {
    const double q = (i + 1.0) / (k + 1.0);
    const int idx = static_cast<int>(std::round(q * (n - 1)));
    centers[static_cast<size_t>(i)] = x[std::max(0, std::min(n - 1, idx))];
  }

  for (int it = 0; it < 60; ++it) {
    bool changed = false;
    for (int i = 0; i < n; ++i) {
      double best = std::numeric_limits<double>::infinity();
      int best_k = 0;
      for (int j = 0; j < k; ++j) {
        const double d = std::abs(x[static_cast<size_t>(i)] -
                                  centers[static_cast<size_t>(j)]);
        if (d < best) {
          best = d;
          best_k = j;
        }
      }
      if ((*labels_out)[static_cast<size_t>(i)] != best_k) {
        (*labels_out)[static_cast<size_t>(i)] = best_k;
        changed = true;
      }
    }

    std::vector<double> new_centers(static_cast<size_t>(k), 0.0);
    std::vector<int> counts(static_cast<size_t>(k), 0);
    for (int i = 0; i < n; ++i) {
      const int lbl = (*labels_out)[static_cast<size_t>(i)];
      new_centers[static_cast<size_t>(lbl)] += x[static_cast<size_t>(i)];
      counts[static_cast<size_t>(lbl)]++;
    }
    for (int j = 0; j < k; ++j) {
      if (counts[static_cast<size_t>(j)] > 0) {
        new_centers[static_cast<size_t>(j)] /=
            counts[static_cast<size_t>(j)];
      } else {
        new_centers[static_cast<size_t>(j)] = centers[static_cast<size_t>(j)];
      }
    }

    centers = std::move(new_centers);
    if (!changed) {
      break;
    }
  }

  *centers_out = std::move(centers);
}

static double rowWindowLinearityScore(const std::vector<int> &idxs_sorted,
                                      int start, int cols,
                                      const std::vector<double> &col_coord) {
  const int n = cols;
  if (n < 2) {
    return std::numeric_limits<double>::infinity();
  }

  const double sum_j =
      0.5 * static_cast<double>(n - 1) * static_cast<double>(n);
  const double sum_jj = static_cast<double>(n - 1) * static_cast<double>(n) *
                        static_cast<double>(2 * n - 1) / 6.0;

  double sum_x = 0.0;
  double sum_jx = 0.0;
  for (int j = 0; j < n; ++j) {
    const double x =
        col_coord[static_cast<size_t>(idxs_sorted[static_cast<size_t>(start + j)])];
    sum_x += x;
    sum_jx += static_cast<double>(j) * x;
  }

  const double dn = static_cast<double>(n);
  const double denom = dn * sum_jj - sum_j * sum_j;
  if (std::abs(denom) < 1e-12) {
    return std::numeric_limits<double>::infinity();
  }

  const double b = (dn * sum_jx - sum_j * sum_x) / denom;
  if (b <= 1e-6) {
    return std::numeric_limits<double>::infinity();
  }
  const double a = (sum_x - b * sum_j) / dn;

  double mse = 0.0;
  for (int j = 0; j < n; ++j) {
    const double x =
        col_coord[static_cast<size_t>(idxs_sorted[static_cast<size_t>(start + j)])];
    const double pred = a + b * static_cast<double>(j);
    const double e = x - pred;
    mse += e * e;
  }
  mse /= dn;

  return std::sqrt(mse) / (std::abs(b) + 1e-6);
}

static bool selectBestRowSubset(const std::vector<int> &idxs_sorted,
                                const std::vector<double> &col_coord, int cols,
                                std::vector<int> *selected) {
  selected->clear();
  const int m = static_cast<int>(idxs_sorted.size());
  if (m < cols) {
    return false;
  }
  if (m == cols) {
    *selected = idxs_sorted;
    return true;
  }

  double best_score = std::numeric_limits<double>::infinity();
  int best_start = -1;
  for (int start = 0; start <= m - cols; ++start) {
    const double s = rowWindowLinearityScore(idxs_sorted, start, cols, col_coord);
    if (s < best_score) {
      best_score = s;
      best_start = start;
    }
  }

  if (best_start < 0) {
    const int mid = m / 2;
    int start = std::max(0, mid - cols / 2);
    if (start + cols > m) {
      start = m - cols;
    }
    best_start = start;
  }

  selected->assign(idxs_sorted.begin() + best_start,
                   idxs_sorted.begin() + best_start + cols);
  return true;
}

static bool buildIndexMapFromRowsFilter(const std::vector<double> &row_coord,
                                        const std::vector<double> &col_coord,
                                        int rows, int cols,
                                        std::vector<int> *index_map) {
  const int N = static_cast<int>(row_coord.size());
  const int K = rows * cols;
  if (static_cast<int>(col_coord.size()) != N) {
    return false;
  }

  std::vector<int> rlab;
  std::vector<double> rcent;
  kmeans1d(row_coord, rows, &rlab, &rcent);

  std::vector<int> row_order(static_cast<size_t>(rows));
  std::iota(row_order.begin(), row_order.end(), 0);
  std::sort(row_order.begin(), row_order.end(),
            IndexByCoordAscending{&rcent});

  index_map->clear();
  index_map->reserve(static_cast<size_t>(K));

  for (int rid : row_order) {
    std::vector<int> idxs;
    for (int i = 0; i < N; ++i) {
      if (rlab[static_cast<size_t>(i)] == rid) {
        idxs.push_back(i);
      }
    }
    if (idxs.empty()) {
      return false;
    }

    std::sort(idxs.begin(), idxs.end(), IndexByCoordAscending{&col_coord});

    std::vector<int> row_sel;
    if (!selectBestRowSubset(idxs, col_coord, cols, &row_sel)) {
      return false;
    }
    index_map->insert(index_map->end(), row_sel.begin(), row_sel.end());
  }

  return static_cast<int>(index_map->size()) == K;
}

static bool solveGridFromProjectionFilter(const std::vector<cv::Point2f> &points,
                                          int rows, int cols, int axis_rows,
                                          std::vector<cv::Point2f> *ordered,
                                          float *score) {
  const int N = static_cast<int>(points.size());
  const int K = rows * cols;
  if (N <= 0 || K <= 0) {
    return false;
  }

  cv::Point2d mean(0.0, 0.0);
  for (const auto &p : points) {
    mean.x += p.x;
    mean.y += p.y;
  }
  mean.x /= std::max(1, N);
  mean.y /= std::max(1, N);

  cv::Mat X(N, 2, CV_64F);
  for (int i = 0; i < N; ++i) {
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

  std::vector<double> proj_u(static_cast<size_t>(N));
  std::vector<double> proj_v(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    const double x = X.at<double>(i, 0);
    const double y = X.at<double>(i, 1);
    proj_u[static_cast<size_t>(i)] = x * u[0] + y * u[1];
    proj_v[static_cast<size_t>(i)] = x * v[0] + y * v[1];
  }

  const std::vector<double> &row_coord = (axis_rows == 1) ? proj_v : proj_u;
  const std::vector<double> &col_coord = (axis_rows == 1) ? proj_u : proj_v;

  std::vector<int> index_map;
  if (!buildIndexMapFromRowsFilter(row_coord, col_coord, rows, cols,
                                   &index_map)) {
    return false;
  }

  ordered->resize(static_cast<size_t>(K));
  for (int i = 0; i < K; ++i) {
    (*ordered)[static_cast<size_t>(i)] =
        points[static_cast<size_t>(index_map[static_cast<size_t>(i)])];
  }
  *ordered = reshapeGrid(*ordered, rows, cols);
  *score = scoreGrid(*ordered, rows, cols);
  return std::isfinite(*score);
}

static void buildRectHomographyHypotheses(const std::vector<cv::Point2f> &points,
                                          int rows, int cols,
                                          std::vector<cv::Mat> *out_hypotheses) {
  const int n = static_cast<int>(points.size());
  if (n < 4 || rows < 2 || cols < 2) {
    return;
  }

  cv::Point2d mean(0.0, 0.0);
  for (const auto &p : points) {
    mean.x += p.x;
    mean.y += p.y;
  }
  mean.x /= static_cast<double>(n);
  mean.y /= static_cast<double>(n);

  cv::Mat X(n, 2, CV_64F);
  for (int i = 0; i < n; ++i) {
    X.at<double>(i, 0) = points[static_cast<size_t>(i)].x - mean.x;
    X.at<double>(i, 1) = points[static_cast<size_t>(i)].y - mean.y;
  }

  cv::SVD svd(X, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  cv::Vec2d u(svd.vt.at<double>(0, 0), svd.vt.at<double>(0, 1));
  cv::Vec2d v(svd.vt.at<double>(1, 0), svd.vt.at<double>(1, 1));

  std::vector<double> pu;
  std::vector<double> pv;
  pu.reserve(static_cast<size_t>(n));
  pv.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double x = X.at<double>(i, 0);
    const double y = X.at<double>(i, 1);
    pu.push_back(x * u[0] + y * u[1]);
    pv.push_back(x * v[0] + y * v[1]);
  }

  const double ulo = percentileCopy(pu, 0.08);
  const double uhi = percentileCopy(pu, 0.92);
  const double vlo = percentileCopy(pv, 0.08);
  const double vhi = percentileCopy(pv, 0.92);
  if ((uhi - ulo) < 1e-6 || (vhi - vlo) < 1e-6) {
    return;
  }

  const std::array<cv::Point2f, 4> rect = {
      uvToXy(mean, u, v, ulo, vlo), uvToXy(mean, u, v, uhi, vlo),
      uvToXy(mean, u, v, uhi, vhi), uvToXy(mean, u, v, ulo, vhi)};

  const std::array<cv::Point2f, 4> gcorners = {
      cv::Point2f(0.0f, 0.0f), cv::Point2f(static_cast<float>(cols - 1), 0.0f),
      cv::Point2f(static_cast<float>(cols - 1), static_cast<float>(rows - 1)),
      cv::Point2f(0.0f, static_cast<float>(rows - 1))};

  const int perms[8][4] = {
      {0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2},
      {1, 0, 3, 2}, {2, 1, 0, 3}, {3, 2, 1, 0}, {0, 3, 2, 1}};

  std::vector<cv::Point2f> src(4);
  std::vector<cv::Point2f> dst(4);
  for (int i = 0; i < 4; ++i) {
    src[static_cast<size_t>(i)] = gcorners[static_cast<size_t>(i)];
  }

  for (int p = 0; p < 8; ++p) {
    for (int i = 0; i < 4; ++i) {
      dst[static_cast<size_t>(i)] = rect[static_cast<size_t>(perms[p][i])];
    }
    const cv::Mat H = cv::findHomography(src, dst, 0);
    if (!H.empty()) {
      out_hypotheses->push_back(H);
    }
  }
}

static bool hungarianMinCost(const std::vector<std::vector<double>> &cost,
                             std::vector<int> *col_for_row,
                             double *total_cost) {
  const int n = static_cast<int>(cost.size());
  if (n <= 0) {
    return false;
  }
  const int m = static_cast<int>(cost[0].size());
  if (m < n) {
    return false;
  }
  for (const auto &row : cost) {
    if (static_cast<int>(row.size()) != m) {
      return false;
    }
  }

  const double INF = 1e30;
  std::vector<double> u(static_cast<size_t>(n + 1), 0.0);
  std::vector<double> v(static_cast<size_t>(m + 1), 0.0);
  std::vector<int> p(static_cast<size_t>(m + 1), 0);
  std::vector<int> way(static_cast<size_t>(m + 1), 0);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(static_cast<size_t>(m + 1), INF);
    std::vector<unsigned char> used(static_cast<size_t>(m + 1), 0U);

    do {
      used[static_cast<size_t>(j0)] = 1U;
      const int i0 = p[static_cast<size_t>(j0)];
      int j1 = 0;
      double delta = INF;
      for (int j = 1; j <= m; ++j) {
        if (used[static_cast<size_t>(j)] != 0U) {
          continue;
        }
        const double cur = cost[static_cast<size_t>(i0 - 1)]
                               [static_cast<size_t>(j - 1)] -
                           u[static_cast<size_t>(i0)] -
                           v[static_cast<size_t>(j)];
        if (cur < minv[static_cast<size_t>(j)]) {
          minv[static_cast<size_t>(j)] = cur;
          way[static_cast<size_t>(j)] = j0;
        }
        if (minv[static_cast<size_t>(j)] < delta) {
          delta = minv[static_cast<size_t>(j)];
          j1 = j;
        }
      }
      for (int j = 0; j <= m; ++j) {
        if (used[static_cast<size_t>(j)] != 0U) {
          u[static_cast<size_t>(p[static_cast<size_t>(j)])] += delta;
          v[static_cast<size_t>(j)] -= delta;
        } else {
          minv[static_cast<size_t>(j)] -= delta;
        }
      }
      j0 = j1;
    } while (p[static_cast<size_t>(j0)] != 0);

    do {
      const int j1 = way[static_cast<size_t>(j0)];
      p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
      j0 = j1;
    } while (j0 != 0);
  }

  col_for_row->assign(static_cast<size_t>(n), -1);
  for (int j = 1; j <= m; ++j) {
    const int i = p[static_cast<size_t>(j)];
    if (i > 0) {
      (*col_for_row)[static_cast<size_t>(i - 1)] = j - 1;
    }
  }

  for (int i = 0; i < n; ++i) {
    if ((*col_for_row)[static_cast<size_t>(i)] < 0) {
      return false;
    }
  }

  if (total_cost) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      sum += cost[static_cast<size_t>(i)]
                 [static_cast<size_t>((*col_for_row)[static_cast<size_t>(i)])];
    }
    *total_cost = sum;
  }

  return true;
}

static bool assignGridNodesHungarian(const std::vector<cv::Point2f> &canon_all,
                                     int rows, int cols,
                                     std::vector<int> *node_to_point,
                                     float *mean_residual_out,
                                     float *q90_residual_out) {
  const int K = rows * cols;
  const int N = static_cast<int>(canon_all.size());
  if (K <= 0 || N < K) {
    return false;
  }

  std::vector<std::vector<double>> cost(
      static_cast<size_t>(K), std::vector<double>(static_cast<size_t>(N), 1e9));

  for (int node = 0; node < K; ++node) {
    const int r = node / cols;
    const int c = node % cols;
    for (int i = 0; i < N; ++i) {
      const cv::Point2f q = canon_all[static_cast<size_t>(i)];
      if (!std::isfinite(q.x) || !std::isfinite(q.y)) {
        continue;
      }
      const double dx = static_cast<double>(q.x) - static_cast<double>(c);
      const double dy = static_cast<double>(q.y) - static_cast<double>(r);
      const double d2 = dx * dx + dy * dy;

      double ox = 0.0;
      double oy = 0.0;
      if (q.x < -0.6f) {
        ox = -0.6 - static_cast<double>(q.x);
      } else if (q.x > static_cast<float>(cols - 1) + 0.6f) {
        ox = static_cast<double>(q.x) - (static_cast<double>(cols - 1) + 0.6);
      }
      if (q.y < -0.6f) {
        oy = -0.6 - static_cast<double>(q.y);
      } else if (q.y > static_cast<float>(rows - 1) + 0.6f) {
        oy = static_cast<double>(q.y) - (static_cast<double>(rows - 1) + 0.6);
      }
      const double out2 = ox * ox + oy * oy;

      cost[static_cast<size_t>(node)][static_cast<size_t>(i)] = d2 + 4.0 * out2;
    }
  }

  std::vector<int> assigned_col;
  double total_cost = 0.0;
  if (!hungarianMinCost(cost, &assigned_col, &total_cost)) {
    return false;
  }
  (void)total_cost;

  node_to_point->assign(static_cast<size_t>(K), -1);
  std::vector<double> residuals;
  residuals.reserve(static_cast<size_t>(K));
  double sum_res = 0.0;

  for (int node = 0; node < K; ++node) {
    const int point_idx = assigned_col[static_cast<size_t>(node)];
    if (point_idx < 0 || point_idx >= N) {
      return false;
    }
    (*node_to_point)[static_cast<size_t>(node)] = point_idx;

    const int r = node / cols;
    const int c = node % cols;
    const cv::Point2f q = canon_all[static_cast<size_t>(point_idx)];
    const double dx = static_cast<double>(q.x) - static_cast<double>(c);
    const double dy = static_cast<double>(q.y) - static_cast<double>(r);
    const double res = std::sqrt(dx * dx + dy * dy);
    residuals.push_back(res);
    sum_res += res;
  }

  if (mean_residual_out) {
    *mean_residual_out = static_cast<float>(sum_res / static_cast<double>(K));
  }
  if (q90_residual_out) {
    *q90_residual_out = static_cast<float>(percentileCopy(residuals, 0.90));
  }
  return true;
}

static bool refineLatticeFromHomography(const std::vector<cv::Point2f> &all_points,
                                        const cv::Mat &H_init, int rows, int cols,
                                        std::vector<cv::Point2f> *snapped_out,
                                        float *mean_residual_out) {
  const int K = rows * cols;
  const int N = static_cast<int>(all_points.size());
  if (K <= 0 || N < K) {
    return false;
  }

  const std::vector<cv::Point2f> ideal = buildIdealGridPoints(rows, cols);

  cv::Mat H;
  H_init.convertTo(H, CV_64F);
  if (H.empty() || H.rows != 3 || H.cols != 3) {
    return false;
  }

  float mean_residual = std::numeric_limits<float>::infinity();
  float q90_residual = std::numeric_limits<float>::infinity();
  std::vector<cv::Point2f> current(static_cast<size_t>(K));

  for (int iter = 0; iter < 4; ++iter) {
    const cv::Mat Hinv = H.inv();
    if (Hinv.empty()) {
      return false;
    }
    const cv::Matx33d A = Hinv;

    std::vector<cv::Point2f> canon_all(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
      const cv::Point2f &p = all_points[static_cast<size_t>(i)];
      const cv::Matx31d v(p.x, p.y, 1.0);
      const cv::Matx31d q = A * v;
      const double w = q(2, 0);
      if (std::abs(w) < 1e-12) {
        canon_all[static_cast<size_t>(i)] =
            cv::Point2f(std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity());
      } else {
        canon_all[static_cast<size_t>(i)] =
            cv::Point2f(static_cast<float>(q(0, 0) / w),
                        static_cast<float>(q(1, 0) / w));
      }
    }

    std::vector<int> node_to_point;
    if (!assignGridNodesHungarian(canon_all, rows, cols, &node_to_point,
                                  &mean_residual, &q90_residual)) {
      return false;
    }

    for (int node = 0; node < K; ++node) {
      const int point_idx = node_to_point[static_cast<size_t>(node)];
      current[static_cast<size_t>(node)] =
          all_points[static_cast<size_t>(point_idx)];
    }

    const cv::Mat H_new = cv::findHomography(ideal, current, 0);
    if (H_new.empty()) {
      return false;
    }
    H = H_new;
  }

  *snapped_out = reshapeGrid(current, rows, cols);
  if (mean_residual_out) {
    *mean_residual_out = 0.5f * mean_residual + 0.5f * q90_residual;
  }
  return true;
}

static bool snapSeedToLattice(const std::vector<cv::Point2f> &all_points,
                              const std::vector<cv::Point2f> &seed_ordered,
                              int rows, int cols,
                              std::vector<cv::Point2f> *snapped_out,
                              float *mean_residual_out) {
  const int K = rows * cols;
  const int N = static_cast<int>(all_points.size());
  if (K <= 0 || N < K || static_cast<int>(seed_ordered.size()) != K) {
    return false;
  }

  const std::vector<cv::Point2f> ideal = buildIdealGridPoints(rows, cols);
  const cv::Mat H_seed = cv::findHomography(ideal, seed_ordered, 0);
  if (H_seed.empty()) {
    return false;
  }
  return refineLatticeFromHomography(all_points, H_seed, rows, cols, snapped_out,
                                     mean_residual_out);
}

struct FilterCandidate {
  bool success = false;
  std::vector<cv::Point2f> ordered;
  float composite_score = std::numeric_limits<float>::infinity();
  float grid_score = std::numeric_limits<float>::infinity();
  float mean_residual = std::numeric_limits<float>::infinity();
};

static bool isFilterCandidateAcceptable(const FilterCandidate &cand, int K) {
  if (!cand.success) {
    return false;
  }
  if (static_cast<int>(cand.ordered.size()) != K) {
    return false;
  }
  if (!std::isfinite(cand.grid_score) || !std::isfinite(cand.composite_score)) {
    return false;
  }
  return true;
}

static void updateBestFilterCandidate(const FilterCandidate &cand, int K,
                                      FilterCandidate *best) {
  if (!isFilterCandidateAcceptable(cand, K)) {
    return;
  }
  if (!best->success || cand.composite_score < best->composite_score) {
    *best = cand;
  }
}

static FilterCandidate evaluateHomographyCandidate(
    const std::vector<cv::Point2f> &points, const cv::Mat &H, int rows, int cols,
    float prior_bias) {
  FilterCandidate cand;
  std::vector<cv::Point2f> snapped;
  float mean_residual = std::numeric_limits<float>::infinity();
  if (!refineLatticeFromHomography(points, H, rows, cols, &snapped,
                                   &mean_residual)) {
    return cand;
  }

  const float g = scoreGrid(snapped, rows, cols);
  if (!std::isfinite(g)) {
    return cand;
  }
  cand.success = true;
  cand.ordered = std::move(snapped);
  cand.grid_score = g;
  cand.mean_residual = mean_residual;
  cand.composite_score = g + 0.55f * mean_residual + prior_bias;
  return cand;
}

static FilterCandidate evaluateSeedCandidate(
    const std::vector<cv::Point2f> &points, const std::vector<cv::Point2f> &seed,
    int rows, int cols, float seed_score) {
  FilterCandidate cand;
  const int K = rows * cols;
  if (static_cast<int>(seed.size()) != K) {
    return cand;
  }

  std::vector<cv::Point2f> snapped;
  float mean_residual = std::numeric_limits<float>::infinity();
  if (snapSeedToLattice(points, seed, rows, cols, &snapped, &mean_residual)) {
    const float g = scoreGrid(snapped, rows, cols);
    if (std::isfinite(g)) {
      cand.success = true;
      cand.ordered = std::move(snapped);
      cand.grid_score = g;
      cand.mean_residual = mean_residual;
      cand.composite_score = g + 0.45f * mean_residual;
      return cand;
    }
  }

  const float g = scoreGrid(seed, rows, cols);
  if (!std::isfinite(g)) {
    return cand;
  }
  cand.success = true;
  cand.ordered = seed;
  cand.grid_score = g;
  cand.mean_residual = std::isfinite(seed_score) ? seed_score : 0.0f;
  cand.composite_score = g + 0.15f * cand.mean_residual;
  return cand;
}

CheckerboardFilterResult
filterCheckerboardCorners(const std::vector<cv::Point2f> &points, int rows,
                          int cols) {
  CheckerboardFilterResult out;
  if (rows <= 0 || cols <= 0) {
    return out;
  }

  const int K = rows * cols;
  if (static_cast<int>(points.size()) < K) {
    return out;
  }

  FilterCandidate best;

  std::vector<cv::Point2f> seed1;
  std::vector<cv::Point2f> seed2;
  float s1 = std::numeric_limits<float>::infinity();
  float s2 = std::numeric_limits<float>::infinity();
  const bool ok1 = solveGridFromProjectionFilter(points, rows, cols, 1, &seed1,
                                                  &s1);
  const bool ok2 = solveGridFromProjectionFilter(points, rows, cols, 0, &seed2,
                                                  &s2);

  if (ok1) {
    updateBestFilterCandidate(
        evaluateSeedCandidate(points, seed1, rows, cols, s1), K, &best);
  }
  if (ok2) {
    updateBestFilterCandidate(
        evaluateSeedCandidate(points, seed2, rows, cols, s2), K, &best);
  }

  std::vector<cv::Mat> h2d;
  buildRectHomographyHypotheses(points, rows, cols, &h2d);
  for (const auto &H : h2d) {
    updateBestFilterCandidate(
        evaluateHomographyCandidate(points, H, rows, cols, 0.05f), K, &best);
  }

  if (!best.success) {
    return out;
  }

  out.filtered = std::move(best.ordered);
  out.score = best.grid_score;
  out.success = static_cast<int>(out.filtered.size()) == K;
  return out;
}

} // namespace corner2tag::core
