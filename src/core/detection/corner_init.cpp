#include "corner2tag/core/detection/corner_init.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace corner2tag::core {

static inline float safeAbs(float v) { return std::abs(v); }

static bool comparePairByFirstDescending(const std::pair<float, cv::Point> &a,
                                         const std::pair<float, cv::Point> &b) {
  return a.first > b.first;
}

struct GrowCandidate {
  int n_lin;
  int p_lin;
  float n_val;
  float p_val;
};

static bool compareGrowCandidate(const GrowCandidate &a,
                                 const GrowCandidate &b) {
  if (a.n_lin != b.n_lin)
    return a.n_lin < b.n_lin;
  if (a.n_val != b.n_val)
    return a.n_val > b.n_val;
  return a.p_val > b.p_val;
}

static void perimeterIndices(int h, int w, std::vector<cv::Point> &out) {
  out.clear();
  if (h < 2 || w < 2) {
    return;
  }
  for (int x = 0; x < w; ++x) {
    out.emplace_back(x, 0);
  }
  for (int y = 1; y < h - 1; ++y) {
    out.emplace_back(w - 1, y);
  }
  for (int x = w - 1; x >= 0; --x) {
    out.emplace_back(x, h - 1);
  }
  for (int y = h - 2; y >= 1; --y) {
    out.emplace_back(0, y);
  }
}

static float percentileValue(std::vector<float> &vals, float pct) {
  if (vals.empty()) {
    return 0.0f;
  }
  if (pct < 0.0f)
    pct = 0.0f;
  if (pct > 100.0f)
    pct = 100.0f;
  const float q = pct / 100.0f;
  const size_t k = static_cast<size_t>(std::floor(q * (vals.size() - 1)));
  std::nth_element(vals.begin(), vals.begin() + static_cast<long>(k),
                   vals.end());
  return vals[k];
}

static float perimeterPercentile(const cv::Mat &iwe_f32, float pct) {
  std::vector<float> vals;
  vals.reserve(static_cast<size_t>(iwe_f32.rows * iwe_f32.cols));
  std::vector<cv::Point> perim;
  perimeterIndices(iwe_f32.rows, iwe_f32.cols, perim);
  for (const auto &p : perim) {
    const float v = iwe_f32.at<float>(p.y, p.x);
    if (std::isfinite(v)) {
      vals.push_back(v);
    }
  }
  if (vals.empty()) {
    return 0.0f;
  }
  return percentileValue(vals, pct);
}

static float imagePercentile(const cv::Mat &iwe_f32, float pct) {
  std::vector<float> vals;
  vals.reserve(static_cast<size_t>(iwe_f32.rows * iwe_f32.cols));
  for (int y = 0; y < iwe_f32.rows; ++y) {
    const float *row = iwe_f32.ptr<float>(y);
    for (int x = 0; x < iwe_f32.cols; ++x) {
      const float v = row[x];
      if (std::isfinite(v)) {
        vals.push_back(v);
      }
    }
  }
  if (vals.empty()) {
    return 0.0f;
  }
  return percentileValue(vals, pct);
}

void computeGlobalIweThresholds(const cv::Mat &iwe_f32,
                                const CornerInitOptions &opt,
                                float *seed_thr_out, float *tau_thr_out) {
  float seed_thr = opt.seed_thr;
  float tau_thr = opt.tau;

  if (!iwe_f32.empty() && iwe_f32.type() == CV_32F) {
    if (opt.seed_percentile > 0.0f) {
      seed_thr = imagePercentile(iwe_f32, opt.seed_percentile);
    }
    if (opt.tau_percentile > 0.0f) {
      tau_thr = imagePercentile(iwe_f32, opt.tau_percentile);
    }
  }

  if (tau_thr >= seed_thr) {
    tau_thr = seed_thr - 1e-6f;
  }
  if (seed_thr_out) {
    *seed_thr_out = seed_thr;
  }
  if (tau_thr_out) {
    *tau_thr_out = tau_thr;
  }
}

static cv::Mat perimeterPeakMask(const cv::Mat &iwe_f32, float eps, float thr) {
  cv::Mat mask(iwe_f32.rows, iwe_f32.cols, CV_8U, cv::Scalar(0));
  std::vector<cv::Point> perim;
  perimeterIndices(iwe_f32.rows, iwe_f32.cols, perim);
  if (perim.size() < 3) {
    return mask;
  }

  const size_t n = perim.size();
  std::vector<float> v(n, -std::numeric_limits<float>::infinity());
  std::vector<uint8_t> valid(n, 0);

  for (size_t i = 0; i < n; ++i) {
    const cv::Point &p = perim[i];
    const float val = iwe_f32.at<float>(p.y, p.x);
    if (std::isfinite(val)) {
      v[i] = val;
      valid[i] = 1;
    }
  }

  for (size_t i = 0; i < n; ++i) {
    const size_t il = (i == 0) ? (n - 1) : (i - 1);
    const size_t ir = (i + 1) % n;
    if (!valid[i] || !valid[il] || !valid[ir]) {
      continue;
    }
    const float vi = v[i];
    if (vi < thr) {
      continue;
    }
    if (vi > v[il] + eps && vi > v[ir] + eps) {
      mask.at<uint8_t>(perim[i].y, perim[i].x) = 255;
    }
  }
  return mask;
}

static std::vector<cv::Point> seedsFromMask(const cv::Mat &mask_u8) {
  std::vector<cv::Point> pts;
  for (int y = 0; y < mask_u8.rows; ++y) {
    const uint8_t *row = mask_u8.ptr<uint8_t>(y);
    for (int x = 0; x < mask_u8.cols; ++x) {
      if (row[x]) {
        pts.emplace_back(x, y);
      }
    }
  }
  return pts;
}

static void filterSeedsToFour(cv::Mat &seeds_u8, const cv::Mat &iwe_f32) {
  std::vector<cv::Point> seeds = seedsFromMask(seeds_u8);
  if (seeds.size() <= 4) {
    return;
  }

  const float cx = 0.5f * static_cast<float>(iwe_f32.cols - 1);
  const float cy = 0.5f * static_cast<float>(iwe_f32.rows - 1);

  struct SeedPick {
    cv::Point p;
    float v;
    bool valid = false;
  };

  SeedPick quad[4];
  std::vector<std::pair<float, cv::Point>> all;
  all.reserve(seeds.size());

  for (const auto &p : seeds) {
    const float v = iwe_f32.at<float>(p.y, p.x);
    all.emplace_back(v, p);
    const bool right = p.x >= cx;
    const bool bottom = p.y >= cy;
    const int q = (bottom ? 2 : 0) + (right ? 1 : 0);
    if (!quad[q].valid || v > quad[q].v) {
      quad[q] = {p, v, true};
    }
  }

  std::vector<cv::Point> kept;
  kept.reserve(4);
  for (int q = 0; q < 4; ++q) {
    if (quad[q].valid) {
      kept.push_back(quad[q].p);
    }
  }

  if (kept.size() < 4) {
    std::sort(all.begin(), all.end(), comparePairByFirstDescending);
    for (const auto &kv : all) {
      if (kept.size() >= 4) {
        break;
      }
      bool exists = false;
      for (const auto &p : kept) {
        if (p == kv.second) {
          exists = true;
          break;
        }
      }
      if (!exists) {
        kept.push_back(kv.second);
      }
    }
  }

  seeds_u8.setTo(0);
  for (const auto &p : kept) {
    seeds_u8.at<uint8_t>(p.y, p.x) = 255;
  }
}

struct GrowResult {
  cv::Mat mask_u8;
  cv::Mat depth_i32;
  cv::Mat seeds_u8;
};

static GrowResult
growFromPerimeterPeaks(const cv::Mat &iwe_f32, float tau, float tau_pct,
                       float seed_thr, float seed_pct, float seed_thr_override,
                       float tau_thr_override, int max_iters) {
  const int H = iwe_f32.rows;
  const int W = iwe_f32.cols;

  GrowResult out;
  float thr = seed_thr;
  if (seed_pct > 0.0f) {
    thr = perimeterPercentile(iwe_f32, seed_pct);
  }
  float tau_thr = tau;
  if (tau_pct > 0.0f) {
    tau_thr = imagePercentile(iwe_f32, tau_pct);
  }
  if (seed_thr_override > 0.0f) {
    thr = seed_thr_override;
  }
  if (tau_thr_override > 0.0f) {
    tau_thr = tau_thr_override;
  }
  if (tau_thr >= thr) {
    tau_thr = thr - 1e-6f;
  }
  out.seeds_u8 = perimeterPeakMask(iwe_f32, 1e-6f, thr);
  filterSeedsToFour(out.seeds_u8, iwe_f32);
  out.mask_u8 = out.seeds_u8.clone();
  out.depth_i32 = cv::Mat(H, W, CV_32S, cv::Scalar(-1));

  std::vector<uint8_t> selected(static_cast<size_t>(H * W), 0);
  std::vector<uint8_t> active(static_cast<size_t>(H * W), 0);
  std::vector<float> a(static_cast<size_t>(H * W),
                       -std::numeric_limits<float>::infinity());

  for (int y = 0; y < H; ++y) {
    const float *row = iwe_f32.ptr<float>(y);
    for (int x = 0; x < W; ++x) {
      const int idx = y * W + x;
      const float v = row[x];
      if (std::isfinite(v)) {
        a[idx] = v;
      }
      if (out.seeds_u8.at<uint8_t>(y, x) != 0) {
        selected[idx] = 1;
        out.depth_i32.at<int>(y, x) = 0;
        if (a[idx] >= tau_thr) {
          active[idx] = 1;
        }
      }
    }
  }

  const int off[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                         {0, 1},   {1, -1}, {1, 0},  {1, 1}};

  for (int t = 1; t <= max_iters; ++t) {
    bool any_active = false;
    std::vector<float> best_val(static_cast<size_t>(H * W),
                                -std::numeric_limits<float>::infinity());
    std::vector<int> best_n(static_cast<size_t>(H * W), -1);

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int p_lin = y * W + x;
        if (!active[p_lin]) {
          continue;
        }
        any_active = true;
        float bv = -std::numeric_limits<float>::infinity();
        int bn = -1;
        for (int k = 0; k < 8; ++k) {
          const int ny = y + off[k][0];
          const int nx = x + off[k][1];
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) {
            continue;
          }
          const int n_lin = ny * W + nx;
          if (selected[n_lin]) {
            continue;
          }
          const float nv = a[n_lin];
          if (!(nv >= tau_thr)) {
            continue;
          }
          if (nv > bv) {
            bv = nv;
            bn = n_lin;
          }
        }
        if (bn >= 0) {
          best_val[p_lin] = bv;
          best_n[p_lin] = bn;
        }
      }
    }

    if (!any_active) {
      break;
    }

    std::vector<GrowCandidate> cands;
    cands.reserve(static_cast<size_t>(H * W / 4));
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int p_lin = y * W + x;
        if (!active[p_lin]) {
          continue;
        }
        const int n_lin = best_n[p_lin];
        if (n_lin < 0) {
          continue;
        }
        cands.push_back({n_lin, p_lin, best_val[p_lin], a[p_lin]});
      }
    }
    if (cands.empty()) {
      break;
    }

    std::sort(cands.begin(), cands.end(), compareGrowCandidate);

    std::vector<int> new_pixels;
    new_pixels.reserve(cands.size());
    int last_n = -1;
    for (const auto &c : cands) {
      if (c.n_lin == last_n) {
        continue;
      }
      last_n = c.n_lin;
      if (!selected[c.n_lin]) {
        selected[c.n_lin] = 1;
        new_pixels.push_back(c.n_lin);
      }
    }

    if (new_pixels.empty()) {
      break;
    }

    std::vector<uint8_t> active_next(static_cast<size_t>(H * W), 0);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int p_lin = y * W + x;
        if (active[p_lin] && best_val[p_lin] >= tau_thr) {
          active_next[p_lin] = 1;
        }
      }
    }

    for (int n_lin : new_pixels) {
      const int ny = n_lin / W;
      const int nx = n_lin % W;
      out.depth_i32.at<int>(ny, nx) = t;
      if (a[n_lin] >= tau_thr) {
        active_next[n_lin] = 1;
      }
    }
    active.swap(active_next);
  }

  for (int y = 0; y < H; ++y) {
    uint8_t *mr = out.mask_u8.ptr<uint8_t>(y);
    for (int x = 0; x < W; ++x) {
      if (selected[y * W + x]) {
        mr[x] = 255;
      }
    }
  }

  return out;
}

// Weighted TLS fit in 2D without Eigen.
// Direction v is the eigenvector of covariance with largest eigenvalue.
static Line2D fitLineTLS2D(const std::vector<cv::Point> &pts,
                           const cv::Mat &iwe_f32, bool weighted) {
  Line2D out;
  if (pts.size() < 2) {
    out.rms = std::numeric_limits<float>::infinity();
    return out;
  }

  // Weighted centroid
  double sw = 0.0;
  double mx = 0.0;
  double my = 0.0;

  for (const auto &p : pts) {
    double w = 1.0;
    if (weighted) {
      const float v = iwe_f32.at<float>(p.y, p.x);
      w = static_cast<double>(safeAbs(v));
      if (!(w > 0.0)) {
        w = 1.0;
      }
    }
    sw += w;
    mx += w * static_cast<double>(p.x);
    my += w * static_cast<double>(p.y);
  }
  if (!(sw > 0.0)) {
    sw = 1.0;
  }
  mx /= sw;
  my /= sw;

  out.c = cv::Point2f(static_cast<float>(mx), static_cast<float>(my));

  // Covariance
  double cxx = 0.0;
  double cxy = 0.0;
  double cyy = 0.0;

  for (const auto &p : pts) {
    double w = 1.0;
    if (weighted) {
      const float v = iwe_f32.at<float>(p.y, p.x);
      w = static_cast<double>(safeAbs(v));
      if (!(w > 0.0)) {
        w = 1.0;
      }
    }
    const double dx = static_cast<double>(p.x) - mx;
    const double dy = static_cast<double>(p.y) - my;
    cxx += w * dx * dx;
    cxy += w * dx * dy;
    cyy += w * dy * dy;
  }
  cxx /= sw;
  cxy /= sw;
  cyy /= sw;

  // Largest-eigenvector of 2x2 symmetric matrix [[cxx, cxy],[cxy, cyy]]
  // Compute via closed form:
  // angle = 0.5 * atan2(2*cxy, cxx - cyy)
  const double angle = 0.5 * std::atan2(2.0 * cxy, cxx - cyy);
  const double vx = std::cos(angle);
  const double vy = std::sin(angle);

  const double vnorm = std::sqrt(vx * vx + vy * vy);
  const double ux = (vnorm > 0.0) ? (vx / vnorm) : 1.0;
  const double uy = (vnorm > 0.0) ? (vy / vnorm) : 0.0;

  out.v = cv::Point2f(static_cast<float>(ux), static_cast<float>(uy));
  out.n = cv::Point2f(static_cast<float>(-uy), static_cast<float>(ux));
  out.d = -(out.n.x * out.c.x + out.n.y * out.c.y);

  // RMS distance
  double sum_sq = 0.0;
  for (const auto &p : pts) {
    const double sx = static_cast<double>(p.x) - static_cast<double>(out.c.x);
    const double sy = static_cast<double>(p.y) - static_cast<double>(out.c.y);
    const double dist =
        sx * static_cast<double>(out.n.x) + sy * static_cast<double>(out.n.y);
    sum_sq += dist * dist;
  }
  out.rms =
      static_cast<float>(std::sqrt(sum_sq / static_cast<double>(pts.size())));

  return out;
}

static bool intersectLines(const Line2D &a, const Line2D &b,
                           cv::Point2f &out_xy) {
  // Solve:
  // a.n.x * x + a.n.y * y + a.d = 0
  // b.n.x * x + b.n.y * y + b.d = 0
  const double A00 = a.n.x;
  const double A01 = a.n.y;
  const double A10 = b.n.x;
  const double A11 = b.n.y;

  const double B0 = -a.d;
  const double B1 = -b.d;

  const double det = A00 * A11 - A01 * A10;
  if (std::abs(det) < 1e-12) {
    return false;
  }

  const double x = (B0 * A11 - A01 * B1) / det;
  const double y = (A00 * B1 - B0 * A10) / det;
  out_xy = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
  return true;
}

static std::array<int, 4>
ccwOrderFromCentroids(const std::array<cv::Point2f, 4> &c) {
  cv::Point2f pivot(0.0f, 0.0f);
  for (int i = 0; i < 4; ++i) {
    pivot.x += c[i].x;
    pivot.y += c[i].y;
  }
  pivot.x *= 0.25f;
  pivot.y *= 0.25f;

  std::array<double, 4> ang{};
  for (int i = 0; i < 4; ++i) {
    const double dx = static_cast<double>(c[i].x - pivot.x);
    // y axis down -> flip for CCW like your python
    const double dy = static_cast<double>(-(c[i].y - pivot.y));
    ang[i] = std::atan2(dy, dx);
  }

  std::array<int, 4> idx{0, 1, 2, 3};
  std::sort(idx.begin(), idx.end(),
            [&ang](int a, int b) { return ang[a] < ang[b]; });
  return idx;
}

cv::Mat colorizeLabels4(const cv::Mat &labels_i32) {
  if (labels_i32.empty() || labels_i32.type() != CV_32S) {
    return cv::Mat();
  }
  cv::Mat out(labels_i32.rows, labels_i32.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  const cv::Vec3b colors[4] = {
      cv::Vec3b(255, 0, 0),  // blue
      cv::Vec3b(0, 255, 0),  // green
      cv::Vec3b(0, 0, 255),  // red
      cv::Vec3b(255, 255, 0) // cyan
  };

  for (int y = 0; y < labels_i32.rows; ++y) {
    const int *row = labels_i32.ptr<int>(y);
    cv::Vec3b *orow = out.ptr<cv::Vec3b>(y);
    for (int x = 0; x < labels_i32.cols; ++x) {
      const int v = row[x];
      if (v >= 0 && v < 4) {
        orow[x] = colors[v];
      } else {
        // background
        orow[x] = cv::Vec3b(0, 0, 0);
      }
    }
  }
  return out;
}

CornerInitResult initializeCornerFromIwePatch(const cv::Mat &iwe_f32,
                                              const CornerInitOptions &opt,
                                              float seed_thr_override,
                                              float tau_thr_override) {
  CornerInitResult best;
  best.success = false;

  if (iwe_f32.empty()) {
    return best;
  }
  if (iwe_f32.type() != CV_32F) {
    throw std::runtime_error(
        "initializeCornerFromIwePatch: iwe must be CV_32F");
  }

  const int max_iters = std::max(1, opt.max_grow_iters);
  GrowResult grow = growFromPerimeterPeaks(
      iwe_f32, opt.tau, opt.tau_percentile, opt.seed_thr, opt.seed_percentile,
      seed_thr_override, tau_thr_override, max_iters);
  best.seeds_u8 = grow.seeds_u8.clone();
  best.mask_u8 = grow.mask_u8.clone();
  best.depth_i32 = grow.depth_i32.clone();

  struct Candidate {
    int depth = -1;
    int area_sum = -1;
    cv::Mat mask_u8;
    cv::Mat labels_i32;
    cv::Mat depth_i32;
    cv::Mat centroids;
    std::array<int, 4> orig_lbl{};
  };

  Candidate best_strict;
  Candidate best_loose;

  for (int t = 1; t <= max_iters; ++t) {
    // Build mask for depth <= t
    cv::Mat mask(iwe_f32.rows, iwe_f32.cols, CV_8U, cv::Scalar(0));
    for (int y = 0; y < grow.depth_i32.rows; ++y) {
      const int *dr = grow.depth_i32.ptr<int>(y);
      uint8_t *mr = mask.ptr<uint8_t>(y);
      for (int x = 0; x < grow.depth_i32.cols; ++x) {
        if (dr[x] >= 0 && dr[x] <= t) {
          mr[x] = 255;
        }
      }
    }

    cv::Mat labels, stats, centroids;
    const int nlabels = cv::connectedComponentsWithStats(mask, labels, stats,
                                                         centroids, 8, CV_32S);
    const int fg = nlabels - 1;
    if (fg != 4) {
      continue;
    }

    std::array<int, 4> areas{};
    std::array<int, 4> orig_lbl{};
    int count = 0;
    bool strict_ok = true;
    for (int lbl = 1; lbl < nlabels; ++lbl) {
      const int area = stats.at<int>(lbl, cv::CC_STAT_AREA);
      if (count < 4) {
        areas[count] = area;
        orig_lbl[count] = lbl;
        count++;
      }
      if (area < opt.min_component_area) {
        strict_ok = false;
      }
    }
    if (count != 4) {
      continue;
    }

    const int area_sum = areas[0] + areas[1] + areas[2] + areas[3];
    Candidate cand;
    cand.depth = t;
    cand.area_sum = area_sum;
    cand.mask_u8 = mask;
    cand.depth_i32 = grow.depth_i32.clone();
    cand.centroids = centroids;
    cand.orig_lbl = orig_lbl;

    // Remap labels to 0..3
    cv::Mat remapped(labels.rows, labels.cols, CV_32S, cv::Scalar(-1));
    for (int y = 0; y < labels.rows; ++y) {
      const int *lr = labels.ptr<int>(y);
      int *rr = remapped.ptr<int>(y);
      for (int x = 0; x < labels.cols; ++x) {
        const int v = lr[x];
        if (v == 0) {
          rr[x] = -1;
          continue;
        }
        int rid = -1;
        for (int i = 0; i < 4; ++i) {
          if (v == orig_lbl[i]) {
            rid = i;
            break;
          }
        }
        rr[x] = rid;
      }
    }
    cand.labels_i32 = remapped;

    if (strict_ok) {
      const bool cand_better =
          (cand.depth > best_strict.depth) ||
          (cand.depth == best_strict.depth &&
           cand.area_sum > best_strict.area_sum);
      if (best_strict.depth < 0 || cand_better) {
        best_strict = cand;
      }
    } else {
      const bool cand_better =
          (cand.depth > best_loose.depth) ||
          (cand.depth == best_loose.depth &&
           cand.area_sum > best_loose.area_sum);
      if (best_loose.depth < 0 || cand_better) {
        best_loose = cand;
      }
    }
  }

  const bool use_strict = best_strict.depth >= 0;
  const Candidate *pick = use_strict ? &best_strict : &best_loose;
  if (pick->depth < 0) {
    return best;
  }

  // Centroids for CCW ordering
  std::array<cv::Point2f, 4> c{};
  for (int i = 0; i < 4; ++i) {
    const int lbl = pick->orig_lbl[i];
    c[i] = cv::Point2f(static_cast<float>(pick->centroids.at<double>(lbl, 0)),
                       static_cast<float>(pick->centroids.at<double>(lbl, 1)));
  }

  const auto order = ccwOrderFromCentroids(c);
  const int bit0_a = order[0];
  const int bit0_b = order[2];
  const int bit1_a = order[1];
  const int bit1_b = order[3];

  std::vector<cv::Point> pts0;
  std::vector<cv::Point> pts1;
  for (int y = 0; y < pick->labels_i32.rows; ++y) {
    const int *row = pick->labels_i32.ptr<int>(y);
    for (int x = 0; x < pick->labels_i32.cols; ++x) {
      const int v = row[x];
      if (v == bit0_a || v == bit0_b) {
        pts0.emplace_back(x, y);
      } else if (v == bit1_a || v == bit1_b) {
        pts1.emplace_back(x, y);
      }
    }
  }
  if (pts0.size() < 2 || pts1.size() < 2) {
    return best;
  }

  Line2D line0 = fitLineTLS2D(pts0, iwe_f32, opt.weighted_line_fit);
  Line2D line1 = fitLineTLS2D(pts1, iwe_f32, opt.weighted_line_fit);

  cv::Point2f corner;
  if (!intersectLines(line0, line1, corner)) {
    return best;
  }
  if (corner.x < 0.0f || corner.y < 0.0f ||
      corner.x > static_cast<float>(iwe_f32.cols - 1) ||
      corner.y > static_cast<float>(iwe_f32.rows - 1)) {
    return best;
  }

  best.success = true;
  best.init_corner_xy = corner;
  best.corner_xy = corner;
  best.line0 = line0;
  best.line1 = line1;
  best.chosen_tau = opt.tau;
  best.num_components = 4;
  best.mask_u8 = pick->mask_u8.clone();
  best.labels_i32 = pick->labels_i32.clone();
  best.labels_bgr_u8 = colorizeLabels4(best.labels_i32);
  best.depth_i32 = pick->depth_i32.clone();

  return best;
}

} // namespace corner2tag::core
