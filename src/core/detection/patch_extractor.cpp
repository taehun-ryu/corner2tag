#include "corner2tag/core/detection/patch_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <opencv2/imgproc.hpp> // dilate, connectedComponentsWithStats, circle, rectangle

namespace corner2tag::core {

static inline uint64_t packXY(int x, int y) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(y));
}

struct AnglePoint {
  double angle = 0.0;
  cv::Point point;
};

static bool compareAnglePoint(const AnglePoint &a, const AnglePoint &b) {
  return a.angle < b.angle;
}

PatchExtractor::PatchExtractor(PatchPointsOptions pp_opt,
                               PatchClusterOptions cl_opt)
    : pp_(pp_opt), cl_(cl_opt) {
  if (pp_.radius <= 0) {
    throw std::runtime_error("PatchExtractor: radius must be > 0");
  }
  if (pp_.abs_percentile <= 0.0f || pp_.abs_percentile > 100.0f) {
    throw std::runtime_error(
        "PatchExtractor: abs_percentile must be in (0,100]");
  }
  if (pp_.sign_eps_ratio < 0.0f) {
    throw std::runtime_error("PatchExtractor: sign_eps_ratio must be >= 0");
  }
  if (pp_.sign_eps_min < 0.0f) {
    throw std::runtime_error("PatchExtractor: sign_eps_min must be >= 0");
  }
  if (cl_.eps < 0) {
    throw std::runtime_error("PatchExtractor: cluster eps must be >= 0");
  }
  if (cl_.min_component_area < 1) {
    throw std::runtime_error("PatchExtractor: min_component_area must be >= 1");
  }

  circle_offsets_ = buildCirclePerimeterOffsets(pp_.radius);
  if (circle_offsets_.empty()) {
    throw std::runtime_error("PatchExtractor: circle perimeter offsets empty");
  }
  std::vector<AnglePoint> angle_points;
  angle_points.reserve(circle_offsets_.size());
  for (const cv::Point &p : circle_offsets_) {
    AnglePoint ap;
    ap.angle = std::atan2(static_cast<double>(p.y), static_cast<double>(p.x));
    ap.point = p;
    angle_points.push_back(ap);
  }
  std::sort(angle_points.begin(), angle_points.end(), compareAnglePoint);

  circle_offsets_ordered_.clear();
  circle_offsets_ordered_.reserve(angle_points.size());
  for (const AnglePoint &ap : angle_points) {
    circle_offsets_ordered_.push_back(ap.point);
  }
}

std::vector<cv::Point> PatchExtractor::buildCirclePerimeterOffsets(int radius) {
  // Midpoint circle algorithm, 8-way symmetry. Outputs integer (dx,dy).
  std::unordered_set<uint64_t> uniq;
  std::vector<cv::Point> pts;
  pts.reserve(static_cast<size_t>(8 * radius + 8));

  int x = radius;
  int y = 0;
  int err = 1 - x;

  auto add_sym = [&uniq, &pts](int xx, int yy) {
    const int xs[8] = {xx, yy, -yy, -xx, -xx, -yy, yy, xx};
    const int ys[8] = {yy, xx, xx, yy, -yy, -xx, -xx, -yy};
    for (int k = 0; k < 8; ++k) {
      const int dx = xs[k];
      const int dy = ys[k];
      const uint64_t key = packXY(dx, dy);
      if (uniq.insert(key).second) {
        pts.emplace_back(dx, dy);
      }
    }
  };

  while (x >= y) {
    add_sym(x, y);
    ++y;
    if (err < 0) {
      err += 2 * y + 1;
    } else {
      --x;
      err += 2 * (y - x) + 1;
    }
  }

  return pts;
}

float PatchExtractor::percentileAbs(const cv::Mat &piwe_f32, float pct) {
  // Robust scale estimate: percentile of |piwe|.
  // NOTE: copying all pixels is okay at 346x260; keep it simple and correct.
  std::vector<float> v;
  v.reserve(static_cast<size_t>(piwe_f32.rows * piwe_f32.cols));

  for (int y = 0; y < piwe_f32.rows; ++y) {
    const float *row = piwe_f32.ptr<float>(y);
    for (int x = 0; x < piwe_f32.cols; ++x) {
      v.push_back(std::abs(row[x]));
    }
  }
  if (v.empty()) {
    return 0.0f;
  }

  const float clamped = std::min(100.0f, std::max(0.0f, pct));
  const size_t k = static_cast<size_t>(
      std::round((clamped / 100.0f) * static_cast<float>(v.size() - 1)));

  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k),
                   v.end());
  return v[k];
}

bool PatchExtractor::perimeterHasPosNeg(const cv::Mat &piwe_f32, int x, int y,
                                        float alpha) const {
  const int H = piwe_f32.rows;
  const int W = piwe_f32.cols;

  int pos = 0;
  int neg = 0;

  for (const auto &off : circle_offsets_) {
    const int xx = x + off.x;
    const int yy = y + off.y;
    if (xx < 0 || xx >= W || yy < 0 || yy >= H) {
      continue;
    }
    const float v = piwe_f32.at<float>(yy, xx);
    if (v > alpha) {
      pos = 1;
    } else if (v < -alpha) {
      neg = 1;
    }
    if (pos && neg) {
      return true;
    }
  }
  return false;
}

int PatchExtractor::perimeterSignChanges(const cv::Mat &piwe_f32, int x, int y,
                                         float alpha) const {
  const int H = piwe_f32.rows;
  const int W = piwe_f32.cols;

  std::vector<int> signs;
  signs.reserve(circle_offsets_ordered_.size());

  for (const auto &off : circle_offsets_ordered_) {
    const int xx = x + off.x;
    const int yy = y + off.y;
    if (xx < 0 || xx >= W || yy < 0 || yy >= H) {
      continue;
    }
    const float v = piwe_f32.at<float>(yy, xx);
    if (v > alpha) {
      signs.push_back(1);
    } else if (v < -alpha) {
      signs.push_back(-1);
    } else {
      signs.push_back(0);
    }
  }

  if (signs.empty()) {
    return 0;
  }

  // compress zeros by skipping them in transitions
  int changes = 0;
  int last = 0;
  int first = 0;

  for (size_t i = 0; i < signs.size(); ++i) {
    const int s = signs[i];
    if (s == 0) {
      continue;
    }
    if (first == 0) {
      first = s;
      last = s;
      continue;
    }
    if (s != last) {
      changes++;
      last = s;
    }
  }

  if (first != 0 && last != 0 && first != last) {
    changes++;
  }

  return changes;
}

std::vector<cv::Point>
PatchExtractor::computePatchPoints(const cv::Mat &piwe_f32) const {
  if (piwe_f32.empty()) {
    return {};
  }
  if (piwe_f32.type() != CV_32F) {
    throw std::runtime_error("computePatchPoints: piwe must be CV_32F");
  }

  const int H = piwe_f32.rows;
  const int W = piwe_f32.cols;

  // Dynamic sign_eps from robust scale (percentile of abs)
  const float scale = percentileAbs(piwe_f32, pp_.abs_percentile);
  const float sign_eps = std::max(pp_.sign_eps_min, pp_.sign_eps_ratio * scale);
  float boundary_alpha = sign_eps;
  if (pp_.boundary_abs_ratio > 0.0f || pp_.boundary_abs_min > 0.0f) {
    const float a = (pp_.boundary_abs_ratio > 0.0f)
                        ? (pp_.boundary_abs_ratio * scale)
                        : 0.0f;
    boundary_alpha = std::max(pp_.boundary_abs_min, a);
    if (boundary_alpha <= 0.0f) {
      boundary_alpha = sign_eps;
    }
  }

  std::vector<cv::Point> pts;
  pts.reserve(static_cast<size_t>(H * W / 20));

  for (int y = 0; y < H; ++y) {
    const float *row = piwe_f32.ptr<float>(y);
    for (int x = 0; x < W; ++x) {
      // optional center gate (default 0 => no gate)
      if (pp_.center_abs_min > 0.0f) {
        if (std::abs(row[x]) < pp_.center_abs_min) {
          continue;
        }
      }

      if (!perimeterHasPosNeg(piwe_f32, x, y, boundary_alpha)) {
        continue;
      }
      if (pp_.min_sign_changes > 0) {
        const int changes =
            perimeterSignChanges(piwe_f32, x, y, boundary_alpha);
        if (changes < pp_.min_sign_changes) {
          continue;
        }
      }

      pts.emplace_back(x, y); // cv::Point is (x,y)
    }
  }

  return pts;
}

PatchBox PatchExtractor::makeSquareAndClipFromRect(int x, int y, int w, int h,
                                                   int W, int H, int label) {
  const int side = std::max(w, h);
  const float cx = static_cast<float>(x) + 0.5f * static_cast<float>(w);
  const float cy = static_cast<float>(y) + 0.5f * static_cast<float>(h);

  int x0 = static_cast<int>(std::floor(cx - 0.5f * static_cast<float>(side)));
  int x1 = static_cast<int>(std::ceil(cx + 0.5f * static_cast<float>(side)));
  int y0 = static_cast<int>(std::floor(cy - 0.5f * static_cast<float>(side)));
  int y1 = static_cast<int>(std::ceil(cy + 0.5f * static_cast<float>(side)));

  // make end-exclusive
  x1 += 1;
  y1 += 1;

  x0 = std::max(0, x0);
  y0 = std::max(0, y0);
  x1 = std::min(W, x1);
  y1 = std::min(H, y1);

  PatchBox b;
  b.x0 = x0;
  b.x1 = x1;
  b.y0 = y0;
  b.y1 = y1;
  b.label = label;
  return b;
}

std::vector<PatchBox>
PatchExtractor::clusterToBoxes(const std::vector<cv::Point> &patch_points,
                               int width, int height) const {
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("clusterToBoxes: invalid image size");
  }
  if (patch_points.empty()) {
    return {};
  }

  cv::Mat mask = cv::Mat::zeros(height, width, CV_8U);
  for (const auto &p : patch_points) {
    if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height) {
      continue;
    }
    mask.at<uint8_t>(p.y, p.x) = 255;
  }

  cv::Mat dilated = mask;
  if (cl_.eps > 0) {
    const int k = 2 * cl_.eps + 1;
    cv::Mat kse = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
    cv::dilate(mask, dilated, kse);
  }

  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  const int nlabels = cv::connectedComponentsWithStats(dilated, labels, stats,
                                                       centroids, 4, CV_32S);

  std::vector<PatchBox> boxes;
  boxes.reserve(static_cast<size_t>(std::max(0, nlabels - 1)));

  for (int lbl = 1; lbl < nlabels; ++lbl) {
    const int area = stats.at<int>(lbl, cv::CC_STAT_AREA);
    if (area < cl_.min_component_area) {
      continue;
    }

    const int x = stats.at<int>(lbl, cv::CC_STAT_LEFT);
    const int y = stats.at<int>(lbl, cv::CC_STAT_TOP);
    const int w = stats.at<int>(lbl, cv::CC_STAT_WIDTH);
    const int h = stats.at<int>(lbl, cv::CC_STAT_HEIGHT);

    PatchBox b = makeSquareAndClipFromRect(x, y, w, h, width, height, lbl);
    if (b.x1 <= b.x0 || b.y1 <= b.y0) {
      continue;
    }
    boxes.push_back(b);
  }

  return boxes;
}

std::vector<PatchBox> PatchExtractor::extract(const cv::Mat &piwe_f32) const {
  const auto pts = computePatchPoints(piwe_f32);
  return clusterToBoxes(pts, piwe_f32.cols, piwe_f32.rows);
}

void PatchExtractor::drawPatchPoints(cv::Mat &bgr_u8,
                                     const std::vector<cv::Point> &pts,
                                     float zoom_factor, const cv::Scalar &color,
                                     int radius, int thickness) {
  if (bgr_u8.empty()) {
    return;
  }
  float z = zoom_factor;
  if (z <= 0.0f) {
    z = 1.0f;
  }

  for (const auto &p : pts) {
    const cv::Point c(static_cast<int>(std::lround(p.x * z)),
                      static_cast<int>(std::lround(p.y * z)));
    cv::circle(bgr_u8, c, radius, color, thickness, cv::LINE_AA);
  }
}

void PatchExtractor::drawPatchBoxes(cv::Mat &bgr_u8,
                                    const std::vector<PatchBox> &boxes,
                                    float zoom_factor, const cv::Scalar &color,
                                    int thickness) {
  if (bgr_u8.empty()) {
    return;
  }
  float z = zoom_factor;
  if (z <= 0.0f) {
    z = 1.0f;
  }

  for (const auto &b : boxes) {
    const int x0 = static_cast<int>(std::lround(b.x0 * z));
    const int y0 = static_cast<int>(std::lround(b.y0 * z));
    const int x1 = static_cast<int>(std::lround(b.x1 * z));
    const int y1 = static_cast<int>(std::lround(b.y1 * z));

    cv::rectangle(bgr_u8, cv::Point(x0, y0), cv::Point(x1 - 1, y1 - 1), color,
                  thickness, cv::LINE_AA);
  }
}

} // namespace corner2tag::core
