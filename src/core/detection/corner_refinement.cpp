#include "corner2tag/core/detection/corner_refinement.hpp"

#include <algorithm>
#include <cmath>

namespace corner2tag::core {

static inline float clampf(float v, float lo, float hi) {
  if (v < lo)
    return lo;
  if (v > hi)
    return hi;
  return v;
}

static cv::Point2f projectToStrips(const cv::Point2f &p_in, const Line2D &l0,
                                   float d0, const Line2D &l1, float d1, int W,
                                   int H) {
  cv::Point2f p = p_in;

  auto clampToBand = [&p](const Line2D &l, float band) {
    const float s = l.n.x * p.x + l.n.y * p.y + l.d;
    if (std::abs(s) > band) {
      const float target = (s > 0.0f) ? band : -band;
      const float delta = s - target;
      p.x -= delta * l.n.x;
      p.y -= delta * l.n.y;
    }
  };

  clampToBand(l0, d0);
  clampToBand(l1, d1);
  clampToBand(l0, d0);

  p.x = clampf(p.x, 0.0f, static_cast<float>(W - 1));
  p.y = clampf(p.y, 0.0f, static_cast<float>(H - 1));
  return p;
}

static std::pair<float, cv::Point2f> bilinearValueAndGrad(const cv::Mat &Z,
                                                          float x, float y) {
  const int W = Z.cols;
  const int H = Z.rows;

  const float x0f = clampf(x, 0.0f, static_cast<float>(W - 1));
  const float y0f = clampf(y, 0.0f, static_cast<float>(H - 1));

  const int x0 = static_cast<int>(std::floor(x0f));
  const int y0 = static_cast<int>(std::floor(y0f));
  const int x1 = std::min(x0 + 1, W - 1);
  const int y1 = std::min(y0 + 1, H - 1);

  const float dx = x0f - static_cast<float>(x0);
  const float dy = y0f - static_cast<float>(y0);

  const float v00 = Z.at<float>(y0, x0);
  const float v10 = Z.at<float>(y0, x1);
  const float v01 = Z.at<float>(y1, x0);
  const float v11 = Z.at<float>(y1, x1);

  const float v0 = (1.0f - dx) * v00 + dx * v10;
  const float v1 = (1.0f - dx) * v01 + dx * v11;
  const float v = (1.0f - dy) * v0 + dy * v1;

  const float gx = (1.0f - dy) * (v10 - v00) + dy * (v11 - v01);
  const float gy = (1.0f - dx) * (v01 - v00) + dx * (v11 - v10);

  return {v, cv::Point2f(gx, gy)};
}

CornerRefineResult refineCornerInIwePatch(const cv::Mat &iwe_f32,
                                          const cv::Point2f &init_xy,
                                          const Line2D &line0,
                                          const Line2D &line1,
                                          const CornerRefineOptions &opt) {
  CornerRefineResult out;
  if (!opt.enable || iwe_f32.empty() || iwe_f32.type() != CV_32F) {
    return out;
  }

  cv::Point2f p =
      projectToStrips(init_xy, line0, opt.strip_half_width0, line1,
                      opt.strip_half_width1, iwe_f32.cols, iwe_f32.rows);

  if (opt.keep_path) {
    out.path.push_back(p);
  }

  for (int it = 0; it < opt.max_iter; ++it) {
    auto vg = bilinearValueAndGrad(iwe_f32, p.x, p.y);
    float f = vg.first;
    cv::Point2f g = vg.second;
    const float gnorm = std::sqrt(g.x * g.x + g.y * g.y);
    if (gnorm < opt.gtol) {
      out.f = f;
      break;
    }

    float step = opt.lr;
    float f_now = f;
    bool stepped = false;
    while (step >= opt.min_step) {
      cv::Point2f p_try(p.x - step * g.x, p.y - step * g.y);
      p_try =
          projectToStrips(p_try, line0, opt.strip_half_width0, line1,
                          opt.strip_half_width1, iwe_f32.cols, iwe_f32.rows);

      auto vg_try = bilinearValueAndGrad(iwe_f32, p_try.x, p_try.y);
      const float f_try = vg_try.first;
      if (f_try <= f_now - opt.armijo_c * step * gnorm * gnorm) {
        p = p_try;
        out.f = f_try;
        stepped = true;
        break;
      }
      step *= 0.5f;
    }
    if (!stepped) {
      break;
    }
    if (opt.keep_path) {
      out.path.push_back(p);
    }
  }

  out.refined_xy = p;
  out.success = true;
  return out;
}

} // namespace corner2tag::core
