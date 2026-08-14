#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace corner2tag::core {

struct PatchPointsOptions {
  // circle radius for perimeter test
  int radius = 3;

  // sign thresholding:
  // sign_eps = max(sign_eps_min, sign_eps_ratio * abs_percentile(|pIWE|))
  float sign_eps_ratio = 0.2f;
  float sign_eps_min = 1e-6f;

  // percentile for |pIWE| to set dynamic scale (robust to outliers)
  float abs_percentile = 99.0f;

  // optional: require |pIWE(center)| >= center_abs_min (default 0 means "no
  // gate")
  float center_abs_min = 0.0f;

  // Corner-like sign changes around the perimeter.
  // Edges typically have ~2 sign changes; corners have >= 4.
  int min_sign_changes = 2;

  // Perimeter "alpha": if |pIWE| < boundary_abs_min, treat as 0 in the
  // circular boundary vector (ignored for sign changes).
  // If boundary_abs_ratio > 0, alpha = max(boundary_abs_min,
  //                                     boundary_abs_ratio * scale(|pIWE|)).
  // If boundary_abs_ratio <= 0 and boundary_abs_min <= 0, fall back to
  // sign_eps.
  float boundary_abs_ratio = 0.0f;
  float boundary_abs_min = 0.0f;
};

struct PatchClusterOptions {
  // DBSCAN eps equivalent (pixels). Implemented as dilation radius.
  int eps = 0;

  // DBSCAN min_samples equivalent (approx).
  // We enforce by dropping CCs whose (dilated) area < min_component_area.
  int min_component_area = 1;
};

struct PatchBox {
  int x0 = 0; // inclusive
  int x1 = 0; // exclusive
  int y0 = 0; // inclusive
  int y1 = 0; // exclusive
  int label = -1;
};

class PatchExtractor {
public:
  PatchExtractor(PatchPointsOptions pp_opt, PatchClusterOptions cl_opt);

  // Step 1: compute patch_points (centers) from pIWE
  std::vector<cv::Point> computePatchPoints(const cv::Mat &piwe_f32) const;

  // Step 2: cluster patch_points -> square bboxes
  std::vector<PatchBox>
  clusterToBoxes(const std::vector<cv::Point> &patch_points, int width,
                 int height) const;

  // Convenience: compute points then boxes
  std::vector<PatchBox> extract(const cv::Mat &piwe_f32) const;

  // Visualization helpers
  static void drawPatchPoints(cv::Mat &bgr_u8,
                              const std::vector<cv::Point> &pts,
                              float zoom_factor = 1.0f,
                              const cv::Scalar &color = cv::Scalar(0, 0, 255),
                              int radius = 1, int thickness = -1);

  static void drawPatchBoxes(cv::Mat &bgr_u8,
                             const std::vector<PatchBox> &boxes,
                             float zoom_factor = 1.0f,
                             const cv::Scalar &color = cv::Scalar(0, 255, 0),
                             int thickness = 2);

private:
  PatchPointsOptions pp_;
  PatchClusterOptions cl_;

  std::vector<cv::Point> circle_offsets_;
  std::vector<cv::Point> circle_offsets_ordered_;

  static std::vector<cv::Point> buildCirclePerimeterOffsets(int radius);

  static PatchBox makeSquareAndClipFromRect(int x, int y, int w, int h, int W,
                                            int H, int label);

  static float percentileAbs(const cv::Mat &piwe_f32, float pct);

  bool perimeterHasPosNeg(const cv::Mat &piwe_f32, int x, int y,
                          float alpha) const;
  int perimeterSignChanges(const cv::Mat &piwe_f32, int x, int y,
                           float alpha) const;
};

} // namespace corner2tag::core
