#include "corner2tag/core/calib/checkerboard_calibration.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>

namespace corner2tag::core {

double
computeReprojectionError(const std::vector<std::vector<cv::Point3f>> &objpoints,
                         const std::vector<std::vector<cv::Point2f>> &imgpoints,
                         const cv::Mat &K, const cv::Mat &dist,
                         const std::vector<cv::Mat> &rvecs,
                         const std::vector<cv::Mat> &tvecs) {
  double total_err = 0.0;
  size_t total_pts = 0;
  std::vector<cv::Point2f> proj;
  for (size_t i = 0; i < objpoints.size(); ++i) {
    cv::projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist, proj);
    double err = 0.0;
    for (size_t j = 0; j < proj.size(); ++j) {
      const cv::Point2f d = proj[j] - imgpoints[i][j];
      err += std::sqrt(d.dot(d));
    }
    total_err += err;
    total_pts += proj.size();
  }
  return (total_pts > 0) ? (total_err / total_pts) : 0.0;
}

static int buildFlags(const CalibrationOptions &opt) {
  int flags = 0;
  if (opt.fix_k3plus) {
    flags |= cv::CALIB_FIX_K3;
    flags |= cv::CALIB_FIX_K4;
    flags |= cv::CALIB_FIX_K5;
    flags |= cv::CALIB_FIX_K6;
  }
  return flags;
}

CalibrationResult
calibrateCheckerboard(const std::vector<std::vector<cv::Point3f>> &objpoints,
                      const std::vector<std::vector<cv::Point2f>> &imgpoints,
                      const cv::Size &image_size,
                      const CalibrationOptions &opt) {
  CalibrationResult out;
  if (objpoints.empty() || imgpoints.empty()) {
    return out;
  }
  const int flags_base = buildFlags(opt);
  const cv::TermCriteria criteria(cv::TermCriteria::EPS +
                                      cv::TermCriteria::MAX_ITER,
                                  std::max(1, opt.max_iter), opt.eps);

  cv::Mat K, dist;
  std::vector<cv::Mat> rvecs, tvecs;

  double rms = cv::calibrateCamera(objpoints, imgpoints, image_size, K, dist,
                                   rvecs, tvecs, flags_base, criteria);

  if (opt.use_intrinsic_guess) {
    const int flags2 = flags_base | cv::CALIB_USE_INTRINSIC_GUESS;
    rms = cv::calibrateCamera(objpoints, imgpoints, image_size, K, dist, rvecs,
                              tvecs, flags2, criteria);
  }

  out.success = std::isfinite(rms) && rms > 0.0;
  out.camera_matrix = K;
  out.dist_coeffs = dist;
  out.rvecs = rvecs;
  out.tvecs = tvecs;
  out.reprojection_error =
      computeReprojectionError(objpoints, imgpoints, K, dist, rvecs, tvecs);
  return out;
}

} // namespace corner2tag::core
