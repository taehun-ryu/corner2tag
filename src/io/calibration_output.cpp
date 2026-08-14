#include "corner2tag/io/calibration_output.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <opencv2/core.hpp>

namespace corner2tag::io {

static std::string formatNowTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf {};
#if defined(_WIN32)
  localtime_s(&tm_buf, &now_tt);
#else
  localtime_r(&now_tt, &tm_buf);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
  return oss.str();
}

std::string
prepareCalibrationRunOutputDir(const std::string &base_out_dir,
                               const DetectionOutputLayout &layout) {
  if (base_out_dir.empty()) {
    return "";
  }

  const std::filesystem::path base_path(base_out_dir);
  std::filesystem::create_directories(base_path);

  const std::string timestamp = formatNowTimestamp();
  std::filesystem::path run_path = base_path / ("run_" + timestamp);
  int suffix = 1;
  while (std::filesystem::exists(run_path)) {
    run_path =
        base_path / ("run_" + timestamp + "_" + std::to_string(suffix++));
  }

  ensureDetectionOutputDirs(run_path.string(), layout);
  return run_path.string();
}

static void writeMatYaml(std::ofstream &ofs, const std::string &key,
                         const cv::Mat &m) {
  ofs << key << ":\n";
  if (m.empty()) {
    ofs << "  []\n";
    return;
  }
  for (int r = 0; r < m.rows; ++r) {
    ofs << "  - [";
    for (int c = 0; c < m.cols; ++c) {
      const double v = m.at<double>(r, c);
      ofs << v;
      if (c + 1 < m.cols)
        ofs << ", ";
    }
    ofs << "]\n";
  }
}

static void writeVecYaml(std::ofstream &ofs, const std::string &key,
                         const cv::Mat &m) {
  ofs << key << ": [";
  if (!m.empty()) {
    const size_t total = m.total();
    for (size_t i = 0; i < total; ++i) {
      ofs << m.at<double>(static_cast<int>(i));
      if (i + 1 < total)
        ofs << ", ";
    }
  }
  ofs << "]\n";
}

static void writeCalibrationYamlCore(std::ofstream &ofs,
                                     const cv::Mat &camera_matrix,
                                     const cv::Mat &dist_coeffs,
                                     double reprojection_error) {
  ofs << "reprojection_error: " << reprojection_error << "\n";

  cv::Mat Kd, distd;
  camera_matrix.convertTo(Kd, CV_64F);
  dist_coeffs.convertTo(distd, CV_64F);
  writeMatYaml(ofs, "camera_matrix", Kd);
  writeVecYaml(ofs, "dist_coeffs", distd.reshape(1, 1));
}

void saveCheckerboardCalibrationYaml(const std::string &out_dir,
                                     size_t used_windows,
                                     size_t total_windows, int board_rows,
                                     int board_cols, float square_size,
                                     const cv::Mat &camera_matrix,
                                     const cv::Mat &dist_coeffs,
                                     double reprojection_error) {
  if (out_dir.empty()) {
    return;
  }
  const std::string path =
      (std::filesystem::path(out_dir) / "calibration.yaml").string();
  std::ofstream ofs(path);
  if (!ofs) {
    return;
  }

  ofs << "used_windows: " << used_windows << "\n";
  ofs << "total_windows: " << total_windows << "\n";
  ofs << "checkerboard:\n";
  ofs << "  rows: " << board_rows << "\n";
  ofs << "  cols: " << board_cols << "\n";
  ofs << "  square_size: " << square_size << "\n";
  writeCalibrationYamlCore(ofs, camera_matrix, dist_coeffs, reprojection_error);
}

} // namespace corner2tag::io
