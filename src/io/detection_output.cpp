#include "corner2tag/io/detection_output.hpp"

#include <filesystem>

#include <opencv2/imgcodecs.hpp>

namespace corner2tag::io {

static std::string zeroPad(size_t v, int width) {
  std::string s = std::to_string(v);
  if (static_cast<int>(s.size()) >= width) {
    return s;
  }
  return std::string(static_cast<size_t>(width - s.size()), '0') + s;
}

void ensureDetectionOutputDirs(const std::string &out_dir,
                               const DetectionOutputLayout &layout) {
  if (out_dir.empty()) {
    return;
  }
  if (layout.detection_dir.empty()) {
    return;
  }
  std::filesystem::create_directories(out_dir);
  std::filesystem::create_directories(std::filesystem::path(out_dir) / "raw");
  std::filesystem::create_directories(std::filesystem::path(out_dir) / "iwe");
  std::filesystem::create_directories(std::filesystem::path(out_dir) /
                                      layout.detection_dir);
}

void saveDetectionOutputs(const std::string &out_dir, size_t window_idx,
                          const cv::Mat &raw_vis, const cv::Mat &iwe_vis,
                          const cv::Mat &detection_vis,
                          const DetectionOutputLayout &layout) {
  if (out_dir.empty()) {
    return;
  }
  if (layout.detection_dir.empty() || layout.detection_prefix.empty()) {
    return;
  }
  const std::string idx_str = zeroPad(window_idx, 6);
  const std::string raw_path = (std::filesystem::path(out_dir) / "raw" /
                                ("raw_events_" + idx_str + ".png"))
                                   .string();
  const std::string iwe_path =
      (std::filesystem::path(out_dir) / "iwe" / ("iwe_" + idx_str + ".png"))
          .string();
  const std::string detection_path =
      (std::filesystem::path(out_dir) / layout.detection_dir /
       (layout.detection_prefix + "_" + idx_str + ".png"))
          .string();

  if (!raw_vis.empty()) {
    cv::imwrite(raw_path, raw_vis);
  }
  if (!iwe_vis.empty()) {
    cv::imwrite(iwe_path, iwe_vis);
  }
  if (!detection_vis.empty()) {
    cv::imwrite(detection_path, detection_vis);
  }
}

} // namespace corner2tag::io
