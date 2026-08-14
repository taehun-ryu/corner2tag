#include "corner2tag/io/calibration_config.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace corner2tag::io {

static inline std::string trim(const std::string &s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) {
    return "";
  }
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

static bool hasKey(const std::unordered_map<std::string, std::string> &kv,
                   const char *k) {
  return kv.find(k) != kv.end();
}

static const std::string &
getValue(const std::unordered_map<std::string, std::string> &kv,
         const char *k) {
  return kv.at(k);
}

bool loadCalibrationConfig(const std::string &path, CalibrationConfig &cfg,
                           std::string *err) {
  std::ifstream ifs(path);
  if (!ifs) {
    if (err) {
      *err = "Failed to open config: " + path;
    }
    return false;
  }

  std::unordered_map<std::string, std::string> kv;
  std::string line;
  while (std::getline(ifs, line)) {
    const size_t hash = line.find('#');
    if (hash != std::string::npos) {
      line = line.substr(0, hash);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    const size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string key = trim(line.substr(0, colon));
    std::string val = trim(line.substr(colon + 1));
    if (!val.empty() && val.front() == '\"' && val.back() == '\"') {
      val = val.substr(1, val.size() - 2);
    }
    kv[key] = val;
  }

  if (hasKey(kv, "h5_path"))
    cfg.h5_path = getValue(kv, "h5_path");
  if (hasKey(kv, "out_dir"))
    cfg.out_dir = getValue(kv, "out_dir");
  if (hasKey(kv, "width"))
    cfg.width = std::stoi(getValue(kv, "width"));
  if (hasKey(kv, "height"))
    cfg.height = std::stoi(getValue(kv, "height"));
  if (hasKey(kv, "fixed_window_us"))
    cfg.fixed_window_us = std::stoull(getValue(kv, "fixed_window_us"));

  if (hasKey(kv, "expected_corners"))
    cfg.expected_corners = std::stoi(getValue(kv, "expected_corners"));
  if (hasKey(kv, "board_w"))
    cfg.board_w = std::stoi(getValue(kv, "board_w"));
  if (hasKey(kv, "board_h"))
    cfg.board_h = std::stoi(getValue(kv, "board_h"));
  if (hasKey(kv, "square_size"))
    cfg.square_size = std::stof(getValue(kv, "square_size"));
  if (hasKey(kv, "viz_zoom"))
    cfg.viz_zoom = std::stof(getValue(kv, "viz_zoom"));
  if (hasKey(kv, "pp_radius"))
    cfg.pp_radius = std::stoi(getValue(kv, "pp_radius"));
  if (hasKey(kv, "use_variance"))
    cfg.use_variance = (getValue(kv, "use_variance") == "true");
  if (hasKey(kv, "cm_iwe.sigma"))
    cfg.cm_iwe_sigma = std::stof(getValue(kv, "cm_iwe.sigma"));
  if (hasKey(kv, "cm_iwe.cutoff_factor"))
    cfg.cm_iwe_cutoff_factor =
        std::stof(getValue(kv, "cm_iwe.cutoff_factor"));
  if (hasKey(kv, "cm_iwe.patch_radius_override"))
    cfg.cm_iwe_patch_radius_override =
        std::stoi(getValue(kv, "cm_iwe.patch_radius_override"));
  if (hasKey(kv, "tracker.num_threads"))
    cfg.tracker_num_threads = std::stoi(getValue(kv, "tracker.num_threads"));
  if (hasKey(kv, "tracker.max_iterations"))
    cfg.tracker_max_iterations =
        std::stoi(getValue(kv, "tracker.max_iterations"));
  if (hasKey(kv, "checkerboard_validity.tor_spacing"))
    cfg.checkerboard_tor_spacing =
        std::stof(getValue(kv, "checkerboard_validity.tor_spacing"));
  if (hasKey(kv, "checkerboard_validity.tor_orth"))
    cfg.checkerboard_tor_orth =
        std::stof(getValue(kv, "checkerboard_validity.tor_orth"));
  if (hasKey(kv, "corner_refine.max_iter"))
    cfg.corner_refine_max_iter =
        std::stoi(getValue(kv, "corner_refine.max_iter"));
  if (hasKey(kv, "corner_refine.lr"))
    cfg.corner_refine_lr = std::stof(getValue(kv, "corner_refine.lr"));
  if (hasKey(kv, "corner_refine.strip_half_width0"))
    cfg.corner_refine_strip_half_width0 =
        std::stof(getValue(kv, "corner_refine.strip_half_width0"));
  if (hasKey(kv, "corner_refine.strip_half_width1"))
    cfg.corner_refine_strip_half_width1 =
        std::stof(getValue(kv, "corner_refine.strip_half_width1"));
  if (hasKey(kv, "calib_max_iter"))
    cfg.calib_max_iter = std::stoi(getValue(kv, "calib_max_iter"));
  if (hasKey(kv, "calib_fix_k3plus"))
    cfg.calib_fix_k3plus = (getValue(kv, "calib_fix_k3plus") == "true");
  if (hasKey(kv, "calib_use_intrinsic_guess"))
    cfg.calib_use_intrinsic_guess =
        (getValue(kv, "calib_use_intrinsic_guess") == "true");

  if (cfg.h5_path.empty()) {
    if (err)
      *err = "h5_path is empty in config";
    return false;
  }
  if (cfg.width <= 0 || cfg.height <= 0) {
    if (err)
      *err = "width/height must be > 0";
    return false;
  }
  if (cfg.fixed_window_us == 0) {
    if (err)
      *err = "fixed_window_us must be > 0";
    return false;
  }

  if (cfg.expected_corners < 0 && cfg.board_w > 0 && cfg.board_h > 0) {
    cfg.expected_corners = cfg.board_w * cfg.board_h;
  }

  return true;
}

} // namespace corner2tag::io
