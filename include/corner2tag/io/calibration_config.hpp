#pragma once

#include <cstdint>
#include <string>

namespace corner2tag::io {

struct CalibrationConfig {
  // input/output
  std::string h5_path;
  std::string out_dir;

  // sensor size
  int width = -1;
  int height = -1;

  // Fixed temporal windowing (microseconds), relative to the first event.
  uint64_t fixed_window_us = 0;

  int expected_corners = -1; // if < 0, auto = board_w * board_h
  int board_w = -1;
  int board_h = -1;
  float square_size = 0.0f;

  // visualization
  float viz_zoom = 2.0f; // visualization only; calibration uses original size

  // patch extraction
  int pp_radius = 5;

  // cm iwe
  bool use_variance = true; // true=L2(variance), false=IWE
  float cm_iwe_sigma = 1.0f;
  float cm_iwe_cutoff_factor = 3.0f;
  int cm_iwe_patch_radius_override = -1;

  // tracker
  int tracker_num_threads = 4;
  int tracker_max_iterations = 200;

  // checkerboard validity
  float checkerboard_tor_spacing = 0.25f;
  float checkerboard_tor_orth = 0.2f;

  // corner refinement
  int corner_refine_max_iter = 200;
  float corner_refine_lr = 0.25f;
  float corner_refine_strip_half_width0 = 0.5f;
  float corner_refine_strip_half_width1 = 0.5f;

  // calibration
  int calib_max_iter = 50;
  bool calib_fix_k3plus = true;
  bool calib_use_intrinsic_guess = true;
};

bool loadCalibrationConfig(const std::string &path, CalibrationConfig &cfg,
                         std::string *err = nullptr);

} // namespace corner2tag::io
