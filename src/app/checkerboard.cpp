#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "corner2tag/core/calib/checkerboard_calibration.hpp"
#include "corner2tag/core/postprocess/checkerboard_postprocess.hpp"
#include "corner2tag/core/cm/cm_tracker_2d.hpp"
#include "corner2tag/core/detection/corner_init.hpp"
#include "corner2tag/core/detection/corner_refinement.hpp"
#include "corner2tag/core/detection/patch_extractor.hpp"
#include "corner2tag/core/fixed_window.hpp"
#include "corner2tag/core/simple_event.hpp"
#include "corner2tag/io/calibration_config.hpp"
#include "corner2tag/io/detection_output.hpp"
#include "corner2tag/io/calibration_output.hpp"
#include "corner2tag/io/h5_events.hpp"
#include "corner2tag/viz/calibration_report.hpp"
#include "corner2tag/viz/corner_detection_viz.hpp"

namespace {
bool parseArgs(int argc, char **argv, corner2tag::io::CalibrationConfig &cfg) {
  // checkerboard [config_path]
  std::string config_path = "config/checkerboard.yaml";
  if (argc >= 2) {
    config_path = argv[1];
  }
  std::string err;
  if (!corner2tag::io::loadCalibrationConfig(config_path, cfg, &err)) {
    std::cerr << err << "\n";
    return false;
  }
  return true;
}

inline double usToSec(uint64_t t_us) {
  return static_cast<double>(t_us) * 1e-6;
}

bool parseBoolEnv(const char *name, bool default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "no" || value == "off") {
    return false;
  }
  return default_value;
}

bool shouldEnableGui() {
  const bool has_display = (std::getenv("DISPLAY") != nullptr) ||
                           (std::getenv("WAYLAND_DISPLAY") != nullptr);
  return parseBoolEnv("CORNER2TAG_ENABLE_GUI", has_display);
}

} // namespace

int main(int argc, char **argv) {
  corner2tag::io::CalibrationConfig cfg;
  if (!parseArgs(argc, argv, cfg)) {
    return 1;
  }

  corner2tag::io::H5Events h5;
  std::string h5_err;
  if (!corner2tag::io::loadH5Events(cfg.h5_path, h5, &h5_err)) {
    std::cerr << "H5 load failed: " << h5_err << "\n";
    return 1;
  }
  if (h5.ts_us.empty()) {
    std::cerr << "No events in file\n";
    return 1;
  }

  cfg.out_dir = corner2tag::io::prepareCalibrationRunOutputDir(
      cfg.out_dir, corner2tag::io::kCheckerboardDetectionOutputLayout);
  if (!cfg.out_dir.empty()) {
    std::cout << "[out] saving to: " << cfg.out_dir << "\n";
  }

  bool gui_enabled = shouldEnableGui();
  if (!gui_enabled) {
    std::cout << "[viz] GUI disabled (set CORNER2TAG_ENABLE_GUI=1 to force enable)\n";
  }

  // Tracker config
  corner2tag::core::CmIweOptions iwe_opt;
  iwe_opt.sigma = cfg.cm_iwe_sigma;
  iwe_opt.cutoff_factor = cfg.cm_iwe_cutoff_factor;
  iwe_opt.patch_radius_override = cfg.cm_iwe_patch_radius_override;
  iwe_opt.use_variance = cfg.use_variance; // L2 or IWE

  corner2tag::core::CmTracker2DOptions track_opt;
  track_opt.max_iterations = cfg.tracker_max_iterations;
  track_opt.verbose = false;
  track_opt.compute_final_iwe = true;
  track_opt.final_use_full_events = true;

  corner2tag::core::CmTracker2D tracker(cfg.width, cfg.height,
                                  cfg.tracker_num_threads, iwe_opt);

  const uint64_t t0_us = h5.ts_us.front();
  size_t idx = 0;
  size_t window_idx = 0;
  auto fixed_window = corner2tag::core::fixedWindowForRelativeTimestampUs(
      0, cfg.fixed_window_us);

  double vx = 0.0;
  double vy = 0.0;

  std::vector<corner2tag::core::TimedEventNs> window;
  window.reserve(100000);

  const int board_rows = cfg.board_h;
  const int board_cols = cfg.board_w;
  int expected = cfg.expected_corners;
  if (expected <= 0 && board_rows > 0 && board_cols > 0) {
    expected = board_rows * board_cols;
  }
  const std::vector<cv::Point3f> objp =
      corner2tag::core::buildObjectPoints(board_rows, board_cols, cfg.square_size);
  std::vector<std::vector<cv::Point3f>> objpoints;
  std::vector<std::vector<cv::Point2f>> imgpoints;

  while (idx < h5.ts_us.size() || !window.empty()) {
    if (idx < h5.ts_us.size()) {
      const uint64_t t_us = h5.ts_us[idx];
      if (t_us < t0_us) {
        std::cerr << "Event timestamps are not sorted\n";
        return 1;
      }
      const uint64_t relative_t_us = t_us - t0_us;
      if (relative_t_us < fixed_window.end_us) {
        corner2tag::core::TimedEventNs ev;
        ev.x = h5.xs[idx];
        ev.y = h5.ys[idx];
        ev.polarity = (h5.ps[idx] != 0);
        ev.t_ns = static_cast<int64_t>((t_us - t0_us) * 1000ULL);
        window.push_back(ev);
        idx++;
        continue;
      }
    }

    if (!window.empty()) {
      // Convert to SimpleEvent (seconds relative to window start)
      const uint64_t win_t0_us = window.front().t_ns / 1000ULL;
      std::vector<corner2tag::core::SimpleEvent> frame_events;
      frame_events.reserve(window.size());
      for (const auto &e : window) {
        corner2tag::core::SimpleEvent s;
        s.x = e.x;
        s.y = e.y;
        s.polarity = e.polarity;
        const uint64_t t_us_rel =
            static_cast<uint64_t>(e.t_ns / 1000ULL) - win_t0_us;
        s.t = usToSec(t_us_rel);
        frame_events.push_back(s);
      }

      const auto res = tracker.track(frame_events, track_opt, vx, vy);

      vx = res.vx;
      vy = res.vy;

      // Debug stats: dt range, polarity ratio
      uint64_t min_dt = std::numeric_limits<uint64_t>::max();
      uint64_t max_dt = 0;
      size_t pos_cnt = 0;
      const uint64_t t0_ns = static_cast<uint64_t>(window.front().t_ns);
      for (const auto &ev : window) {
        const uint64_t dt = static_cast<uint64_t>(ev.t_ns) - t0_ns;
        min_dt = std::min(min_dt, dt);
        max_dt = std::max(max_dt, dt);
        if (ev.polarity) {
          pos_cnt++;
        }
      }
      const double pos_ratio =
          window.empty() ? 0.0 : static_cast<double>(pos_cnt) / window.size();

      std::cout << "[cm] window=" << window_idx
                << " fixed_window_index=" << fixed_window.index
                << " time_us=[" << fixed_window.begin_us << ","
                << fixed_window.end_us << ") events=" << window.size()
                << " v=(" << vx << "," << vy << ")"
                << " dt_s=[" << (min_dt * 1e-9) << "," << (max_dt * 1e-9) << "]"
                << " pos=" << pos_ratio << " ok=" << res.success << "\n";

      // Patch extraction + corner init
      std::vector<cv::Point> patch_points;
      std::vector<corner2tag::core::PatchBox> boxes;
      std::vector<cv::Point2f> init_corners;
      std::vector<cv::Point2f> refined_corners;
      std::vector<cv::Point2f> filter_corners;
      std::vector<cv::Point2f> ordered_corners;
      bool order_ok = false;
      bool validity_ok = false;

      corner2tag::core::CornerInitOptions corner_opt;
      corner_opt.seed_thr = 10.0f;
      corner_opt.seed_percentile = 90.0f;
      corner_opt.tau = 0.5f;
      corner_opt.tau_percentile = 70.0f;
      corner_opt.max_grow_iters = 20;
      corner_opt.min_component_area = 2;
      corner_opt.weighted_line_fit = true;

      corner2tag::core::CornerRefineOptions ref_opt;
      ref_opt.enable = true;
      ref_opt.lr = cfg.corner_refine_lr;
      ref_opt.max_iter = cfg.corner_refine_max_iter;
      ref_opt.gtol = 1e-6f;
      ref_opt.armijo_c = 1e-4f;
      ref_opt.min_step = 1e-6f;
      ref_opt.strip_half_width0 = cfg.corner_refine_strip_half_width0;
      ref_opt.strip_half_width1 = cfg.corner_refine_strip_half_width1;

      float global_seed_thr = corner_opt.seed_thr;
      float global_tau_thr = corner_opt.tau;

      if (!res.iwe.empty() && !res.piwe.empty()) {
        corner2tag::core::computeGlobalIweThresholds(
            res.iwe, corner_opt, &global_seed_thr, &global_tau_thr);

        corner2tag::core::PatchPointsOptions pp_opt;
        pp_opt.radius = cfg.pp_radius;

        corner2tag::core::PatchClusterOptions cl_opt;
        cl_opt.eps = 0;
        cl_opt.min_component_area = 1;

        corner2tag::core::PatchExtractor extractor(pp_opt, cl_opt);
        patch_points = extractor.computePatchPoints(res.piwe);
        boxes = extractor.clusterToBoxes(patch_points, res.piwe.cols,
                                         res.piwe.rows);

        init_corners.reserve(boxes.size());
        refined_corners.reserve(boxes.size());
        for (const auto &b : boxes) {
          const int w = b.x1 - b.x0;
          const int h = b.y1 - b.y0;
          if (w <= 1 || h <= 1) {
            continue;
          }
          const cv::Rect roi(b.x0, b.y0, w, h);
          const cv::Mat iwe_patch = res.iwe(roi);
          const auto init_res = corner2tag::core::initializeCornerFromIwePatch(
              iwe_patch, corner_opt, global_seed_thr, global_tau_thr);
          if (!init_res.success) {
            continue;
          }
          auto ref_res = corner2tag::core::refineCornerInIwePatch(
              iwe_patch, init_res.init_corner_xy, init_res.line0,
              init_res.line1, ref_opt);
          const cv::Point2f corner_global(
              static_cast<float>(b.x0) + init_res.corner_xy.x,
              static_cast<float>(b.y0) + init_res.corner_xy.y);
          init_corners.push_back(corner_global);
          if (ref_res.success) {
            refined_corners.emplace_back(
                static_cast<float>(b.x0) + ref_res.refined_xy.x,
                static_cast<float>(b.y0) + ref_res.refined_xy.y);
          }
        }

        const std::vector<cv::Point2f> &cand =
            refined_corners.empty() ? init_corners : refined_corners;
        if (expected > 0 && board_rows > 0 && board_cols > 0 &&
            static_cast<int>(cand.size()) >= expected) {
          auto filt = corner2tag::core::filterCheckerboardCorners(cand, board_rows,
                                                            board_cols);
          if (filt.success &&
              static_cast<int>(filt.filtered.size()) == expected) {
            filter_corners = filt.filtered;
            auto ord = corner2tag::core::orderCheckerboardCorners(
                filt.filtered, board_rows, board_cols);
            if (ord.success &&
                static_cast<int>(ord.ordered.size()) == expected) {
              ordered_corners = ord.ordered;
              order_ok = true;
              validity_ok = corner2tag::core::isCheckerboardValid(
                  ord.ordered, board_rows, board_cols,
                  cfg.checkerboard_tor_spacing, cfg.checkerboard_tor_orth);
              if (validity_ok) {
                imgpoints.push_back(ordered_corners);
                objpoints.push_back(objp);
              }
            }
          }
        }
      }

      std::cout << "[corner] window=" << window_idx
                << " refined=" << refined_corners.size()
                << " filtered=" << filter_corners.size()
                << " validity=" << (validity_ok ? "ok" : "no") << "\n";

      const double window_dt_s =
          window.size() > 1
              ? (static_cast<double>(window.back().t_ns - window.front().t_ns) *
                 1e-9)
              : 0.0;

      // Visualization
      const auto vis = corner2tag::viz::buildCheckerboardWindowVis(
          window, cfg.width, cfg.height, res.iwe, init_corners, refined_corners,
          filter_corners, ordered_corners, order_ok, validity_ok,
          static_cast<float>(vx), static_cast<float>(vy), window_dt_s,
          board_rows, board_cols,
          cfg.viz_zoom);
      if (gui_enabled) {
        try {
          corner2tag::viz::showCheckerboardWindowVis(vis);
          cv::waitKey(1);
        } catch (const cv::Exception &e) {
          std::cerr
              << "[viz] GUI unavailable, disabling window rendering: "
              << e.what() << "\n";
          gui_enabled = false;
        }
      }
      // Save detection outputs
      corner2tag::io::saveDetectionOutputs(
          cfg.out_dir, window_idx, vis.raw_vis, vis.iwe_vis, vis.detection_vis,
          corner2tag::io::kCheckerboardDetectionOutputLayout);

      window_idx++;
    }

    window.clear();
    if (idx < h5.ts_us.size()) {
      fixed_window = corner2tag::core::fixedWindowForRelativeTimestampUs(
          h5.ts_us[idx] - t0_us, cfg.fixed_window_us);
    }
  }
  if (gui_enabled) {
    cv::destroyAllWindows();
  }

  if (!imgpoints.empty() && !objpoints.empty()) {
    const size_t total_windows = window_idx;
    const size_t used_windows = imgpoints.size();
    std::cout << "[calib] Calibrating with " << used_windows << "/"
              << total_windows << " windows...\n";
    const cv::Size image_size(cfg.width, cfg.height);
    corner2tag::core::CalibrationOptions calib_opt;
    calib_opt.max_iter = cfg.calib_max_iter;
    calib_opt.fix_k3plus = cfg.calib_fix_k3plus;
    calib_opt.use_intrinsic_guess = cfg.calib_use_intrinsic_guess;
    const auto calib = corner2tag::core::calibrateCheckerboard(
        objpoints, imgpoints, image_size, calib_opt);
    if (calib.success) {
      std::cout << "[calib] success\n";
      std::cout << "[calib] reproj=" << calib.reprojection_error << "\n";
      corner2tag::io::saveCheckerboardCalibrationYaml(
          cfg.out_dir, used_windows, total_windows, board_rows, board_cols,
          cfg.square_size, calib.camera_matrix, calib.dist_coeffs,
          calib.reprojection_error);
      corner2tag::viz::saveCalibrationReportImages(
          cfg.out_dir, image_size, calib.camera_matrix, calib.dist_coeffs,
          calib.rvecs, calib.tvecs, objpoints, imgpoints);
    } else {
      std::cout << "[calib] failed\n";
    }
  } else {
    std::cout << "[calib] no valid views\n";
  }

  return 0;
}
