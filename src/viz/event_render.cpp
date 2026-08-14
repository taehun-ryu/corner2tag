#include "corner2tag/viz/event_render.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "corner2tag/core/simple_event.hpp"

namespace corner2tag::viz {

cv::Mat eventsToImage(const std::vector<corner2tag::core::TimedEventNs> &events,
                      int width, int height, float zoom_factor,
                      bool non_max_suppression, int add_value) {
  float z = zoom_factor;
  if (z <= 0.0f) {
    z = 1.0f;
  }

  cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

  for (const auto &e : events) {
    // Match ROS1-style visualization: ON -> Blue, OFF -> Red
    const int channel = (e.polarity == 1) ? 0 : 2;

    int u = static_cast<int>(e.x);
    int v = static_cast<int>(e.y);

    if (u < 0 || v < 0 || u >= width || v >= height) {
      continue;
    }

    auto &pix = image.at<cv::Vec3b>(v, u)[channel];
    if (pix > 255 - add_value) {
      pix = 255;
    } else {
      pix = static_cast<unsigned char>(pix + add_value);
    }
  }

  if (non_max_suppression) {
    for (int u = 0; u < width; ++u) {
      for (int v = 0; v < height; ++v) {
        for (int c = 0; c < 3; ++c) {
          if (image.at<cv::Vec3b>(v, u)[c] < 80) {
            image.at<cv::Vec3b>(v, u)[c] = 0;
          }
        }
      }
    }
  }

  if (std::abs(z - 1.0f) < 1e-6f) {
    return image;
  }

  const int out_w = std::max(1, static_cast<int>(std::lround(width * z)));
  const int out_h = std::max(1, static_cast<int>(std::lround(height * z)));
  cv::Mat out;
  cv::resize(image, out, cv::Size(out_w, out_h), 0, 0, cv::INTER_NEAREST);
  return out;
}

} // namespace corner2tag::viz
