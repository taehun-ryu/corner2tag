#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace corner2tag::io {

struct H5Events {
  std::vector<uint64_t> ts_us;
  std::vector<uint16_t> xs;
  std::vector<uint16_t> ys;
  std::vector<uint8_t> ps;
};

// Load events from H5 file.
// Expected datasets:
//   /events/ts (uint64, microseconds)
//   /events/xs (uint16)
//   /events/ys (uint16)
//   /events/ps (uint8)
bool loadH5Events(const std::string &path, H5Events &out,
                  std::string *err = nullptr);

} // namespace corner2tag::io
