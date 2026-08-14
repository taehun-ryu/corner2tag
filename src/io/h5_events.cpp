#include "corner2tag/io/h5_events.hpp"

#include <hdf5.h>

#include <stdexcept>

namespace corner2tag::io {

namespace {

template <typename T>
std::vector<T> read1D(hid_t file, const std::string &path) {
  hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    throw std::runtime_error("H5Dopen2 failed: " + path);
  }
  hid_t space = H5Dget_space(dset);
  int ndims = H5Sget_simple_extent_ndims(space);
  if (ndims != 1) {
    H5Sclose(space);
    H5Dclose(dset);
    throw std::runtime_error("dataset is not 1D: " + path);
  }
  hsize_t dims[1];
  H5Sget_simple_extent_dims(space, dims, nullptr);
  const size_t n = static_cast<size_t>(dims[0]);

  std::vector<T> out(n);
  hid_t memtype = H5T_NATIVE_UINT8;
  if constexpr (std::is_same<T, uint64_t>::value) {
    memtype = H5T_NATIVE_UINT64;
  } else if constexpr (std::is_same<T, uint32_t>::value) {
    memtype = H5T_NATIVE_UINT32;
  } else if constexpr (std::is_same<T, uint16_t>::value) {
    memtype = H5T_NATIVE_UINT16;
  } else if constexpr (std::is_same<T, int32_t>::value) {
    memtype = H5T_NATIVE_INT32;
  } else if constexpr (std::is_same<T, int64_t>::value) {
    memtype = H5T_NATIVE_INT64;
  }

  if (H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) {
    H5Sclose(space);
    H5Dclose(dset);
    throw std::runtime_error("H5Dread failed: " + path);
  }
  H5Sclose(space);
  H5Dclose(dset);
  return out;
}

} // namespace

bool loadH5Events(const std::string &path, H5Events &out, std::string *err) {
  try {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
      throw std::runtime_error("H5Fopen failed: " + path);
    }

    out.ts_us = read1D<uint64_t>(file, "/events/ts");
    out.xs = read1D<uint16_t>(file, "/events/xs");
    out.ys = read1D<uint16_t>(file, "/events/ys");
    out.ps = read1D<uint8_t>(file, "/events/ps");

    H5Fclose(file);

    if (out.ts_us.size() != out.xs.size() ||
        out.ts_us.size() != out.ys.size() ||
        out.ts_us.size() != out.ps.size()) {
      throw std::runtime_error("events arrays have mismatched sizes");
    }
    return true;
  } catch (const std::exception &e) {
    if (err) {
      *err = e.what();
    }
    return false;
  }
}

} // namespace corner2tag::io
