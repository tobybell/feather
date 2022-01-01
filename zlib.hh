#pragma once

#include "common.hh"

namespace tbf {

struct zlib_result {
  u8 CMF;
  u8 FLG;
  buffer compressed_data_blocks;
  u32 adler32_check_value;
};

auto zlib_compress_static_huffman(buffer_view b, double level) -> zlib_result;

}
