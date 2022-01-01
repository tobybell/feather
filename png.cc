#include "png.hh"

#include "zlib.hh"

#include "tb/util/range.hh"

#include <array>
#include <vector>

using std::vector;

using tb::range;

namespace tbf {

struct png_ihdr {
  u32 width;
  u32 height;
  u8 bit_depth;
  u8 color_type;
  u8 compression_method;
  u8 filter_method;
  u8 interlace_method;
};

struct png_image {
  png_ihdr ihdr;
  zlib_result zlib;
};

static constexpr auto png_header_length = 4 + 4 + 1 + 1 + 1 + 1 + 1;
static constexpr auto png_signature = {137, 80, 78, 71, 13, 10, 26, 10};

static auto png_idat_length(png_image const& png) -> size_t {
  return 2 + png.zlib.compressed_data_blocks.size() + 4;
}

static auto write_byte(buffer& data, u8 b, size_t& position){
  data[position++] = b;
}

static auto write_4_bytes_be(buffer& data, u32 b, size_t& position) -> void {
  data[position++] = b >> 24;
  data[position++] = b >> 16;
  data[position++] = b >> 8;
  data[position++] = b;
}

static auto write_string(buffer &data, const char* str, size_t& position) -> void {
  auto length = strlen(str);
  for (auto i: range(length)) {
    data[position++] = str[i];
  }
}

using Crc32 = u32;

using Crc32Table = std::array<u32, 256>;

static auto make_crc32_table() -> Crc32Table {
  auto crcTable = Crc32Table();
  for (u32 n = 0; n < 256; n += 1) {
    auto c = n;
    for (auto k = 0; k < 8; k += 1) {
      if (c % 2) {
        c = 3988292384 ^ (c >> 1);
      } else {
        c >>= 1;
      }
    }
    crcTable[n] = c;
  }
  return crcTable;
}

static auto update_crc32(Crc32 crc, buffer_view buf, Crc32Table const& crc_table) -> Crc32 {
  for (auto b: buf) {
    auto index = (crc ^ b) & 0xff;
    crc = crc_table[index] ^ crc >> 8;
  }
  return crc;
}

static auto calculate_crc32(buffer_view data) -> Crc32 {
  auto crcTable = make_crc32_table();
  return update_crc32(0xffffffff, data, crcTable) ^ 0xffffffff;
}

static auto crc32(buffer const& data, size_t from, size_t length) -> u32 {
  return calculate_crc32(buffer_view(&data[from], length));
}

static auto png_serialize_chunks(png_image const& png) -> buffer {
  auto length = png_signature.size() + 12 + png_header_length + 12 + png_idat_length(png) + 12;
  auto data = buffer(length);
  auto position = 0ul;

  // Signature
  for (auto x: png_signature) {
    write_byte(data, x, position);
  }

  // Header
  auto chunk_length = png_header_length;
  write_4_bytes_be(data, chunk_length, position);
  write_string(data, "IHDR", position);
  write_4_bytes_be(data, png.ihdr.width, position);
  write_4_bytes_be(data, png.ihdr.height, position);
  write_byte(data, png.ihdr.bit_depth, position);
  write_byte(data, png.ihdr.color_type, position);
  write_byte(data, png.ihdr.compression_method, position);
  write_byte(data, png.ihdr.filter_method, position);
  write_byte(data, png.ihdr.interlace_method, position);
  write_4_bytes_be(data, crc32(data, position - chunk_length - 4, chunk_length + 4), position);

  // IDAT
  chunk_length = png_idat_length(png);
  write_4_bytes_be(data, chunk_length, position);
  write_string(data, "IDAT", position);
  write_byte(data, png.zlib.CMF, position);
  write_byte(data, png.zlib.FLG, position);
  for (auto x: png.zlib.compressed_data_blocks) {
    write_byte(data, x, position);
  }
  write_4_bytes_be(data, png.zlib.adler32_check_value, position);
  write_4_bytes_be(data, crc32(data, position - chunk_length - 4, chunk_length + 4), position);

  // IEND
  chunk_length = 0.0;
  write_4_bytes_be(data, chunk_length, position);
  write_string(data, "IEND", position);
  write_4_bytes_be(data, crc32(data, position - 4.0, 4.0), position);

  return data;
}

struct rgba {
  u8 r, g, b, a;
};

struct rgba_bitmap {
  u32 width;
  u32 height;
  std::vector<rgba> data;
  auto operator()(u32 row, u32 col) -> rgba& {
    return data[row * width + col];
  }
  auto operator()(u32 row, u32 col) const -> rgba {
    return data[row * width + col];
  }
};

auto get_png_color_data(rgba_bitmap const& image) -> buffer {
  auto length = 4 * image.width * image.height + image.height;
  auto color_data = buffer(length, uninitialized);
  auto next = color_data.begin();
  for (auto y = 0; y < image.height; y += 1) {
    *next++ = 0;
    for (auto x = 0; x < image.width; x += 1) {
      auto p = image(y, x);
      *next++ = p.r;
      *next++ = p.g;
      *next++ = p.b;
      *next++ = p.a;
    }
  }
  return color_data;
}

auto get_png_color_data(byte const* rgba, size_t height, size_t width) -> buffer {
  auto length = 4 * width * height + height;
  auto color_data = buffer(length, uninitialized);
  auto next = color_data.begin();
  for (auto y = 0; y < height; y += 1) {
    *next++ = 0;
    for (auto x = 0; x < width; x += 1) {
      auto p = &rgba[4 * (y * width + x)];
      *next++ = p[0];
      *next++ = p[1];
      *next++ = p[2];
      *next++ = p[3];
    }
  }
  return color_data;
}

// static auto convert_to_png_with_options(rgba_bitmap const& image, u8 color_type, double compressionLevel) -> vector<u8> {
//   auto color_data = get_png_color_data(image);
//   auto png = png_image {
//     {image.width, image.height, 8, color_type, 0, 0, 0},
//     zlib_compress_static_huffman(color_data, compressionLevel),
//   };
//   // TODO: handle grayscale elsewhere
//   // auto color_data = color_type == 6. ? GetPNGColorData(image) : GetPNGColorDataGreyscale(image);
//   // png.ihdr = ;
//   // png.zlib = ;

//   return png_serialize_chunks(png);
// }

// static auto convert_to_png(rgba_bitmap const& image) -> vector<u8> {
//   return convert_to_png_with_options(image, 6.0, 0.001);
// }

auto save_rgba(byte const* rgba, size_t height, size_t width) -> buffer {
  u8 color_type = 6;
  auto color_data = get_png_color_data(rgba, height, width);
  auto png = png_image {
    {static_cast<u32>(width), static_cast<u32>(height), 8, color_type, 0, 0, 0},
    zlib_compress_static_huffman(color_data, .001),
  };
  return png_serialize_chunks(png);
}

}
