#include "zlib.hh"

#include "tb/util/range.hh"

#include <vector>
#include <optional>

using tb::range;
using std::optional;
using std::vector;
using std::min;
using std::max;

namespace tbf {

static auto reverse_32(u32 x) -> u32 {
  x = ((x & 0xffff0000) >> 16) | ((x & 0x0000ffff) << 16);
  x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
  x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
  x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
  x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
  return x;
}

static auto reverse_16(u16 x) -> u16 {
  x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
  x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
  x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
  x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
  return x;
}

[[maybe_unused]] static auto generate_bit_reverse_lookup_table(u8 bits) -> vector<u16> {
  auto table = vector<u16>(1 << bits);
  for (auto i: range(table.size())) {
    table[i] = reverse_16(i);
  }
  return table;
}

// Lookup table for reversing bit patterns up to (1 << 9) = 512;
static constexpr u16 reverse_9_table[] = {
  0, 32768, 16384, 49152, 8192, 40960, 24576, 57344, 4096, 36864, 20480, 53248,
  12288, 45056, 28672, 61440, 2048, 34816, 18432, 51200, 10240, 43008, 26624,
  59392, 6144, 38912, 22528, 55296, 14336, 47104, 30720, 63488, 1024, 33792,
  17408, 50176, 9216, 41984, 25600, 58368, 5120, 37888, 21504, 54272, 13312,
  46080, 29696, 62464, 3072, 35840, 19456, 52224, 11264, 44032, 27648, 60416,
  7168, 39936, 23552, 56320, 15360, 48128, 31744, 64512, 512, 33280, 16896,
  49664, 8704, 41472, 25088, 57856, 4608, 37376, 20992, 53760, 12800, 45568,
  29184, 61952, 2560, 35328, 18944, 51712, 10752, 43520, 27136, 59904, 6656,
  39424, 23040, 55808, 14848, 47616, 31232, 64000, 1536, 34304, 17920, 50688,
  9728, 42496, 26112, 58880, 5632, 38400, 22016, 54784, 13824, 46592, 30208,
  62976, 3584, 36352, 19968, 52736, 11776, 44544, 28160, 60928, 7680, 40448,
  24064, 56832, 15872, 48640, 32256, 65024, 256, 33024, 16640, 49408, 8448,
  41216, 24832, 57600, 4352, 37120, 20736, 53504, 12544, 45312, 28928, 61696,
  2304, 35072, 18688, 51456, 10496, 43264, 26880, 59648, 6400, 39168, 22784,
  55552, 14592, 47360, 30976, 63744, 1280, 34048, 17664, 50432, 9472, 42240,
  25856, 58624, 5376, 38144, 21760, 54528, 13568, 46336, 29952, 62720, 3328,
  36096, 19712, 52480, 11520, 44288, 27904, 60672, 7424, 40192, 23808, 56576,
  15616, 48384, 32000, 64768, 768, 33536, 17152, 49920, 8960, 41728, 25344,
  58112, 4864, 37632, 21248, 54016, 13056, 45824, 29440, 62208, 2816, 35584,
  19200, 51968, 11008, 43776, 27392, 60160, 6912, 39680, 23296, 56064, 15104,
  47872, 31488, 64256, 1792, 34560, 18176, 50944, 9984, 42752, 26368, 59136,
  5888, 38656, 22272, 55040, 14080, 46848, 30464, 63232, 3840, 36608, 20224,
  52992, 12032, 44800, 28416, 61184, 7936, 40704, 24320, 57088, 16128, 48896,
  32512, 65280, 128, 32896, 16512, 49280, 8320, 41088, 24704, 57472, 4224,
  36992, 20608, 53376, 12416, 45184, 28800, 61568, 2176, 34944, 18560, 51328,
  10368, 43136, 26752, 59520, 6272, 39040, 22656, 55424, 14464, 47232, 30848,
  63616, 1152, 33920, 17536, 50304, 9344, 42112, 25728, 58496, 5248, 38016,
  21632, 54400, 13440, 46208, 29824, 62592, 3200, 35968, 19584, 52352, 11392,
  44160, 27776, 60544, 7296, 40064, 23680, 56448, 15488, 48256, 31872, 64640,
  640, 33408, 17024, 49792, 8832, 41600, 25216, 57984, 4736, 37504, 21120,
  53888, 12928, 45696, 29312, 62080, 2688, 35456, 19072, 51840, 10880, 43648,
  27264, 60032, 6784, 39552, 23168, 55936, 14976, 47744, 31360, 64128, 1664,
  34432, 18048, 50816, 9856, 42624, 26240, 59008, 5760, 38528, 22144, 54912,
  13952, 46720, 30336, 63104, 3712, 36480, 20096, 52864, 11904, 44672, 28288,
  61056, 7808, 40576, 24192, 56960, 16000, 48768, 32384, 65152, 384, 33152,
  16768, 49536, 8576, 41344, 24960, 57728, 4480, 37248, 20864, 53632, 12672,
  45440, 29056, 61824, 2432, 35200, 18816, 51584, 10624, 43392, 27008, 59776,
  6528, 39296, 22912, 55680, 14720, 47488, 31104, 63872, 1408, 34176, 17792,
  50560, 9600, 42368, 25984, 58752, 5504, 38272, 21888, 54656, 13696, 46464,
  30080, 62848, 3456, 36224, 19840, 52608, 11648, 44416, 28032, 60800, 7552,
  40320, 23936, 56704, 15744, 48512, 32128, 64896, 896, 33664, 17280, 50048,
  9088, 41856, 25472, 58240, 4992, 37760, 21376, 54144, 13184, 45952, 29568,
  62336, 2944, 35712, 19328, 52096, 11136, 43904, 27520, 60288, 7040, 39808,
  23424, 56192, 15232, 48000, 31616, 64384, 1920, 34688, 18304, 51072, 10112,
  42880, 26496, 59264, 6016, 38784, 22400, 55168, 14208, 46976, 30592, 63360,
  3968, 36736, 20352, 53120, 12160, 44928, 28544, 61312, 8064, 40832, 24448,
  57216, 16256, 49024, 32640, 65408};

struct bit_chunk {
  u32 code;
  u32 length;
  bit_chunk(u32 code, u32 length): code(code), length(length) {}
};

static auto reverse_bits(u32 code, u32 length) -> bit_chunk {
  assert(length <= 9);
  return {static_cast<u32>(reverse_9_table[code] >> (16 - length)), length};
}

static auto concat_bits(bit_chunk b1, bit_chunk b2) -> bit_chunk {
  return {b1.code | (b2.code << b1.length), b1.length + b2.length};
}

static auto get_deflate_static_huffman_code(u16 b) -> bit_chunk {
  assert(b < 288);
  auto [code, length] = (b < 144) ? std::make_pair(48 + b, 8) :
                        (b < 256) ? std::make_pair(b - 144 + 400, 9) :
                        (b < 280) ? std::make_pair(b - 256 + 0, 7) :
                        std::make_pair(b - 280 + 192, 8);
  return reverse_bits(code, length);
}

static auto get_deflate_distance_code(u32 distance) -> bit_chunk {
  auto offset = distance - 1;
  if (offset < 4)
    return reverse_bits(offset, 5);
  auto width = 30u - __builtin_clz(offset);
  auto index = offset - (1 << (width + 1));
  auto code = 2 * (width + 1) + (index >> width);
  auto extra = bit_chunk(index & ((1 << width) - 1), width);
  return concat_bits(reverse_bits(code, 5), extra);
}

static auto get_deflate_length_code(u32 length) -> bit_chunk {
  auto offset = length - 3;
  if (offset < 8)
    return get_deflate_static_huffman_code(257 + offset);
  if (length == 258)
    return get_deflate_static_huffman_code(285);
  auto width = 29 - __builtin_clz(offset);
  auto index = offset - (1 << (width + 2));
  auto code = 261 + (4 * width) + (index >> width);
  auto extra = bit_chunk(index & ((1 << width) - 1), width);
  return concat_bits(get_deflate_static_huffman_code(code), extra);
}

static auto append_bits_to_bytes_right(buffer& bytes, size_t& nextbit, u32 data, u32 length) -> void {
  auto bytePos = nextbit / 8;
  auto bitPos = nextbit % 8;
  int len = length;
  if (bitPos) {
    auto segment = 8 - bitPos;
    bytes[bytePos++] |= (data & ((1 << segment) - 1)) << bitPos;
    data >>= segment;
    len -= segment;
  }
  auto remaining = (len + 7) / 8;
  for (auto i = 0; i < remaining; i += 1) {
    bytes[bytePos++] |= data;
    data >>= 8;
  }
  nextbit += length;
}

struct match_result {
  u16 distance;
  u16 length;
};

static constexpr auto nil = std::nullopt;

static auto find_match(buffer_view data, int pos, double level) -> optional<match_result> {
  constexpr auto deflateMinLength = 3;
  constexpr auto deflateMaxLength = 258;

  auto longest = min({pos - 1, deflateMaxLength, static_cast<int>(data.size() - pos)});

  if (longest < deflateMinLength)
    return nil;

  auto deflateMaxDistance = static_cast<int>(3276.8 * level);

  auto startDistance = min(pos, deflateMaxDistance);

  auto maxLength = u16 {0};
  auto distanceForMax = u16 {0};

  for (auto i = pos - 1; i >= pos - startDistance && maxLength != longest; i -= 1){
    auto matchLength = 0;
    auto done = false;
    for (auto j = 0; j < longest && !done; j += 1) {
      if (data[i + j] == data[pos + j]) {
        matchLength += 1;
      } else {
        done = true;
      }
    }

    if (matchLength >= deflateMinLength && matchLength > maxLength){
      maxLength = matchLength;
      distanceForMax = pos - i;
    }
  }

  if (deflateMinLength > maxLength)
    return nil;

  return {{distanceForMax, maxLength}};
}

class bit_stream {
  buffer bytes_;
  size_t current_bit_;
public:
  bit_stream(size_t capacity_in_bytes):
    bytes_(capacity_in_bytes),
    current_bit_(0) {};
  auto append(bit_chunk b) -> void {
    append_bits_to_bytes_right(bytes_, current_bit_, b.code, b.length);
  }
  auto finish() -> buffer {
    auto result = buffer((current_bit_ + 7) / 8);
    copy_n(bytes_.begin(), (current_bit_ + 7) / 8, result.begin());
    // return bytes_.resize((current_bit_ + 7) / 8);
    return result;
  }
};

static auto deflate_data_static_huffman(buffer_view data, double level) -> buffer {
  auto output = bit_stream(max(4 * data.size() / 3, 100ul));
  output.append({1, 1});  // final block
  output.append({1, 2});  // fixed code
  auto i = 0ul;
  while (i < data.size()) {
    auto match = find_match(data, i, level);
    if (match) {
      auto lengthCode = get_deflate_length_code(match->length);
      auto distanceCode = get_deflate_distance_code(match->distance);
      output.append(lengthCode);
      output.append(distanceCode);
      i += match->length;
    } else {
      auto code = get_deflate_static_huffman_code(data[i]);
      output.append(code);
      i += 1;
    }
  }
  output.append(get_deflate_static_huffman_code(256));  // stop symbol
  return output.finish();
}

static auto compute_adler32(buffer_view data) -> u32 {
  auto a = u16 {1};
  auto b = u16 {0};
  auto m = u16 {65521};
  for (auto x: data) {
    a = (a + x) % m;
    b = (b + a) % m;
  }
  return (b << 16) + a;
}

auto zlib_compress_static_huffman(buffer_view data, double level) -> zlib_result {
  auto zlib = zlib_result {120, 1, deflate_data_static_huffman(data, level), compute_adler32(data)};
  // zlib.CMF = 120;
  // zlib.FLG = 1;
  // zlib.compressed_data_blocks = deflate_data_static_huffman(data, level);
  // zlib.adler32_check_value = compute_adler32(data);
  return zlib;
}

}