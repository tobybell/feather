#pragma once

namespace tbf {

using byte = unsigned char;
using size_t = unsigned long;
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long;

constexpr auto copy_n(byte const* first, size_t n, byte* result) -> byte* {
  while (n--) {
    *result++ = *first++;
  }
  return result;
}

class buffer_view {
  byte const* data_;
  size_t size_;
public:
  buffer_view(byte const* data, size_t size): data_(data), size_(size) {};
  buffer_view(byte const* begin, byte const* end): data_(begin), size_(end - begin) {};
  auto data() const { return data_; }
  auto size() const { return size_; }
  auto begin() const { return data_; }
  auto end() const { return data_ + size_; }
  auto operator[](size_t index) const -> byte {
    // assert(index < size_);
    return data_[index];
  }
};

struct uninitialized_t {};
constexpr uninitialized_t uninitialized;

constexpr auto swap(size_t& x1, size_t& x2) -> void {
  auto tmp = x1;
  x1 = x2;
  x2 = tmp;
}
constexpr auto swap(byte* x1, byte* x2) -> void {
  auto tmp = x1;
  x1 = x2;
  x2 = tmp;
}

class buffer {
  byte* data_;
  size_t size_;
public:
  buffer(size_t size): data_(new byte[size]()), size_(size) {}
  buffer(size_t size, uninitialized_t): data_(new byte[size]), size_(size) {}
  buffer(buffer&& other): data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  buffer(buffer const& other) = delete;
  buffer& operator=(buffer other) {
    swap(data_, other.data_);
    swap(size_, other.size_);
    return *this;
  }
  ~buffer() {
    delete[] data_;
    data_ = nullptr;
  };
  operator buffer_view() const { return {data_, size_}; }
  auto data() const { return data_; }
  auto size() const { return size_; }
  auto begin() const { return data_; }
  auto end() const { return data_ + size_; }
  auto operator[](size_t index) const -> byte const& {
    // assert(index < size_);
    return data_[index];
  }
  auto operator[](size_t index) -> byte& {
    // assert(index < size_);
    return data_[index];
  }
  auto copy() -> buffer {
    auto result = buffer(size_, uninitialized);
    copy_n(result.data_, size_, data_);
    return result;
  }
};


}
