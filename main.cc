#include <tb/mutable-file-buffer.hh>
#include <tb/array.hh>
#include <tb/util/range.hh>

#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <vector>

using namespace tb;
using std::cerr;
using std::chrono::system_clock;
using std::clamp;
using std::endl;
using std::max;
using std::min;
using std::mt19937;
using std::pair;
using std::sort;
using std::swap;
using std::vector;

struct ppm_pixel {
  unsigned char* data;
  auto& r() const { return data[0]; }
  auto& g() const { return data[1]; }
  auto& b() const { return data[2]; }
};

/** Pixel image saved in a PPM file. */
struct ppm_image {

  mutable_file_buffer buf;
  unsigned int w_;
  unsigned int h_;

  ppm_image(
    const char* filename,
    unsigned int w,
    unsigned int h):
      buf {filename, 24 + w * h * 3},
      w_ {w},
      h_ {h} {
    assert(w < 10'000);
    assert(h < 10'000);
    sprintf(reinterpret_cast<char*>(buf.base()), "P6 %8d %7d 255\n", w, h);
  }

  auto operator()(unsigned int i, unsigned int j) {
    return ppm_pixel {
      reinterpret_cast<unsigned char*>(buf.base() + 24 + 3 * (i * w_ + j))
    };
  }

  constexpr auto width() const { return w_; }
  constexpr auto height() const { return h_; }
};

struct color {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;

  color(uint32_t rgba):
    r {static_cast<uint8_t>(rgba >> 24)},
    g {static_cast<uint8_t>(rgba >> 16)},
    b {static_cast<uint8_t>(rgba >> 8)},
    a {static_cast<uint8_t>(rgba >> 0)} {}

  color(uint8_t r, uint8_t g, uint8_t b, uint8_t a):
    r {r}, g {g}, b {b}, a {a} {}
};

struct pixel {
  unsigned char* data;
  auto& r() const { return data[0]; }
  auto& g() const { return data[1]; }
  auto& b() const { return data[2]; }
  auto& a() const { return data[3]; }
};

color col(pixel p) {
  return {p.r(), p.g(), p.b(), p.a()};
}

/** Saturating add a color to a pixel. */
void operator+=(pixel p, color c) {
  p.r() = min(255, p.r() + c.r);
  p.g() = min(255, p.g() + c.g);
  p.b() = min(255, p.b() + c.b);
  p.a() = min(255, p.a() + c.a);
}

/** A buffer to render non-overlapping coverage pieces into. */
struct layer {
  unsigned char* data;
  unsigned int w_;
  unsigned int h_;

  layer(unsigned int h, unsigned int w):
    data {new unsigned char[4 * h * w]()},
    h_ {h},
    w_ {w} {}

  layer(const layer&) = delete;
  layer(layer&&) = default;

  ~layer() {
    delete[] data;
    data = nullptr;
  }

  constexpr auto width() const { return w_; }
  constexpr auto height() const { return h_; }

  auto operator()(unsigned int i, unsigned int j) {
    assert(i < height());
    assert(j < width());
    return pixel {data + 4 * (i * w_ + j)};
  }
};

struct point {
  float x;
  float y;
  constexpr point(float x, float y): x(x), y(y) {}
};

// Represents a line in the mathematical sense.
struct line {
  float x;
  float y;
};

line join(point a, point b) {
  auto i = a.y - b.y;
  auto j = b.x - a.x;
  auto k = 1.f / (a.x * b.y - a.y * b.x);
  return {i * k, j * k};
}

point join(line a, line b) {
  auto i = a.y - b.y;
  auto j = b.x - a.x;
  auto k = 1.f / (a.x * b.y - a.y * b.x);
  return {i * k, j * k};
}

void fill_sample(ppm_image& im, unsigned int x, unsigned int y, color c = 0xff0000ff) {
  if (x >= im.width() || y >= im.height()) return;
  auto px = im(y, x);

  // SVG simple alpha compositing
  int Ea = c.a,
      Er = c.r,
      Eg = c.g,
      Eb = c.b;
	int Ca = 255,
      Cr = px.r(),
      Cg = px.g(),
      Cb = px.b();
  // int Ea_ = 255 - (255 - Ea) * (255 - Ca) / 255;
	px.r() = ((255 - Ea) * Cr / 255 + Er);
	px.g() = ((255 - Ea) * Cg / 255 + Eg);
	px.b() = ((255 - Ea) * Cb / 255 + Eb);
	// sample_buffer[4 * (x + y * w) + 3] = Ea_;
}

void cover(layer& l, unsigned int i, unsigned int j, color c) {
  l(i, j) += c;
}

float partialArea(float s0, float s1, float x0, float x1) {
  float w = x1 - x0;
  float h = w == 0 ? .5f : .5f / w;
  auto entry = clamp(s1 - x0, 0.f, w);
  auto exit = clamp(s0 - x0, 0.f, w);
  auto full = clamp(s1 - x1, 0.f, 1.f);
  auto area = full + h * (entry * entry - exit * exit);
  return area;
}

auto fade(color c, float a) {
  return color(c.r * a, c.g * a, c.b * a, c.a * a);
}

void asaRow(layer& l, unsigned int i, color c, float min1, float min2, float max1, float max2) {
  assert(min2 >= min1);
  assert(max2 >= max1);

  int sLeftMin = int(floor(min1));
  int sLeftMax = int(ceil(min2));
  int sRightMin = int(floor(max1));
  int sRightMax = int(ceil(max2));

  int sLeftLim = sLeftMax;
  int sRightLim = sRightMin;

  if (sRightMin < sLeftMax) {
    // If the left and right alpha sections intersect... idk
    sLeftLim = sRightMin;
    sRightLim = sLeftMax;
  }

  // Left fade.
  for (int sx = sLeftMin; sx < sLeftLim; sx += 1) {
    auto area = partialArea(sx, sx + 1.f, min1, min2);
    cover(l, i, sx, fade(c, area));
  }
  // Solid section.
  for (int j = sLeftMax; j < sRightMin; j += 1) {
    cover(l, i, j, c);
  }
  // Overlapping fade section.
  for (int j = sRightMin; j < sLeftMax; j += 1) {
    auto area =
      partialArea(j, j + 1.f, min1, min2) -
      partialArea(j, j + 1.f, max1, max2);
    cover(l, i, j, fade(c, area));
  }
  // Right fade.
  for (int sx = sRightLim; sx < sRightMax; sx += 1) {
    auto area = 1.f - partialArea(sx, sx + 1.f, max1, max2);
    cover(l, i, sx, fade(c, area));
  }
}

static inline void order(float& x, float& y) {
  if (y < x) swap(x, y);
}

static inline pair<float, float> ordered(float x, float y) {
  if (y < x) return {y, x};
  return {x, y};
}

void axisQuad(layer& l, float y, float x0, float x1, float sL, float sR, unsigned int imin, unsigned int imax, color c) {
  auto ka = y, kb = y - 1.f, kc = ka, kd = kb;
  if (sL < 0) swap(ka, kb);
  if (sR < 0) swap(kc, kd);
  
  for (auto i = imin; i < imax; i += 1) {
    auto py = float(i);
    asaRow(l, i, c,
      x0 + (py - ka) * sL,
      x0 + (py - kb) * sL,
      x1 + (py - kc) * sR,
      x1 + (py - kd) * sR);
  }
}

void axisTrap(layer& l, float y0, float y1, float x0, float x1, float sL, float sR, color c) {

  auto h = y1 - y0;
  if (h <= 0) return;

  auto sy0 = ceil(y0);
  auto sy1t = floor(y1);

  // The `min` here is to account for the case that the quad isn't
  // even a full pixel tall.
  auto yt0 = min(y1, sy0) - y0;
  if (yt0 > 0) {
    auto [xt1, xt2] = ordered(x0, x0 + sL * yt0);
    auto [xt3, xt4] = ordered(x1, x1 + sR * yt0);
    asaRow(l, sy0 - 1, fade(c, yt0), xt1, xt2, xt3, xt4);
  }

  axisQuad(l, y0, x0, x1, sL, sR, sy0, sy1t, c);

  // Bottom row piece.
  auto ytt = y1 - max(sy1t, y0);
  if (ytt > 0) {
    auto [xt1, xt2] = ordered(x0 + sL * h, x0 + sL * (h - ytt));
    auto [xt3, xt4] = ordered(x1 + sR * h, x1 + sR * (h - ytt));
    asaRow(l, sy1t, fade(c, ytt), xt1, xt2, xt3, xt4);
  }
}


void axisAlignedTriangle(layer& l, float x, float y, float sL, float sR, unsigned int imin, unsigned int imax, color c) {
  auto ka = y, kb = y - 1.f, kc = ka, kd = kb;
  if (sL < 0) swap(ka, kb);
  if (sR < 0) swap(kc, kd);
  for (auto i = imin; i < imax; i += 1) {
    auto py = float(i);
    asaRow(l, i, c,
      x + (py - ka) * sL,
      x + (py - kb) * sL,
      x + (py - kc) * sR,
      x + (py - kd) * sR);
  }
}

void triangle(layer& l, point a, point b, point c, color col) {
  point v[3] {a, b, c};
  sort(v, v + 3, [](point a, point b){ return a.y < b.y; });
  auto [x0, y0] = v[0];
  auto [x1, y1] = v[1];
  auto [x2, y2] = v[2];

  // compute the ghost vertex
  float x3 = x0 + (y1 - y0) * (x2 - x0) / (y2 - y0);
  order(x1, x3);

  auto sy1t = floor(y1); // TODO: Middle row.
  auto sy1b = ceil(y1); // TODO: Middle row.

  auto h0 = y1 - y0;
  if (h0 > 0) {
    auto sy0 = ceil(y0);
    auto w0 = (x1 - x0) / h0;
    auto w1 = (x3 - x0) / h0;

    // The `min` here is to account for the case that the top triangle isn't
    // even a full pixel tall.
    auto yt0 = min(y1, sy0) - y0;
    if (yt0 > 0) {
      auto [xt1, xt2] = ordered(x0 + w0 * yt0, x0);
      auto [xt3, xt4] = ordered(x0, x0 + w1 * yt0);
      asaRow(l, sy0 - 1, fade(col, yt0), xt1, xt2, xt3, xt4);
    }
    axisAlignedTriangle(l, x0, y0, w0, w1, sy0, sy1t, col);

    // Middle row piece.
    auto ytt = y1 - max(y0, sy1t);
    if (ytt > 0) {
      auto [xt1, xt2] = ordered(x1, x1 - w0 * ytt);
      auto [xt3, xt4] = ordered(x3 - w1 * ytt, x3);
      asaRow(l, sy1t, fade(col, ytt), xt1, xt2, xt3, xt4);
    }
  }




  auto h1 = y2 - y1;
  if (h1 > 0) {
    auto sy2 = floor(y2);
    auto w2 = (x2 - x1) / h1;
    auto w3 = (x2 - x3) / h1;

    // Middle row piece.
    auto ytb = min(y2, sy1b) - y1;
    if (ytb > 0) {
      auto [xt1, xt2] = ordered(x1, x1 + w2 * ytb);
      auto [xt3, xt4] = ordered(x3 + w3 * ytb, x3);
      asaRow(l, sy1t, fade(col, ytb), xt1, xt2, xt3, xt4);
    }

    axisAlignedTriangle(l, x2, y2, w2, w3, sy1b, sy2, col);

    // The `max` here is to account for the case that the bottom triangle isn't
    // even a full pixel tall.
    auto yt0 = y2 - max(y1, sy2);
    if (yt0 > 0) {
      auto [xt1, xt2] = ordered(x2 - w2 * yt0, x2);
      auto [xt3, xt4] = ordered(x2, x2 - w3 * yt0);
      asaRow(l, sy2, fade(col, yt0), xt1, xt2, xt3, xt4);
    }
  }
}


void ellipse(ppm_image& im, float cx, float cy, float rx, float ry) {
  auto rowMin = int(ceil(cy - ry - .5f));
  auto rowMax = int(ceil(cy + ry - .5f));
  for (auto i = rowMin; i < rowMax; i += 1) {
    auto ty = (i + .5f - cy) / ry;
    auto sx = rx * sqrt(1 - ty * ty);
    auto rowMin = int(ceil(cx - sx - .5f));
    auto rowMax = int(ceil(cx + sx - .5f));
    for (auto j = rowMin; j < rowMax; j += 1) {
      fill_sample(im, j, i);
    }
  }
}

void circle(ppm_image& im, point c, float r) {
  ellipse(im, c.x, c.y, r, r);
}

void fill(ppm_image& im, color c) {
  for (auto i: range(im.height())) {
    for (auto j: range(im.width())) {
      fill_sample(im, j, i, c);
    }
  }
}

void squareLine(layer& l, float x0, float y0, float x1, float y1, float width) {
  auto half = .5f * width;

  auto Tx = x1 - x0;
  auto Ty = y1 - y0;
  auto T = 1.f / sqrt(Tx * Tx + Ty * Ty);
  Tx *= T;
  Ty *= T;

  auto A = Tx + Ty;
  auto B = Tx - Ty;
  auto x00 = x0 - half * A;
  auto y00 = y0 + half * B;
  auto x01 = x1 + half * B;
  auto y01 = y1 + half * A;
  auto x10 = x0 - half * B;
  auto y10 = y0 - half * A;
  auto x11 = x1 + half * A;
  auto y11 = y1 - half * B;

  triangle(l, {x00, y00}, {x01, y01}, {x10, y10}, 0x000000ff);
  triangle(l, {x01, y01}, {x10, y10}, {x11, y11}, 0x000000ff);
}

void buttLine(layer& l, float x0, float y0, float x1, float y1, float width, color c) {
  auto half = .5f * width;
  auto Tx = x1 - x0;
  auto Ty = y1 - y0;
  auto T = half / sqrt(Tx * Tx + Ty * Ty);
  Tx *= T;
  Ty *= T;
  auto x00 = x0 - Ty;
  auto y00 = y0 + Tx;
  auto x01 = x1 - Ty;
  auto y01 = y1 + Tx;
  auto x10 = x0 + Ty;
  auto y10 = y0 - Tx;
  auto x11 = x1 + Ty;
  auto y11 = y1 - Tx;
  triangle(l, {x00, y00}, {x01, y01}, {x10, y10}, c);
  triangle(l, {x01, y01}, {x10, y10}, {x11, y11}, c);
}

static inline void buttLine(layer& l, point p0, point p1, float width, color c) {
  buttLine(l, p0.x, p0.y, p1.x, p1.y, width, c);
}

void xAxis(layer& l, float x0, float x1, float y) {
  buttLine(l, x0, y, x1, y, 2.f, 0x000000ff);
  for (auto i: range(11)) {
    auto t = float(i) / 10.f;
    auto x = x0 + t * (x1 - x0);
    buttLine(l, x, y, x, y + 8.f, 2.f, 0x000000ff);
  }
}

void yAxis(layer& l, float y0, float y1, float x) {
  buttLine(l, x, y0, x, y1, 2.f, 0x000000ff);
  for (auto i: range(11)) {
    auto t = float(i) / 10.f;
    auto y = y0 + t * (y1 - y0);
    buttLine(l, x - 8.f, y, x, y, 2.f, 0x000000ff);
  }
}

constexpr float fh = 16;
constexpr float ft = .08 * fh;

void letterA(layer& l, float x, float y) {
  buttLine(l, x, y, x + .4f * fh, y - fh, ft, 0x000000ff);
  buttLine(l, x + .8f * fh, y, x + .4f * fh, y - fh, ft, 0x000000ff);
  buttLine(l, x + .1f * fh, y - .4f * fh, x + .7f * fh, y +-.4f * fh, ft, 0x000000ff);
}

void letterB(layer& l, float x, float y) {
  buttLine(l, x, y, x, y - fh, ft, 0x000000ff);

  buttLine(l, x, y, x + .4f * fh, y, ft, 0x000000ff);
  buttLine(l, x, y - .5f * fh, x + .4f * fh, y - .5f * fh, ft, 0x000000ff);
  buttLine(l, x, y - fh, x + .4f * fh, y - fh, ft, 0x000000ff);

  buttLine(l, x + .4f * fh, y, x + .6f * fh, y - .1f * fh, ft, 0x000000ff);
  buttLine(l, x + .4f * fh, y - .5f * fh, x + .6f * fh, y - .4f * fh, ft, 0x000000ff);
  buttLine(l, x + .4f * fh, y - .5f * fh, x + .6f * fh, y - .6f * fh, ft, 0x000000ff);
  buttLine(l, x + .4f * fh, y - fh, x + .6f * fh, y - .9f * fh, ft, 0x000000ff);


  buttLine(l, x + .6f * fh, y - .6f * fh, x + .6f * fh, y - .9f * fh, ft, 0x000000ff);
  buttLine(l, x + .6f * fh, y - .1f * fh, x + .6f * fh, y - .4f * fh, ft, 0x000000ff);
}

struct bezier3 {
  point a, b, c, d;
  constexpr point operator()(float t) const {
    auto ab = point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
    auto bc = point(b.x + t * (c.x - b.x), b.y + t * (c.y - b.y));
    auto cd = point(c.x + t * (d.x - c.x), c.y + t * (d.y - c.y));
    auto abc = point(ab.x + t * (bc.x - ab.x), ab.y + t * (bc.y - ab.y));
    auto bcd = point(bc.x + t * (cd.x - bc.x), bc.y + t * (cd.y - bc.y));
    auto abcd = point(abc.x + t * (bcd.x - abc.x), abc.y + t * (bcd.y - abc.y));
    return abcd;
  }
};

struct bezier2 {
  point a, b, c;
  constexpr point operator()(float t) const {
    auto ab = point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
    auto bc = point(b.x + t * (c.x - b.x), b.y + t * (c.y - b.y));
    auto abc = point(ab.x + t * (bc.x - ab.x), ab.y + t * (bc.y - ab.y));
    return abc;
  }
};

static inline float dist(point p0, point p1) {
  auto dx = p1.x - p0.x;
  auto dy = p1.y - p0.y;
  return sqrt(dx * dx + dy * dy);
}

static inline void bez(layer& l, point a, point b, point c, point d, color col) {
  auto lin_len = dist(a, b) + dist(b, c) + dist(c, d);
  auto n_seg = static_cast<unsigned int>(lin_len / 20.f);
  auto f = bezier3 {a, b, c, d};
  auto curr = a;
  for (auto i: range(1, n_seg + 1)) {
    auto t = float(i) / float(n_seg);
    auto next = f(t);
    buttLine(l, curr, next, 2.f, col);
    curr = next;
  }
}

struct shape_record {
  bool forward;
  int i;
  float x;
  float slope;
  constexpr shape_record(): forward(false), i(-1), x(0.f), slope(0.f) {}
  constexpr shape_record(bool forw, int i, float x, float slope): forward(forw), i(i), x(x), slope(slope) {}
};

void spreadTopPoint(const array<point>& shape, int top, vector<shape_record>& trails, int ins_pt) {
  auto i = trails.insert(trails.begin() + ins_pt, 2lu, shape_record());
  auto& first = *i;
  auto& second = *(i + 1);

  auto prev = (top + shape.size() - 1) % shape.size();
  auto next = (top + 1) % shape.size();

  auto hn = shape[next].y - shape[top].y;
  auto hp = shape[prev].y - shape[top].y;
  auto sn = (shape[next].x - shape[top].x) / (hn ? hn : 1.f);
  auto sp = (shape[prev].x - shape[top].x) / (hp ? hp : 1.f);

  if (sp < sn) {
    first  = shape_record(false, top, shape[top].x, sp);
    second = shape_record(true, top, shape[top].x, sn);
  } else {
    first  = shape_record(true, top, shape[top].x, sn);
    second = shape_record(false, top, shape[top].x, sp);
  }
}

int main() {
  constexpr auto dimx = 640;
  constexpr auto dimy = 480;
  auto im = ppm_image("first.ppm", dimx, dimy);

  auto top = 0.f;
  auto left = 0.f;
  auto bottom = dimy;
  auto right = dimx;

  fill(im, 0xffffffff);
  // triangle(im, {0, 0}, {101, 100}, {0, 100});
  // ellipse(im, 150, 150, 90, 20);
  // ellipse(im, 150, 150, 20, 90);

  point a {100, 50};
  point b {232, 210};
  point c {57, 238};
  point d {210, 270};

  // circle(im, a, 3);
  // circle(im, b, 3);
  // circle(im, c, 3);

  // Create a layer for rendering the curve.
  auto L = layer(dimy, dimx);

  xAxis(L, left + 50, right - 50, bottom - 50);
  yAxis(L, top + 50, bottom - 50, left + 50);

  {
    auto left0 = left + 50.f;
    auto right0 = right - 50.f;
    auto top0 = top + 50.f;
    auto bottom0 = bottom - 50.f;
    auto mid0 = .5f * (top0 + bottom0);
    auto scale = -.5f * (bottom0 - mid0);
    for (auto i: range(100)) {
      auto t0 = float(i) / 100.f;
      auto t1 = float(i + 1) / 100.f;
      auto dx0 = t0 * 12.4f;
      auto dx1 = t1 * 12.4f;
      auto dy0 = sinf(dx0);
      auto dy1 = sinf(dx1);
      auto x0 = left0 + t0 * (right0 - left0);
      auto x1 = left0 + t1 * (right0 - left0);
      auto y0 = mid0 + scale * dy0;
      auto y1 = mid0 + scale * dy1;
      buttLine(L, x0, y0, x1, y1, 4.f, 0xff0000ff);
    }
  }

  letterA(L, 100, 100);
  letterB(L, 120, 100);

  {
    mt19937 rng;
    color col {0x4400ccff};
    rng.seed(system_clock::now().time_since_epoch().count());

    array<point> shape;
    array<int> ind;
    constexpr auto npts = 100;
    for (auto i: range(npts)) {
      auto t = 6.28f * i / npts;
      shape.push(300 + 100 * cos(t), 100 - 40 * sin(t) + 40 * cos(2 * t) );
      ind.push(i);
    }

    sort(ind.begin(), ind.end(), [&shape](int i, int j){ return shape[i].y < shape[j].y; });

    vector<shape_record> trails;

    auto top = ind[0];
    auto y0 = shape[top].y;
    spreadTopPoint(shape, top, trails, 0);

    for (auto i: range(1, ind.size())) {
      float y1 = shape[ind[i]].y;

      // Draw all the current segments.
      for (auto j: range(0, trails.size(), 2)) {
        axisTrap(L, y0, y1, trails[j].x, trails[j+1].x, trails[j].slope, trails[j+1].slope, col);
        trails[j].x += trails[j].slope * (y1 - y0);
        trails[j+1].x += trails[j+1].slope * (y1 - y0);
      }

      auto p = ind[i]; // new point index.
    
      // If new point was anticipated, it just updates state. Update state.
      auto pprev = (p + shape.size() - 1) % shape.size();
      auto pnext = (p + 1) % shape.size();

      auto pti = std::find_if(trails.begin(), trails.end(), [pprev](shape_record& r){ return r.forward && r.i == pprev; });
      auto nti = std::find_if(trails.begin(), trails.end(), [pnext](shape_record& r){ return !r.forward && r.i == pnext; });

      if (pti != trails.end() && nti != trails.end()) {
        // If both our neighbors have trails, remove them both.
        assert(abs(nti - pti) == 1);
        auto mi = min(pti, nti);
        trails.erase(mi, mi + 2);
      } else if (pti != trails.end()) {
        auto h = shape[pnext].y - shape[p].y;
        auto s = (shape[pnext].x - shape[p].x) / (h ? h : 1.f);
        pti->i = p;
        pti->x = shape[p].x;
        pti->slope = s;
      } else if (nti != trails.end()) {
        auto h = shape[pprev].y - shape[p].y;
        auto s = (shape[pprev].x - shape[p].x) / (h ? h : 1.f);
        nti->i = p;
        nti->x = shape[p].x;
        nti->slope = s;
      } else {
        // Search through which segment it is in between.
        auto sr = shape_record(false, -1, shape[p].x, 0.f);
        auto ii = std::lower_bound(trails.begin(), trails.end(), sr, [](const shape_record& a, const shape_record& b){
          return a.x < b.x;
        });
        spreadTopPoint(shape, p, trails, ii - trails.begin());
      }

      // Move down.
      y0 = y1;
    }
  }

  // Let's try some bezier stuff.
  {
  mt19937 rng;
  rng.seed(system_clock::now().time_since_epoch().count());
  auto a = point(rng() % dimx, rng() % dimy);
  auto b = point(rng() % dimx, rng() % dimy);
  auto c = point(rng() % dimx, rng() % dimy);
  auto d = point(rng() % dimx, rng() % dimy);

  circle(im, a, 5);
  circle(im, b, 5);
  circle(im, c, 5);
  circle(im, d, 5);
  buttLine(L, a, b, 1, 0x00000088);
  buttLine(L, c, d, 1, 0x00000088);
  bez(L, a, b, c, d, 0x008800ff);
  }

  // Copy layer to output.
  for (auto i: range(dimy)) {
    for (auto j: range(dimx)) {
      fill_sample(im, j, i, col(L(i, j)));
    }
  }
}