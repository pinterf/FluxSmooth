#ifndef __FLUXSMOOTH_H__
#define __FLUXSMOOTH_H__

#include "avisynth.h"
#include "stdint.h"
#include "emmintrin.h"
#include <cassert>
#include <memory.h>
#include <algorithm>

/************************************
// AVX512 enabler switch!!!
************************************/
#define FLUXSMOOTH_AVX512_ENABLED

// disable AVX512 for MSVC, leave if for others like clang
#if !defined(__clang__)
// as of April 2019, MSVC version of immintrin.h does not contain proper AVX512BW support (VS2017 15.9)
// clang (LLVM 8.0) is O.K.: c:\Program Files\LLVM\lib\clang\8.0.0\include
// MSVC is not O.K.: c:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\include
#undef FLUXSMOOTH_AVX512_ENABLED
#endif

/************************************
// Helpers, missing intrinsics
************************************/

// SSE4.1 simulation for SSE2
__forceinline __m128i _MM_BLENDV_EPI8(__m128i const &a, __m128i const &b, __m128i const &selector) {
  return _mm_or_si128(_mm_and_si128(selector, b), _mm_andnot_si128(selector, a));
}

// non-existant simd
__forceinline __m128i _MM_CMPLE_EPU16(__m128i x, __m128i y)
{
  // Returns 0xFFFF where x <= y:
  return _mm_cmpeq_epi16(_mm_subs_epu16(x, y), _mm_setzero_si128());
}

#define _mm_cmpge_epu8(a, b) \
    _mm_cmpeq_epi8(_mm_max_epu8(a, b), a)

#define _mm_cmple_epu8(a, b) _mm_cmpge_epu8(b, a)

// non-existant simd
__forceinline __m128i _mm_cmpgt_epu8(__m128i x, __m128i y)
{
  // Returns 0xFF where x > y:
  return _mm_andnot_si128(
    _mm_cmpeq_epi8(x, y),
    _mm_cmpeq_epi8(_mm_max_epu8(x, y), x)
  );
}

#define _mm_cmplt_epu8(a, b) _mm_cmpgt_epu8(b, a)

__forceinline __m128i _mm_cmpge_epi16(__m128i x, __m128i y)
{
  // Returns 0xFFFF where x >= y:
  return _mm_or_si128(_mm_cmpeq_epi16(x, y), _mm_cmpgt_epi16(x, y));
}

#define _mm_cmple_epi16(a, b) _mm_cmpge_epi16(b, a)

/************************************
// Helper
************************************/

__forceinline void check_neighbour_C(int neighbour, int center, int threshold, int &sum, int &cnt)
{
  if (std::abs(neighbour - center) <= threshold)
  {
    sum += neighbour;
    ++cnt;
  }
}

/************************************
// other constants
************************************/

// Optimizations by 'opt' parameter
enum { USE_OPT_C = 0, USE_OPT_SSE2 = 1, USE_OPT_SSE41 = 2, USE_OPT_AVX2 = 3, USE_OPT_AVX512 = 4};

constexpr int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
constexpr int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };

/************************************
// Prototypes, Temporal
************************************/
#ifdef FLUXSMOOTH_AVX512_ENABLED
void fluxT_avx512_uint16(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);

void fluxT_avx512(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);
#endif

void fluxT_avx2_uint16(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);

void fluxT_avx2(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);

void fluxT_sse41(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);

void fluxT_sse41_uint16(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);

void fluxT_sse2(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);

template<typename pixel_t>
void fluxT_C(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);

/************************************
// Prototypes, Spatial - Temporal
************************************/
#ifdef FLUXSMOOTH_AVX512_ENABLED
void fluxST_avx512_uint16(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);

void fluxST_avx512(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);
#endif

void fluxST_avx2_uint16(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);

void fluxST_avx2(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);

void fluxST_sse41(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);

void fluxST_sse41_uint16(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);

void fluxST_sse2(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);

template<typename pixel_t>
void fluxST_C(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);

/************************************
// Filter classes
************************************/

class FluxSmoothST: public GenericVideoFilter
{
  using proc_ST_t = void(*)(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch, 
    uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);

protected:
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment * env) override;

public:
  FluxSmoothST(PClip _child, int _temporal_threshold, int _spatial_threshold, bool _luma, bool _chroma, int _opt, IScriptEnvironment * env);

  // Auto register AVS+ mode: NICE filter
  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  int spatial_threshold;
  int temporal_threshold;
  bool processPlane[3];
  int opt;
  short scaletab[16];
  proc_ST_t proc_ST[3]; // for all planes
};

class FluxSmoothT : public GenericVideoFilter
{
  using proc_T_t = void(*)(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
    uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);

private:
  int temporal_threshold;
  bool processPlane[3];
  int opt;
  short scaletab[16]; // for C
  proc_T_t proc_T[3]; // for all planes

protected:
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

public:
  FluxSmoothT(PClip _child, int _temporal_threshold, bool _luma, bool _chroma, int _opt, IScriptEnvironment * env);

  // Auto register AVS+ mode: NICE filter
  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

};

#endif // #define __FLUXSMOOTH_H__

