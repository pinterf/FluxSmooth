#include "FluxSmooth.h"
#include <algorithm>
#include "stdint.h"
#include "immintrin.h" // also includes "zmmintrin.h" for AVX512 and "avx512bwintrin.h"

#ifdef FLUXSMOOTH_AVX512_ENABLED

#if !defined(__AVX512F__) || !defined(__AVX512BW__)
#if defined(_MSC_VER)
#error This source file will only work properly when compiled with AVX512F and AVX512BW option. Set /arch=AVX512 to command line options for this file.
#elseif defined(__clang__)
#error This source file will only work properly when compiled with AVX512F and AVX512BW option. Set -mavx512f -mavx512bw command line options for this file.
#else
#error Unsupported compiler. This source file will only work properly when compiled with AVX512F and AVX512BW option. Set ??? command line options for this file.
#endif
// BW: starting with Skylake X and Cannon Lake.
#endif

#if defined(_MSC_VER) && !defined(__clang__)
// As of April 2019, MS version of immintrin.h does not support AVX512BW _k*_mask* functions
// https://developercommunity.visualstudio.com/content/problem/518298/missing-avx512bw-mask-intrinsics.html

__forceinline __mmask64 _kand_mask64(__mmask64 A, __mmask64 B) // AVX512BW
{
  return (__mmask64)(A & B);
}

__forceinline __mmask64 _kor_mask64(__mmask64 A, __mmask64 B) // AVX512BW
{
  return (__mmask64)(A | B);
}

__forceinline __mmask32 _kand_mask32(__mmask32 A, __mmask32 B) // AVX512BW
{
  return (__mmask32)(A & B);
}

__forceinline __mmask32 _kor_mask32(__mmask32 A, __mmask32 B) // AVX512BW
{
  return (__mmask32)(A | B);
}

#endif


/************************************
// Helpers
************************************/

static __forceinline void check_neighbour_simd(__m512i &neighbour, __m512i &center, __m512i &threshold,
  __m512i &sum_lo, __m512i &sum_hi, __m512i &cnt)
{
  auto n_minus_c = _mm512_subs_epu8(neighbour, center); // AVX512BW
  auto c_minus_n = _mm512_subs_epu8(center, neighbour); // AVX512BW
  auto absdiff = _mm512_or_si512(n_minus_c, c_minus_n); // AVX512F
  auto abs_is_lessthanoreq_thresh = _mm512_cmple_epu8_mask(absdiff, threshold); // AVX512BW
  // count. add 1 (increment) when true.
  cnt = _mm512_add_epi8(cnt, _mm512_maskz_set1_epi8(abs_is_lessthanoreq_thresh, 1)); // AVX512BW
  // increase sum elements by neighbour where true
  // sum is 16 bits
  auto masked_neighbour = _mm512_maskz_mov_epi8(abs_is_lessthanoreq_thresh, neighbour); // AVX512BW
  auto zero = _mm512_setzero_si512(); // AVX512F
  auto masked_neighbour_lo = _mm512_unpacklo_epi8(masked_neighbour, zero);
  auto masked_neighbour_hi = _mm512_unpackhi_epi8(masked_neighbour, zero);
  sum_lo = _mm512_add_epi16(sum_lo, masked_neighbour_lo);
  sum_hi = _mm512_add_epi16(sum_hi, masked_neighbour_hi);

  /*
  if (std::abs(neighbour - center) <= threshold)
  {
    sum += neighbour;
    ++cnt;
  }
  */
}

static __forceinline void check_neighbour_simd_uint16(__m512i &neighbour, __m512i &center, __m512i &threshold,
  __m512i &sum_lo, __m512i &sum_hi, __m512i &cnt)
{
  auto n_minus_c = _mm512_subs_epu16(neighbour, center);
  auto c_minus_n = _mm512_subs_epu16(center, neighbour);
  auto absdiff = _mm512_or_si512(n_minus_c, c_minus_n);
  // absdiff <= threshold
  auto abs_is_lessthanoreq_thresh = _mm512_cmple_epu16_mask(absdiff, threshold);
  // count. add 1 (increment) when true.
  cnt = _mm512_add_epi16(cnt, _mm512_maskz_set1_epi16(abs_is_lessthanoreq_thresh, 1)); // AVX512BW
  // increase sum elements by neighbour where true, that is mask is FF
  // sum is 16 bits
  auto masked_neighbour = _mm512_maskz_mov_epi16(abs_is_lessthanoreq_thresh, neighbour); // AVX512BW
  auto zero = _mm512_setzero_si512();
  auto masked_neighbour_lo = _mm512_unpacklo_epi16(masked_neighbour, zero);
  auto masked_neighbour_hi = _mm512_unpackhi_epi16(masked_neighbour, zero);
  sum_lo = _mm512_add_epi32(sum_lo, masked_neighbour_lo);
  sum_hi = _mm512_add_epi32(sum_hi, masked_neighbour_hi);

  /*
  if (std::abs(neighbour - center) <= threshold)
  {
    sum += neighbour;
    ++cnt;
  }
  */
}

/************************************
// Temporal only AVX512, 8 bit
************************************/

static __forceinline void fluxT_core_avx512(const BYTE * currp, 
  const BYTE * prevp, const BYTE * nextp,
  BYTE * destp, int x,  
  __m512i &temporal_threshold_vector,
  __m512i &scaletab_lut_lsbs,
  __m512i &scaletab_lut_msbs
)
{
  auto b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x));
  auto pbt = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(prevp + x));
  auto nbt = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(nextp + x));
  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm512_cmpgt_epu8_mask(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm512_cmpgt_epu8_mask(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm512_cmpgt_epu8_mask(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm512_cmpgt_epu8_mask(nbt, b); // FF where nbt > b
  __mmask64 both_less =  _kand_mask64(pbt_lessthan_b, nbt_lessthan_b); // AVX512BW
  __mmask64 both_greater = _kand_mask64(pbt_greaterthan_b, nbt_greaterthan_b);
  __mmask64 mask_either_is_true = _kor_mask64(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

  // int sum = b, cnt = 1;
  auto zero = _mm512_setzero_si512();
  auto sum_lo = _mm512_unpacklo_epi8(b, zero);
  auto sum_hi = _mm512_unpackhi_epi8(b, zero);
  auto cnt = _mm512_set1_epi8(1);

  check_neighbour_simd(pbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(nbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  // (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);

  // factor1 = sum*2 + cnt, sum elements are 16 bits
  auto cnt_lo = _mm512_unpacklo_epi8(cnt, zero);
  auto cnt_hi = _mm512_unpackhi_epi8(cnt, zero);
  auto factor1_lo = _mm512_add_epi16(_mm512_add_epi16(sum_lo, sum_lo), cnt_lo);
  auto factor1_hi = _mm512_add_epi16(_mm512_add_epi16(sum_hi, sum_hi), cnt_hi);
  // factor2 = scaletab[cnt]
  auto factor2_lsb = _mm512_shuffle_epi8(scaletab_lut_lsbs, cnt);
  auto factor2_msb = _mm512_shuffle_epi8(scaletab_lut_msbs, cnt);
  auto factor2_lo = _mm512_unpacklo_epi8(factor2_lsb, factor2_msb);
  auto factor2_hi = _mm512_unpackhi_epi8(factor2_lsb, factor2_msb);
  // finally mul and shift
  auto mulres_lo = _mm512_mulhi_epi16(factor1_lo, factor2_lo); // upper 16 bit of mul result, no need for >> 16
  auto mulres_hi = _mm512_mulhi_epi16(factor1_hi, factor2_hi); // upper 16 bit of mul result, no need for >> 16
  // move back to 16x8 bits
  auto result = _mm512_packus_epi16(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm512_mask_mov_epi8(b, mask_either_is_true, result); // true: second param, false: 1st param

  _mm512_storeu_si512(reinterpret_cast<__m512i *>(destp + x), finalres);
}


// Temporal only
void fluxT_avx512(const uint8_t* currp, const int src_pitch,
  const uint8_t * prevp, const int prv_pitch,
  const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch,
  const int width, int height,
  int temporal_threshold,
  short *scaletab)
{
  __m512i scaletab_lut_lsbs;
  __m512i scaletab_lut_msbs;
  for (int i = 0; i < 16; i++) {
    ((uint8_t*)&scaletab_lut_lsbs)[i] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i] = (scaletab[i] >> 8) & 0xFF;
    // same for hi 128
    ((uint8_t*)&scaletab_lut_lsbs)[i + 16] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i + 16] = (scaletab[i] >> 8) & 0xFF;
    // same for hilo 128
    ((uint8_t*)&scaletab_lut_lsbs)[i + 16*2] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i + 16*2] = (scaletab[i] >> 8) & 0xFF;
    // same for hihi 128
    ((uint8_t*)&scaletab_lut_lsbs)[i + 16*3] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i + 16*3] = (scaletab[i] >> 8) & 0xFF;
  }

  const int xcnt = width;

  __m512i temporal_threshold_vector = _mm512_set1_epi8(temporal_threshold);

  const int wmod64 = xcnt / 64 * 64;
  const int rest = xcnt - wmod64;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < wmod64; x += 64)
      fluxT_core_avx512(currp, prevp, nextp, destp, x, temporal_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);
    // do rest
    if (rest > 0)
      fluxT_core_avx512(currp, prevp, nextp, destp, xcnt - 64, temporal_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
  //_mm512_zeroupper();
}

/************************************
// Temporal only AVX512, 16 bit
************************************/

__forceinline void fluxT_core_avx512_uint16(const uint8_t * currp, const uint8_t* prevp, const uint8_t *nextp, uint8_t *destp, int x,
  __m512i &temporal_threshold_vector
)
{
  auto b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x));
  auto pbt = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(prevp + x));
  auto nbt = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(nextp + x));

  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm512_cmpgt_epu16_mask(b, pbt); // 1 where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm512_cmpgt_epu16_mask(b, nbt); // 1 where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm512_cmpgt_epu16_mask(pbt, b); // 1 where pbt > b
  auto nbt_greaterthan_b = _mm512_cmpgt_epu16_mask(nbt, b); // 1 where nbt > b
  __mmask32 both_less = _kand_mask32(pbt_lessthan_b, nbt_lessthan_b);
  __mmask32 both_greater = _kand_mask32(pbt_greaterthan_b, nbt_greaterthan_b);
  __mmask32 mask_either_is_true = _kor_mask32(both_less, both_greater);
  // mask will be used at the final decision. Where 1: keep computed result. 0: keep original pixel (dst=curr)

  // int sum = b, cnt = 1;
  auto zero = _mm512_setzero_si512();
  auto sum_lo = _mm512_unpacklo_epi16(b, zero);
  auto sum_hi = _mm512_unpackhi_epi16(b, zero);
  auto cnt = _mm512_set1_epi16(1);

  check_neighbour_simd_uint16(pbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd_uint16(nbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  // (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);

  auto cnt_lo = _mm512_unpacklo_epi16(cnt, zero);
  auto cnt_hi = _mm512_unpackhi_epi16(cnt, zero);
  // Difference from SSE4.1 and C: floating point division
  // sum / count -> (int)((float)sum * 1.0f/(float)count + 0.5f)
  const auto rounder_half = _mm512_set1_ps(0.5f);
  // lower 16 pixels
  auto fcnt_lo = _mm512_cvtepi32_ps(cnt_lo);
  auto fsum_lo = _mm512_cvtepi32_ps(sum_lo);
  // difference from AVX2 or less: rcp14_ps has error less than 2^-14, while rcp_ps error is < 1.5*2^-12
  auto mulres_lo = _mm512_cvttps_epi32(_mm512_fmadd_ps(fsum_lo, _mm512_rcp14_ps(fcnt_lo), rounder_half));
  // upper 16 pixels
  auto fcnt_hi = _mm512_cvtepi32_ps(cnt_hi);
  auto fsum_hi = _mm512_cvtepi32_ps(sum_hi);
  auto mulres_hi = _mm512_cvttps_epi32(_mm512_fmadd_ps(fsum_hi, _mm512_rcp14_ps(fcnt_hi), rounder_half));

  // move back to 32x16 bits
  auto result = _mm512_packus_epi32(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm512_mask_mov_epi16(b, mask_either_is_true, result); // true: second param, false: 1st param

  _mm512_storeu_si512(reinterpret_cast<__m512i *>(destp + x), finalres);
}

// Temporal only
void fluxT_avx512_uint16(const uint8_t* currp, const int src_pitch,
  const uint8_t * prevp, const int prv_pitch,
  const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch,
  const int width, int height,
  int temporal_threshold,
  short *scaletab)
{
  const int xcnt = width;

  __m512i temporal_threshold_vector = _mm512_set1_epi16(temporal_threshold);

  const int wmod32 = xcnt / 32 * 32;
  const int rest = xcnt - wmod32;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < wmod32; x += 32)
      fluxT_core_avx512_uint16(currp, prevp, nextp, destp, x * sizeof(uint16_t), temporal_threshold_vector);
    // do rest
    if (rest > 0)
      fluxT_core_avx512_uint16(currp, prevp, nextp, destp, (xcnt - 32) * sizeof(uint16_t), temporal_threshold_vector);

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
  //_mm512_zeroupper();
}

/************************************
// Spatial Temporal AVX2, 8 bit
************************************/

__forceinline void fluxST_core_avx512(const BYTE * currp, const int src_pitch,
  const BYTE * prevp, const BYTE * nextp,
  BYTE * destp, int x,
  __m512i &temporal_threshold_vector,
  __m512i &spatial_threshold_vector,
  __m512i &scaletab_lut_lsbs,
  __m512i &scaletab_lut_msbs
)
{
  // +1: center of 3x3 pixels [+0,+1,+2]
  auto b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + 1));
  auto pbt = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(prevp + x + 1));
  auto nbt = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(nextp + x + 1));

  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm512_cmpgt_epu8_mask(b, pbt); // 1 where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm512_cmpgt_epu8_mask(b, nbt); // 1 where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm512_cmpgt_epu8_mask(pbt, b); // 1 where pbt > b
  auto nbt_greaterthan_b = _mm512_cmpgt_epu8_mask(nbt, b); // 1 where nbt > b
  __mmask64 both_less = _kand_mask64(pbt_lessthan_b, nbt_lessthan_b);
  __mmask64 both_greater = _kand_mask64(pbt_greaterthan_b, nbt_greaterthan_b);
  __mmask64 mask_either_is_true = _kor_mask64(both_less, both_greater);
  // mask will be used at the final decision. Where 1: keep computed result. 0: keep original pixel (dst=curr)

    // int pb1 = currp[-src_pitch - 1], pb2 = currp[-src_pitch], pb3 = currp[-src_pitch + 1];
    // int b1 = currp[-1], /*b = currp[0], */b2 = currp[1];
    // int nb1 = currp[src_pitch - 1], nb2 = currp[src_pitch], nb3 = currp[src_pitch + 1];

  auto pb1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x - src_pitch + 0));
  auto pb2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x - src_pitch + 1));
  auto pb3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x - src_pitch + 2));

  auto b1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + 0));
  auto b2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + 2));

  auto nb1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + src_pitch + 0));
  auto nb2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + src_pitch + 1));
  auto nb3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + src_pitch + 2));

  // int sum = b, cnt = 1;
  auto zero = _mm512_setzero_si512();
  auto sum_lo = _mm512_unpacklo_epi8(b, zero);
  auto sum_hi = _mm512_unpackhi_epi8(b, zero);
  auto cnt = _mm512_set1_epi8(1);

  check_neighbour_simd(pbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(nbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(pb1, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(pb2, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(pb3, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(b1, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(b2, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(nb1, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(nb2, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(nb3, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  // (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);

  // factor1 = sum*2 + cnt, sum elements are 16 bits
  auto cnt_lo = _mm512_unpacklo_epi8(cnt, zero);
  auto cnt_hi = _mm512_unpackhi_epi8(cnt, zero);
  auto factor1_lo = _mm512_add_epi16(_mm512_add_epi16(sum_lo, sum_lo), cnt_lo);
  auto factor1_hi = _mm512_add_epi16(_mm512_add_epi16(sum_hi, sum_hi), cnt_hi);
  // factor2 = scaletab[cnt]
  auto factor2_lsb = _mm512_shuffle_epi8(scaletab_lut_lsbs, cnt);
  auto factor2_msb = _mm512_shuffle_epi8(scaletab_lut_msbs, cnt);
  auto factor2_lo = _mm512_unpacklo_epi8(factor2_lsb, factor2_msb);
  auto factor2_hi = _mm512_unpackhi_epi8(factor2_lsb, factor2_msb);
  // finally mul and shift
  auto mulres_lo = _mm512_mulhi_epi16(factor1_lo, factor2_lo); // upper 16 bit of mul result, no need for >> 16
  auto mulres_hi = _mm512_mulhi_epi16(factor1_hi, factor2_hi); // upper 16 bit of mul result, no need for >> 16
  // move back to 16x8 bits
  auto result = _mm512_packus_epi16(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm512_mask_mov_epi8(b, mask_either_is_true, result); // true: second param, false: 1st param

  _mm512_storeu_si512(reinterpret_cast<__m512i *>(destp + x + 1), finalres);
}

// Spatial Temporal
void fluxST_avx512(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, int height, int temporal_threshold, int spatial_threshold, short *scaletab)
{
  __m512i scaletab_lut_lsbs;
  __m512i scaletab_lut_msbs;
  for (int i = 0; i < 16; i++) {
    ((uint8_t*)&scaletab_lut_lsbs)[i] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i] = (scaletab[i] >> 8) & 0xFF;
    // same for upper 128
    ((uint8_t*)&scaletab_lut_lsbs)[i + 16] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i + 16] = (scaletab[i] >> 8) & 0xFF;
    // same for upper 2*128
    ((uint8_t*)&scaletab_lut_lsbs)[i + 2 * 16] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i + 2 * 16] = (scaletab[i] >> 8) & 0xFF;
    // same for upper 3*128
    ((uint8_t*)&scaletab_lut_lsbs)[i + 3 * 16] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i + 3 * 16] = (scaletab[i] >> 8) & 0xFF;
  }

  // spatial: because of previous and next line involved, function is called 
  // starting with the 2nd line and with height = (real_height - 2) 
  const int xcnt = width - 2; // leftmost/rightmost column safety

  __m512i temporal_threshold_vector = _mm512_set1_epi8(temporal_threshold);
  __m512i spatial_threshold_vector = _mm512_set1_epi8(spatial_threshold);

  const int wmod64 = xcnt / 64 * 64;
  const int rest = xcnt - wmod64;

  for (int y = 0; y < height; y++)
  {
    destp[0] = currp[0]; // Copy left edge

    for (int x = 0; x < wmod64; x += 64)
      fluxST_core_avx512(currp, src_pitch, prevp, nextp, destp, x, temporal_threshold_vector, spatial_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);
    // do rest
    if (rest > 0)
      fluxST_core_avx512(currp, src_pitch, prevp, nextp, destp, xcnt - 64, temporal_threshold_vector, spatial_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);

    destp[width - 1] = currp[width - 1]; // Copy right edge

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
  //_mm512_zeroupper();
}

/************************************
// Spatial Temporal AVX2, 16 bit
************************************/
__forceinline void fluxST_core_avx512_uint16(const uint8_t * currp, const int src_pitch, const uint8_t* prevp, const uint8_t *nextp, uint8_t *destp, int x,
  __m512i &temporal_threshold_vector,
  __m512i &spatial_threshold_vector
)
{
  // +1: center of 3x3 pixels [+0,+1,+2]
  auto b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + 1 * sizeof(uint16_t)));
  auto pbt = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(prevp + x + 1 * sizeof(uint16_t)));
  auto nbt = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(nextp + x + 1 * sizeof(uint16_t)));

  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm512_cmpgt_epu16_mask(b, pbt); // 1 where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm512_cmpgt_epu16_mask(b, nbt); // 1 where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm512_cmpgt_epu16_mask(pbt, b); // 1 where pbt > b
  auto nbt_greaterthan_b = _mm512_cmpgt_epu16_mask(nbt, b); // 1 where nbt > b
  __mmask32 both_less = _kand_mask32(pbt_lessthan_b, nbt_lessthan_b);
  __mmask32 both_greater = _kand_mask32(pbt_greaterthan_b, nbt_greaterthan_b);
  __mmask32 mask_either_is_true = _kor_mask32(both_less, both_greater);
  // mask will be used at the final decision. Where 1: keep computed result. 00: keep original pixel (dst=curr)

    // int pb1 = currp[-src_pitch - 1], pb2 = currp[-src_pitch], pb3 = currp[-src_pitch + 1];
    // int b1 = currp[-1], /*b = currp[0], */b2 = currp[1];
    // int nb1 = currp[src_pitch - 1], nb2 = currp[src_pitch], nb3 = currp[src_pitch + 1];

  auto pb1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x - src_pitch + 0 * sizeof(uint16_t)));
  auto pb2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x - src_pitch + 1 * sizeof(uint16_t)));
  auto pb3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x - src_pitch + 2 * sizeof(uint16_t)));

  auto b1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + 0 * sizeof(uint16_t)));
  auto b2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + 2 * sizeof(uint16_t)));

  auto nb1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + src_pitch + 0 * sizeof(uint16_t)));
  auto nb2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + src_pitch + 1 * sizeof(uint16_t)));
  auto nb3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(currp + x + src_pitch + 2 * sizeof(uint16_t)));

  // int sum = b, cnt = 1;
  auto zero = _mm512_setzero_si512();
  auto sum_lo = _mm512_unpacklo_epi16(b, zero);
  auto sum_hi = _mm512_unpackhi_epi16(b, zero);
  auto cnt = _mm512_set1_epi16(1);

  check_neighbour_simd_uint16(pbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd_uint16(nbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd_uint16(pb1, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd_uint16(pb2, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd_uint16(pb3, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd_uint16(b1, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd_uint16(b2, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd_uint16(nb1, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd_uint16(nb2, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd_uint16(nb3, b, spatial_threshold_vector, sum_lo, sum_hi, cnt);
  // (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);

  auto cnt_lo = _mm512_unpacklo_epi16(cnt, zero);
  auto cnt_hi = _mm512_unpackhi_epi16(cnt, zero);
  // Difference from SSE4.1 and C: floating point division
  // sum / count -> (int)((float)sum * 1.0f/(float)count + 0.5f)
  const auto rounder_half = _mm512_set1_ps(0.5f);
  // lower 16 pixels
  auto fcnt_lo = _mm512_cvtepi32_ps(cnt_lo);
  auto fsum_lo = _mm512_cvtepi32_ps(sum_lo);
  // difference from AVX2 or less: rcp14_ps has error less than 2^-14, while rcp_ps error is < 1.5*2^-12
  auto mulres_lo = _mm512_cvttps_epi32(_mm512_fmadd_ps(fsum_lo, _mm512_rcp14_ps(fcnt_lo), rounder_half));
  // upper 16 pixels
  auto fcnt_hi = _mm512_cvtepi32_ps(cnt_hi);
  auto fsum_hi = _mm512_cvtepi32_ps(sum_hi);
  auto mulres_hi = _mm512_cvttps_epi32(_mm512_fmadd_ps(fsum_hi, _mm512_rcp14_ps(fcnt_hi), rounder_half));

  // move back to 32x16 bits
  auto result = _mm512_packus_epi32(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm512_mask_mov_epi16(b, mask_either_is_true, result); // true: second param, false: 1st param

  _mm512_storeu_si512(reinterpret_cast<__m512i *>(destp + x + 1 * sizeof(uint16_t)), finalres);
}

// Spatial Temporal
void fluxST_avx512_uint16(const uint8_t* currp, const int src_pitch,
  const uint8_t * prevp, const int prv_pitch,
  const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch,
  const int width, int height,
  int temporal_threshold,
  int spatial_threshold,
  short *scaletab)
{

  // spatial: because of previous and next line involved, function is called 
  // starting with the 2nd line and with height = (real_height - 2) 
  const int xcnt = width - 2; // leftmost/rightmost column safety

  __m512i temporal_threshold_vector = _mm512_set1_epi16(temporal_threshold);
  __m512i spatial_threshold_vector = _mm512_set1_epi16(spatial_threshold);

  const int wmod32 = xcnt / 32 * 32;
  const int rest = xcnt - wmod32;

  for (int y = 0; y < height; y++)
  {
    reinterpret_cast<uint16_t*>(destp)[0] = reinterpret_cast<const uint16_t*>(currp)[0]; // Copy left edge

    for (int x = 0; x < wmod32; x += 32)
      fluxST_core_avx512_uint16(currp, src_pitch, prevp, nextp, destp, x * sizeof(uint16_t), temporal_threshold_vector, spatial_threshold_vector);
    // do rest
    if (rest > 0)
      fluxST_core_avx512_uint16(currp, src_pitch, prevp, nextp, destp, (xcnt - 32) * sizeof(uint16_t), temporal_threshold_vector, spatial_threshold_vector);

    reinterpret_cast<uint16_t*>(destp)[width - 1] = reinterpret_cast<const uint16_t*>(currp)[width - 1]; // Copy right edge

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
  //_mm512_zeroupper();
}

#endif // FLUXSMOOTH_AVX512_ENABLED
