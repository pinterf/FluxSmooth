// FluxSmooth
// Avisynth filter for spatio-temporal smoothing of fluctuations
//
// By Ross Thomas <ross@grinfinity.com>
//
// There is no copyright on this code, and there are no conditions
// on its distribution or use. Do with it what you will.

#include "FluxSmooth.h"
#include <algorithm>
#include "stdint.h"
#include "emmintrin.h" // SSE2
#include "immintrin.h" // SSSE3
#include "smmintrin.h" // SSE4.1

/************************************
// Helpers
************************************/

__forceinline void check_neighbour_simd(__m128i &neighbour, __m128i &center, __m128i &threshold,
  __m128i &sum_lo, __m128i &sum_hi, __m128i &cnt)
{
  auto n_minus_c = _mm_subs_epu8(neighbour, center);
  auto c_minus_n = _mm_subs_epu8(center, neighbour);
  auto absdiff = _mm_or_si128(n_minus_c, c_minus_n);
  auto abs_is_lessthanoreq_thresh = _mm_cmple_epu8(absdiff, threshold);
  // count. increment when true. We simply sub the mask value 00 (0) or FF (-1)
  cnt = _mm_sub_epi8(cnt, abs_is_lessthanoreq_thresh);
  // increase sum elements by neighbour where true, that is mask is FF
  // sum is 16 bits
  auto masked_neighbour = _mm_and_si128(abs_is_lessthanoreq_thresh, neighbour);
  auto zero = _mm_setzero_si128();
  auto masked_neighbour_lo = _mm_unpacklo_epi8(masked_neighbour, zero);
  auto masked_neighbour_hi = _mm_unpackhi_epi8(masked_neighbour, zero);
  sum_lo = _mm_add_epi16(sum_lo, masked_neighbour_lo);
  sum_hi = _mm_add_epi16(sum_hi, masked_neighbour_hi);

  /*
  if (std::abs(neighbour - center) <= threshold)
  {
    sum += neighbour;
    ++cnt;
  }
  */
}

__forceinline void check_neighbour_simd_uint16(__m128i &neighbour, __m128i &center, __m128i &threshold,
  __m128i &sum_lo, __m128i &sum_hi, __m128i &cnt, const __m128i &make_signed_word)
{
  // threshold is shifted to the "signed" int16 domain
  auto n_minus_c = _mm_subs_epu16(neighbour, center);
  auto c_minus_n = _mm_subs_epu16(center, neighbour);
  auto absdiff = _mm_or_si128(n_minus_c, c_minus_n);
  // absdiff <= threshold ==> !(absdiff > threshold)
  // FIXME make it a bit faster: cmpgt and later: andnot, and count in a reverse way (instead of increase-when-match use decrease-by-non-match)
  auto abs_is_lessthanoreq_thresh = _mm_cmple_epi16(_mm_add_epi16(absdiff, make_signed_word), threshold); 
  // count. increment when true. We simply sub the mask value 0000 (0) or FFFF (-1)
  cnt = _mm_sub_epi16(cnt, abs_is_lessthanoreq_thresh);
  // increase sum elements by neighbour where true, that is mask is FF
  // sum is 16 bits
  auto masked_neighbour = _mm_and_si128(abs_is_lessthanoreq_thresh, neighbour);
  auto zero = _mm_setzero_si128();
  auto masked_neighbour_lo = _mm_unpacklo_epi16(masked_neighbour, zero);
  auto masked_neighbour_hi = _mm_unpackhi_epi16(masked_neighbour, zero);
  sum_lo = _mm_add_epi32(sum_lo, masked_neighbour_lo);
  sum_hi = _mm_add_epi32(sum_hi, masked_neighbour_hi);

  /*
  if (std::abs(neighbour - center) <= threshold)
  {
    sum += neighbour;
    ++cnt;
  }
  */
}

/************************************
// Temporal only SSE2, 8 bit
************************************/
__forceinline void fluxT_core_sse2(const uint8_t * currp, const uint8_t* prevp, const uint8_t *nextp, uint8_t *destp, int x,
  __m128i &temporal_threshold_vector,
  __m128i &scaletab_lut_lsbs,
  __m128i &scaletab_lut_msbs
)
{
  auto b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x));
  auto pbt = _mm_loadu_si128(reinterpret_cast<const __m128i*>(prevp + x));
  auto nbt = _mm_loadu_si128(reinterpret_cast<const __m128i*>(nextp + x));
  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm_cmpgt_epu8(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm_cmpgt_epu8(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm_cmpgt_epu8(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm_cmpgt_epu8(nbt, b); // FF where nbt > b
  auto both_less = _mm_and_si128(pbt_lessthan_b, nbt_lessthan_b);
  auto both_greater = _mm_and_si128(pbt_greaterthan_b, nbt_greaterthan_b);
  auto mask_either_is_true = _mm_or_si128(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

  // int sum = b, cnt = 1;
  auto zero = _mm_setzero_si128();
  auto sum_lo = _mm_unpacklo_epi8(b, zero);
  auto sum_hi = _mm_unpackhi_epi8(b, zero);
  auto cnt = _mm_set1_epi8(1);

  check_neighbour_simd(pbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(nbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  // (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);

  // factor1 = sum*2 + cnt, sum elements are 16 bits
  auto cnt_lo = _mm_unpacklo_epi8(cnt, zero);
  auto cnt_hi = _mm_unpackhi_epi8(cnt, zero);

  // Difference from SSE4.1 and C: floating point division
  // SSE2: sum / count -> (int)((float)sum * 1.0f/(float)count + 0.5f)
  const auto rounder_half = _mm_set1_ps(0.5f);
  // lower 8 pixels
  auto fcnt_lo_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(cnt_lo, zero));
  auto fcnt_lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(cnt_lo, zero));
  auto fsum_lo_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(sum_lo, zero));
  auto fsum_lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(sum_lo, zero));

  auto mul_lo_lo = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_lo_lo, _mm_rcp_ps(fcnt_lo_lo)), rounder_half));
  auto mul_lo_hi = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_lo_hi, _mm_rcp_ps(fcnt_lo_hi)), rounder_half));
  auto mulres_lo = _mm_packs_epi32(mul_lo_lo, mul_lo_hi);
  // upper 8 pixels
  auto fcnt_hi_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(cnt_hi, zero));
  auto fcnt_hi_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(cnt_hi, zero));
  auto fsum_hi_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(sum_hi, zero));
  auto fsum_hi_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(sum_hi, zero));

  auto mul_hi_lo = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_hi_lo, _mm_rcp_ps(fcnt_hi_lo)), rounder_half));
  auto mul_hi_hi = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_hi_hi, _mm_rcp_ps(fcnt_hi_hi)), rounder_half));
  auto mulres_hi = _mm_packs_epi32(mul_hi_lo, mul_hi_hi);

  // move back to 16x8 bits
  auto result = _mm_packus_epi16(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _MM_BLENDV_EPI8(b, result, mask_either_is_true); // true: second param, false: 1st param

  _mm_storeu_si128(reinterpret_cast<__m128i *>(destp + x), finalres);
}


// Temporal only
void fluxT_sse2(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab)
{
  __m128i scaletab_lut_lsbs;
  __m128i scaletab_lut_msbs;
  for (int i = 0; i < 16; i++) {
    ((uint8_t*)&scaletab_lut_lsbs)[i] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i] = (scaletab[i] >> 8) & 0xFF;
  }

  const int xcnt = width;

  __m128i temporal_threshold_vector = _mm_set1_epi8(temporal_threshold);

  const int wmod16 = xcnt / 16 * 16;
  const int rest = xcnt - wmod16;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < wmod16; x += 16)
      fluxT_core_sse2(currp, prevp, nextp, destp, x, temporal_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);
    // do rest
    if (rest > 0)
      fluxT_core_sse2(currp, prevp, nextp, destp, xcnt - 16, temporal_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
}

/************************************
// Temporal only SSE4.1, 8 bit
************************************/
#ifdef __clang__
__attribute__((__target__("sse4.1")))
#endif
__forceinline void fluxT_core_sse41(const uint8_t * currp, const uint8_t* prevp, const uint8_t *nextp, uint8_t *destp, int x,
  __m128i &temporal_threshold_vector,
  __m128i &scaletab_lut_lsbs,
  __m128i &scaletab_lut_msbs
)
{

  auto b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x));
  auto pbt = _mm_loadu_si128(reinterpret_cast<const __m128i*>(prevp + x));
  auto nbt = _mm_loadu_si128(reinterpret_cast<const __m128i*>(nextp + x));
  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm_cmpgt_epu8(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm_cmpgt_epu8(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm_cmpgt_epu8(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm_cmpgt_epu8(nbt, b); // FF where nbt > b
  auto both_less = _mm_and_si128(pbt_lessthan_b, nbt_lessthan_b);
  auto both_greater = _mm_and_si128(pbt_greaterthan_b, nbt_greaterthan_b);
  auto mask_either_is_true = _mm_or_si128(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

  // int sum = b, cnt = 1;
  auto zero = _mm_setzero_si128();
  auto sum_lo = _mm_unpacklo_epi8(b, zero);
  auto sum_hi = _mm_unpackhi_epi8(b, zero);
  auto cnt = _mm_set1_epi8(1);

  check_neighbour_simd(pbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(nbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
    // (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);

#if 0
  // Experiment with MADD and rounding: a bit slower, the same result
  // (sum      * scaletab) + 
  // (rounding *        1) in one step
  // >> 15: 2nd step
  auto one = _mm_set1_epi16(1);
  constexpr int FACTOR_BITS = 15;
  auto rounding = _mm_set1_epi16(1 << (FACTOR_BITS - 1));

  // factor2 = scaletab[cnt]
  auto factor2_lsb = _mm_shuffle_epi8(scaletab_lut_lsbs, cnt);
  auto factor2_msb = _mm_shuffle_epi8(scaletab_lut_msbs, cnt);
  auto factor2_lo = _mm_unpacklo_epi8(factor2_lsb, factor2_msb);
  auto factor2_hi = _mm_unpackhi_epi8(factor2_lsb, factor2_msb);

  auto mulres_lo_lo = _mm_srai_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(sum_lo, rounding), _mm_unpacklo_epi16(factor2_lo, one)), FACTOR_BITS);
  auto mulres_lo_hi = _mm_srai_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(sum_lo, rounding), _mm_unpackhi_epi16(factor2_lo, one)), FACTOR_BITS);
  auto mulres_lo = _mm_packs_epi32(mulres_lo_lo, mulres_lo_hi);

  auto mulres_hi_lo = _mm_srai_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(sum_hi, rounding), _mm_unpacklo_epi16(factor2_hi, one)), FACTOR_BITS);
  auto mulres_hi_hi = _mm_srai_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(sum_hi, rounding), _mm_unpackhi_epi16(factor2_hi, one)), FACTOR_BITS);
  auto mulres_hi = _mm_packs_epi32(mulres_hi_lo, mulres_hi_hi);
#else
    // factor1 = sum*2 + cnt, sum elements are 16 bits
  auto cnt_lo = _mm_unpacklo_epi8(cnt, zero);
  auto cnt_hi = _mm_unpackhi_epi8(cnt, zero);
  auto factor1_lo = _mm_add_epi16(_mm_add_epi16(sum_lo, sum_lo), cnt_lo);
  auto factor1_hi = _mm_add_epi16(_mm_add_epi16(sum_hi, sum_hi), cnt_hi);
  // factor2 = scaletab[cnt]
  auto factor2_lsb = _mm_shuffle_epi8(scaletab_lut_lsbs, cnt);
  auto factor2_msb = _mm_shuffle_epi8(scaletab_lut_msbs, cnt);
  auto factor2_lo = _mm_unpacklo_epi8(factor2_lsb, factor2_msb);
  auto factor2_hi = _mm_unpackhi_epi8(factor2_lsb, factor2_msb);
  // finally mul and shift
  auto mulres_lo = _mm_mulhi_epi16(factor1_lo, factor2_lo); // upper 16 bit of mul result, no need for >> 16
  auto mulres_hi = _mm_mulhi_epi16(factor1_hi, factor2_hi); // upper 16 bit of mul result, no need for >> 16
#endif
  // move back to 16x8 bits
  auto result = _mm_packus_epi16(mulres_lo, mulres_hi);
  // decide if original pixel is kept
  auto finalres = _mm_blendv_epi8(b, result, mask_either_is_true); // true: second param, false: 1st param

  _mm_storeu_si128(reinterpret_cast<__m128i *>(destp + x), finalres);
}

// Temporal only
#ifdef __clang__
__attribute__((__target__("sse4.1")))
#endif
void fluxT_sse41(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab)
{
  __m128i scaletab_lut_lsbs;
  __m128i scaletab_lut_msbs;
  for (int i = 0; i < 16; i++) {
    ((uint8_t*)&scaletab_lut_lsbs)[i] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i] = (scaletab[i] >> 8) & 0xFF;
  }

  const int xcnt = width;

  __m128i temporal_threshold_vector = _mm_set1_epi8(temporal_threshold);

  const int wmod16 = xcnt / 16 * 16;
  const int rest = xcnt - wmod16;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < wmod16; x += 16)
      fluxT_core_sse41(currp, prevp, nextp, destp, x, temporal_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);
    // do rest
    if (rest > 0)
      fluxT_core_sse41(currp, prevp, nextp, destp, xcnt - 16, temporal_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
}

/************************************
// Temporal only SSE4.1, 16 bit
************************************/
#ifdef __clang__
__attribute__((__target__("sse4.1")))
#endif
__forceinline void fluxT_core_sse41_uint16(const uint8_t * currp, const uint8_t* prevp, const uint8_t *nextp, uint8_t *destp, int x,
  __m128i &temporal_threshold_vector // already shifted to "signed" domain
)
{
  const auto make_signed_word = _mm_set1_epi16(0x8000); // int16 support is better than of uint16 (cmp, etc...)

  auto b_orig = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x));
  auto pbt_orig = _mm_loadu_si128(reinterpret_cast<const __m128i*>(prevp + x));
  auto nbt_orig = _mm_loadu_si128(reinterpret_cast<const __m128i*>(nextp + x));

  auto b = _mm_add_epi16(b_orig, make_signed_word);
  auto pbt = _mm_add_epi16(pbt_orig, make_signed_word);
  auto nbt = _mm_add_epi16(nbt_orig, make_signed_word);
  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm_cmpgt_epi16(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm_cmpgt_epi16(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm_cmpgt_epi16(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm_cmpgt_epi16(nbt, b); // FF where nbt > b
  auto both_less = _mm_and_si128(pbt_lessthan_b, nbt_lessthan_b);
  auto both_greater = _mm_and_si128(pbt_greaterthan_b, nbt_greaterthan_b);
  auto mask_either_is_true = _mm_or_si128(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

  // int sum = b, cnt = 1;
  auto zero = _mm_setzero_si128();
  auto sum_lo = _mm_unpacklo_epi16(b_orig, zero);
  auto sum_hi = _mm_unpackhi_epi16(b_orig, zero);
  auto cnt = _mm_set1_epi16(1);

  check_neighbour_simd_uint16(pbt_orig, b_orig, temporal_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(nbt_orig, b_orig, temporal_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  // (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);

  auto cnt_lo = _mm_unpacklo_epi16(cnt, zero);
  auto cnt_hi = _mm_unpackhi_epi16(cnt, zero);
  // Difference from SSE4.1 and C: floating point division
  // SSE2: sum / count -> (int)((float)sum * 1.0f/(float)count + 0.5f)
  const auto rounder_half = _mm_set1_ps(0.5f);
  // lower 4 pixels
  auto fcnt_lo = _mm_cvtepi32_ps(cnt_lo);
  auto fsum_lo = _mm_cvtepi32_ps(sum_lo);
  auto mulres_lo = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_lo, _mm_rcp_ps(fcnt_lo)), rounder_half));
  // upper 4 pixels
  auto fcnt_hi = _mm_cvtepi32_ps(cnt_hi);
  auto fsum_hi = _mm_cvtepi32_ps(sum_hi);
  auto mulres_hi = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_hi, _mm_rcp_ps(fcnt_hi)), rounder_half));

  // move back to 8x16 bits
  auto result = _mm_packus_epi32(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm_blendv_epi8(b_orig, result, mask_either_is_true); // true: second param, false: 1st param

  _mm_storeu_si128(reinterpret_cast<__m128i *>(destp + x), finalres);
}

// Temporal only
#ifdef __clang__
__attribute__((__target__("sse4.1")))
#endif
void fluxT_sse41_uint16(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab)
{
  const int xcnt = width;

  __m128i temporal_threshold_vector = _mm_set1_epi16(temporal_threshold - 0x8000); // move to signed int16 domain

  // uint16_t: 8 pixels per cycle
  const int wmod8 = xcnt / 8 * 8;
  const int rest = xcnt - wmod8;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < wmod8; x += 8)
      fluxT_core_sse41_uint16(currp, prevp, nextp, destp, x * sizeof(uint16_t), temporal_threshold_vector);
    // do rest
    if (rest > 0)
      fluxT_core_sse41_uint16(currp, prevp, nextp, destp, (xcnt - 8) * sizeof(uint16_t), temporal_threshold_vector);

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
}

/************************************
// Spatial Temporal SSE2, 8 bit
************************************/
__forceinline void fluxST_core_sse2(const uint8_t * currp, const int src_pitch, const uint8_t* prevp, const uint8_t *nextp, uint8_t *destp, int x,
  __m128i &temporal_threshold_vector,
  __m128i &spatial_threshold_vector,
  __m128i &scaletab_lut_lsbs,
  __m128i &scaletab_lut_msbs
)
{
  // +1: center of 3x3 pixels [+0,+1,+2]
  auto b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + 1));
  auto pbt = _mm_loadu_si128(reinterpret_cast<const __m128i*>(prevp + x + 1));
  auto nbt = _mm_loadu_si128(reinterpret_cast<const __m128i*>(nextp + x + 1));

  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm_cmpgt_epu8(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm_cmpgt_epu8(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm_cmpgt_epu8(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm_cmpgt_epu8(nbt, b); // FF where nbt > b
  auto both_less = _mm_and_si128(pbt_lessthan_b, nbt_lessthan_b);
  auto both_greater = _mm_and_si128(pbt_greaterthan_b, nbt_greaterthan_b);
  auto mask_either_is_true = _mm_or_si128(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

    // int pb1 = currp[-src_pitch - 1], pb2 = currp[-src_pitch], pb3 = currp[-src_pitch + 1];
    // int b1 = currp[-1], /*b = currp[0], */b2 = currp[1];
    // int nb1 = currp[src_pitch - 1], nb2 = currp[src_pitch], nb3 = currp[src_pitch + 1];

  auto pb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x - src_pitch + 0));
  auto pb2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x - src_pitch + 1));
  auto pb3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x - src_pitch + 2));

  auto b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + 0));
  auto b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + 2));

  auto nb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + src_pitch + 0));
  auto nb2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + src_pitch + 1));
  auto nb3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + src_pitch + 2));

  // int sum = b, cnt = 1;
  auto zero = _mm_setzero_si128();
  auto sum_lo = _mm_unpacklo_epi8(b, zero);
  auto sum_hi = _mm_unpackhi_epi8(b, zero);
  auto cnt = _mm_set1_epi8(1);

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
  auto cnt_lo = _mm_unpacklo_epi8(cnt, zero);
  auto cnt_hi = _mm_unpackhi_epi8(cnt, zero);

  // Difference from SSE4.1 and C: floating point division
  // SSE2: sum / count -> (int)((float)sum * 1.0f/(float)count + 0.5f)
  const auto rounder_half = _mm_set1_ps(0.5f);
  // lower 8 pixels
  auto fcnt_lo_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(cnt_lo, zero));
  auto fcnt_lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(cnt_lo, zero));
  auto fsum_lo_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(sum_lo, zero));
  auto fsum_lo_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(sum_lo, zero));

  auto mul_lo_lo = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_lo_lo, _mm_rcp_ps(fcnt_lo_lo)), rounder_half));
  auto mul_lo_hi = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_lo_hi, _mm_rcp_ps(fcnt_lo_hi)), rounder_half));
  auto mulres_lo = _mm_packs_epi32(mul_lo_lo, mul_lo_hi);
  // upper 8 pixels
  auto fcnt_hi_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(cnt_hi, zero));
  auto fcnt_hi_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(cnt_hi, zero));
  auto fsum_hi_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(sum_hi, zero));
  auto fsum_hi_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(sum_hi, zero));

  auto mul_hi_lo = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_hi_lo, _mm_rcp_ps(fcnt_hi_lo)), rounder_half));
  auto mul_hi_hi = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_hi_hi, _mm_rcp_ps(fcnt_hi_hi)), rounder_half));
  auto mulres_hi = _mm_packs_epi32(mul_hi_lo, mul_hi_hi);

  // move back to 16x8 bits
  auto result = _mm_packus_epi16(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _MM_BLENDV_EPI8(b, result, mask_either_is_true); // true: second param, false: 1st param

  _mm_storeu_si128(reinterpret_cast<__m128i *>(destp + x + 1), finalres);
}

// Spatial Temporal
void fluxST_sse2(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab)
{
  __m128i scaletab_lut_lsbs;
  __m128i scaletab_lut_msbs;
  for (int i = 0; i < 16; i++) {
    ((uint8_t*)&scaletab_lut_lsbs)[i] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i] = (scaletab[i] >> 8) & 0xFF;
  }

  // spatial: because of previous and next line involved, function is called 
  // starting with the 2nd line and with height = (real_height - 2) 
  const int xcnt = width - 2; // leftmost/rightmost column safety

  __m128i temporal_threshold_vector = _mm_set1_epi8(temporal_threshold);
  __m128i spatial_threshold_vector = _mm_set1_epi8(spatial_threshold);

  const int wmod16 = xcnt / 16 * 16;
  const int rest = xcnt - wmod16;

  for (int y = 0; y < height; y++)
  {
    destp[0] = currp[0]; // Copy left edge

    for (int x = 0; x < wmod16; x += 16)
      fluxST_core_sse2(currp, src_pitch, prevp, nextp, destp, x, temporal_threshold_vector, spatial_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);
    // do rest
    if (rest > 0)
      fluxST_core_sse2(currp, src_pitch, prevp, nextp, destp, xcnt - 16, temporal_threshold_vector, spatial_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);

    destp[width - 1] = currp[width - 1]; // Copy right edge

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
}

/************************************
// Spatial Temporal SSE4.1, 8 bit
************************************/
#ifdef __clang__
__attribute__((__target__("sse4.1")))
#endif
__forceinline void fluxST_core_sse41(const uint8_t * currp, const int src_pitch, const uint8_t* prevp, const uint8_t *nextp, uint8_t *destp, int x,
  __m128i &temporal_threshold_vector,
  __m128i &spatial_threshold_vector,
  __m128i &scaletab_lut_lsbs,
  __m128i &scaletab_lut_msbs
)
{
  // +1: center of 3x3 pixels [+0,+1,+2]
  auto b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + 1));
  auto pbt = _mm_loadu_si128(reinterpret_cast<const __m128i*>(prevp + x + 1));
  auto nbt = _mm_loadu_si128(reinterpret_cast<const __m128i*>(nextp + x + 1));

  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm_cmpgt_epu8(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm_cmpgt_epu8(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm_cmpgt_epu8(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm_cmpgt_epu8(nbt, b); // FF where nbt > b
  auto both_less = _mm_and_si128(pbt_lessthan_b, nbt_lessthan_b);
  auto both_greater = _mm_and_si128(pbt_greaterthan_b, nbt_greaterthan_b);
  auto mask_either_is_true = _mm_or_si128(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

    // int pb1 = currp[-src_pitch - 1], pb2 = currp[-src_pitch], pb3 = currp[-src_pitch + 1];
    // int b1 = currp[-1], /*b = currp[0], */b2 = currp[1];
    // int nb1 = currp[src_pitch - 1], nb2 = currp[src_pitch], nb3 = currp[src_pitch + 1];

  auto pb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x - src_pitch + 0));
  auto pb2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x - src_pitch + 1));
  auto pb3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x - src_pitch + 2));

  auto b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + 0));
  auto b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + 2));

  auto nb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + src_pitch + 0));
  auto nb2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + src_pitch + 1));
  auto nb3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + src_pitch + 2));

  // int sum = b, cnt = 1;
  auto zero = _mm_setzero_si128();
  auto sum_lo = _mm_unpacklo_epi8(b, zero);
  auto sum_hi = _mm_unpackhi_epi8(b, zero);
  auto cnt = _mm_set1_epi8(1);

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
  auto cnt_lo = _mm_unpacklo_epi8(cnt, zero);
  auto cnt_hi = _mm_unpackhi_epi8(cnt, zero);
  auto factor1_lo = _mm_add_epi16(_mm_add_epi16(sum_lo, sum_lo), cnt_lo);
  auto factor1_hi = _mm_add_epi16(_mm_add_epi16(sum_hi, sum_hi), cnt_hi);
  // factor2 = scaletab[cnt]
  auto factor2_lsb = _mm_shuffle_epi8(scaletab_lut_lsbs, cnt);
  auto factor2_msb = _mm_shuffle_epi8(scaletab_lut_msbs, cnt);
  auto factor2_lo = _mm_unpacklo_epi8(factor2_lsb, factor2_msb);
  auto factor2_hi = _mm_unpackhi_epi8(factor2_lsb, factor2_msb);
  // finally mul and shift
  auto mulres_lo = _mm_mulhi_epi16(factor1_lo, factor2_lo); // upper 16 bit of mul result, no need for >> 16
  auto mulres_hi = _mm_mulhi_epi16(factor1_hi, factor2_hi); // upper 16 bit of mul result, no need for >> 16
  // move back to 16x8 bits
  auto result = _mm_packus_epi16(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm_blendv_epi8(b, result, mask_either_is_true); // true: second param, false: 1st param

  _mm_storeu_si128(reinterpret_cast<__m128i *>(destp + x + 1), finalres);
}

// Spatial Temporal
#ifdef __clang__
__attribute__((__target__("sse4.1")))
#endif
void fluxST_sse41(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab)
{
  __m128i scaletab_lut_lsbs;
  __m128i scaletab_lut_msbs;
  for (int i = 0; i < 16; i++) {
    ((uint8_t*)&scaletab_lut_lsbs)[i] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i] = (scaletab[i] >> 8) & 0xFF;
  }

  // spatial: because of previous and next line involved, function is called 
  // starting with the 2nd line and with height = (real_height - 2) 
  const int xcnt = width - 2; // leftmost/rightmost column safety

  __m128i temporal_threshold_vector = _mm_set1_epi8(temporal_threshold);
  __m128i spatial_threshold_vector = _mm_set1_epi8(spatial_threshold);

  const int wmod16 = xcnt / 16 * 16;
  const int rest = xcnt - wmod16;

  for (int y = 0; y < height; y++)
  {
    destp[0] = currp[0]; // Copy left edge

    for (int x = 0; x < wmod16; x += 16)
      fluxST_core_sse41(currp, src_pitch, prevp, nextp, destp, x, temporal_threshold_vector, spatial_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);
    // do rest
    if (rest > 0)
      fluxST_core_sse41(currp, src_pitch, prevp, nextp, destp, xcnt - 16, temporal_threshold_vector, spatial_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);

    destp[width - 1] = currp[width - 1]; // Copy right edge

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
}

/************************************
// Spatial Temporal SSE4.1, 16 bit
************************************/
#ifdef __clang__
__attribute__((__target__("sse4.1")))
#endif
__forceinline void fluxST_core_sse41_uint16(const uint8_t * currp, const int src_pitch, const uint8_t* prevp, const uint8_t *nextp, uint8_t *destp, int x,
  __m128i &temporal_threshold_vector, // already shifted to "signed" domain
  __m128i &spatial_threshold_vector // already shifted to "signed" domain
)
{
  const auto make_signed_word = _mm_set1_epi16(0x8000); // int16 support is better than of uint16 (cmp, etc...)
  // +1: center of 3x3 pixels [+0,+1,+2]
  auto b_orig = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + 1 * sizeof(uint16_t)));
  auto pbt_orig = _mm_loadu_si128(reinterpret_cast<const __m128i*>(prevp + x + 1 * sizeof(uint16_t)));
  auto nbt_orig = _mm_loadu_si128(reinterpret_cast<const __m128i*>(nextp + x + 1 * sizeof(uint16_t)));

  auto b = _mm_add_epi16(b_orig, make_signed_word);
  auto pbt = _mm_add_epi16(pbt_orig, make_signed_word);
  auto nbt = _mm_add_epi16(nbt_orig, make_signed_word);
  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm_cmpgt_epi16(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm_cmpgt_epi16(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm_cmpgt_epi16(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm_cmpgt_epi16(nbt, b); // FF where nbt > b
  auto both_less = _mm_and_si128(pbt_lessthan_b, nbt_lessthan_b);
  auto both_greater = _mm_and_si128(pbt_greaterthan_b, nbt_greaterthan_b);
  auto mask_either_is_true = _mm_or_si128(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

    // int pb1 = currp[-src_pitch - 1], pb2 = currp[-src_pitch], pb3 = currp[-src_pitch + 1];
    // int b1 = currp[-1], /*b = currp[0], */b2 = currp[1];
    // int nb1 = currp[src_pitch - 1], nb2 = currp[src_pitch], nb3 = currp[src_pitch + 1];

  auto pb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x - src_pitch + 0 * sizeof(uint16_t)));
  auto pb2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x - src_pitch + 1 * sizeof(uint16_t)));
  auto pb3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x - src_pitch + 2 * sizeof(uint16_t)));

  auto b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + 0 * sizeof(uint16_t)));
  auto b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + 2 * sizeof(uint16_t)));

  auto nb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + src_pitch + 0 * sizeof(uint16_t)));
  auto nb2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + src_pitch + 1 * sizeof(uint16_t)));
  auto nb3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(currp + x + src_pitch + 2 * sizeof(uint16_t)));

  // int sum = b, cnt = 1;
  auto zero = _mm_setzero_si128();
  auto sum_lo = _mm_unpacklo_epi16(b_orig, zero);
  auto sum_hi = _mm_unpackhi_epi16(b_orig, zero);
  auto cnt = _mm_set1_epi16(1);

  check_neighbour_simd_uint16(pbt_orig, b_orig, temporal_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(nbt_orig, b_orig, temporal_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(pb1, b_orig, spatial_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(pb2, b_orig, spatial_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(pb3, b_orig, spatial_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(b1, b_orig, spatial_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(b2, b_orig, spatial_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(nb1, b_orig, spatial_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(nb2, b_orig, spatial_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(nb3, b_orig, spatial_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  // (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);

  auto cnt_lo = _mm_unpacklo_epi16(cnt, zero);
  auto cnt_hi = _mm_unpackhi_epi16(cnt, zero);
  // Difference from SSE4.1 and C: floating point division
  // SSE2: sum / count -> (int)((float)sum * 1.0f/(float)count + 0.5f)
  const auto rounder_half = _mm_set1_ps(0.5f);
  // lower 4 pixels
  auto fcnt_lo = _mm_cvtepi32_ps(cnt_lo);
  auto fsum_lo = _mm_cvtepi32_ps(sum_lo);
  auto mulres_lo = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_lo, _mm_rcp_ps(fcnt_lo)), rounder_half));
  // upper 4 pixels
  auto fcnt_hi = _mm_cvtepi32_ps(cnt_hi);
  auto fsum_hi = _mm_cvtepi32_ps(sum_hi);
  auto mulres_hi = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(fsum_hi, _mm_rcp_ps(fcnt_hi)), rounder_half));

  // move back to 8x16 bits
  auto result = _mm_packus_epi32(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm_blendv_epi8(b_orig, result, mask_either_is_true); // true: second param, false: 1st param

  _mm_storeu_si128(reinterpret_cast<__m128i *>(destp + x + 1 * sizeof(uint16_t)), finalres);
}

// Spatial Temporal
#ifdef __clang__
__attribute__((__target__("sse4.1")))
#endif
void fluxST_sse41_uint16(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab)
{
  // spatial: because of previous and next line involved, function is called 
  // starting with the 2nd line and with height = (real_height - 2) 
  const int xcnt = width - 2; // leftmost/rightmost column safety

  __m128i temporal_threshold_vector = _mm_set1_epi16(temporal_threshold - 0x8000); // move to signed int16 domain
  __m128i spatial_threshold_vector = _mm_set1_epi16(spatial_threshold - 0x8000); // move to signed int16 domain);

  const int wmod8 = xcnt / 8 * 8;
  const int rest = xcnt - wmod8;

  for (int y = 0; y < height; y++)
  {
    reinterpret_cast<uint16_t *>(destp)[0] = reinterpret_cast<const uint16_t *>(currp)[0]; // Copy left edge

    for (int x = 0; x < wmod8; x += 8)
      fluxST_core_sse41_uint16(currp, src_pitch, prevp, nextp, destp, x * sizeof(uint16_t), temporal_threshold_vector, spatial_threshold_vector);
    // do rest
    if (rest > 0)
      fluxST_core_sse41_uint16(currp, src_pitch, prevp, nextp, destp, (xcnt - 8) * sizeof(uint16_t), temporal_threshold_vector, spatial_threshold_vector);

    reinterpret_cast<uint16_t *>(destp)[width - 1] = reinterpret_cast<const uint16_t *>(currp)[width - 1]; // Copy right edge

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
}

/************************************
// Helper
************************************/

static __forceinline void check_neighbour_C(int neighbour, int center, int threshold, int& sum, int& cnt)
{
  if (std::abs(neighbour - center) <= threshold)
  {
    sum += neighbour;
    ++cnt;
  }
}

/************************************
// Spatial Temporal C, 8-16 bit
************************************/

template<typename pixel_t>
void fluxST_C(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab)
{
  // spatial: because of previous and next line involved, function is called 
  // starting with the 2nd line and with height = (real_height - 2) 
  for (int y = 0; y < height; y++)
  {
    // leftmost column safety
    reinterpret_cast<pixel_t *>(destp)[0] = reinterpret_cast<const pixel_t *>(currp)[0]; // Copy left edge

    for (int x = 1; x < width-1; x++)
    {

      int b = reinterpret_cast<const pixel_t *>(currp)[x];
      int pbt = reinterpret_cast<const pixel_t *>(prevp)[x];
      int nbt = reinterpret_cast<const pixel_t *>(nextp)[x];
      int pdiff = pbt - b, ndiff = nbt - b;
      if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
      {
        const pixel_t* currp0 = reinterpret_cast<const pixel_t *>(currp);
        const int src_pitch0 = src_pitch / sizeof(pixel_t);
        int pb1 = currp0[x - src_pitch0 - 1];
        int pb2 = currp0[x - src_pitch0];
        int pb3 = currp0[x - src_pitch0 + 1];
        int b1 = currp0[x - 1];
        /*b = currp[0]; */
        int b2 = currp0[x + 1];
        int nb1 = currp0[x + src_pitch0 - 1];
        int nb2 = currp0[x + src_pitch0];
        int nb3 = currp0[x + src_pitch0 + 1];

        int sum = b, cnt = 1;
        check_neighbour_C(pbt, b, temporal_threshold, sum, cnt);
        check_neighbour_C(nbt, b, temporal_threshold, sum, cnt);

        check_neighbour_C(pb1, b, spatial_threshold, sum, cnt);
        check_neighbour_C(pb2, b, spatial_threshold, sum, cnt);
        check_neighbour_C(pb3, b, spatial_threshold, sum, cnt);

        check_neighbour_C(b1, b, spatial_threshold, sum, cnt);
        check_neighbour_C(b2, b, spatial_threshold, sum, cnt);

        check_neighbour_C(nb1, b, spatial_threshold, sum, cnt);
        check_neighbour_C(nb2, b, spatial_threshold, sum, cnt);
        check_neighbour_C(nb3, b, spatial_threshold, sum, cnt);

        using safe_int_t = typename std::conditional<sizeof(pixel_t) == 1, int, int64_t>::type; // 16 bit pixels: int32 overflow

        reinterpret_cast<pixel_t *>(destp)[x] = (pixel_t)(((safe_int_t)(sum * 2 + cnt) * scaletab[cnt]) >> 16);
      }
      else
      {
        reinterpret_cast<pixel_t *>(destp)[x] = b;
      }
    } // for x

    // rightmost column safety
    reinterpret_cast<pixel_t *>(destp)[width - 1] = reinterpret_cast<const pixel_t *>(currp)[width - 1]; // Copy right edge

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;

  } // for y

}

// instantiate
template void fluxST_C<uint8_t>(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);
template void fluxST_C<uint16_t>(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, int spatial_threshold, short *scaletab);

/************************************
// Termporal only C, 8-16 bit
************************************/

template<typename pixel_t>
void fluxT_C(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab)
{
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int b = reinterpret_cast<const pixel_t *>(currp)[x];
      int pbt = reinterpret_cast<const pixel_t *>(prevp)[x];
      int nbt = reinterpret_cast<const pixel_t *>(nextp)[x];
      int pdiff = pbt - b, ndiff = nbt - b;
      if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
      {
        int sum = b, cnt = 1;

        check_neighbour_C(pbt, b, temporal_threshold, sum, cnt);
        check_neighbour_C(nbt, b, temporal_threshold, sum, cnt);
        using safe_int_t = typename std::conditional<sizeof(pixel_t) == 1, int, int64_t>::type; // 16 bit pixels: int32 overflow
        // cnt: 1,2,3
        reinterpret_cast<pixel_t *>(destp)[x] = (pixel_t)(((safe_int_t)(sum * 2 + cnt) * scaletab[cnt]) >> 16);
      }
      else
      {
        reinterpret_cast<pixel_t *>(destp)[x] = (pixel_t)b;
      }
    } // for x

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;

  } // for y
}

// instantiate
template void fluxT_C<uint8_t>(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);
template void fluxT_C<uint16_t>(const uint8_t*, const int, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, const int height, int temporal_threshold, short *scaletab);

