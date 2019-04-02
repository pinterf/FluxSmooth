#include "FluxSmooth.h"
#include <algorithm>
#include "stdint.h"
#include "immintrin.h" // AVX

#if !defined(__AVX2__)
#error This source file will only work properly when compiled with AVX2 option"
#endif

/************************************
// Helpers, missing intrinsics
************************************/

#define _mm256_cmpge_epu8(a, b) _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a)

#define _mm256_cmple_epu8(a, b) _mm256_cmpge_epu8(b, a)

// does not exist
static __forceinline __m256i _mm256_cmpgt_epu8(__m256i x, __m256i y)
{
  // Returns 0xFF where x > y:
  return _mm256_andnot_si256(
    _mm256_cmpeq_epi8(x, y),
    _mm256_cmpeq_epi8(_mm256_max_epu8(x, y), x)
  );
}

__forceinline __m256i _mm256_cmpge_epi16(__m256i x, __m256i y)
{
  // Returns 0xFFFF where x >= y:
  return _mm256_or_si256(_mm256_cmpeq_epi16(x, y), _mm256_cmpgt_epi16(x, y));
}

#define _mm256_cmple_epi16(a, b) _mm256_cmpge_epi16(b, a)

/************************************
// Helpers
************************************/

__forceinline void check_neighbour_simd(__m256i &neighbour, __m256i &center, __m256i &threshold,
  __m256i &sum_lo, __m256i &sum_hi, __m256i &cnt)
{
  auto n_minus_c = _mm256_subs_epu8(neighbour, center);
  auto c_minus_n = _mm256_subs_epu8(center, neighbour);
  auto absdiff = _mm256_or_si256(n_minus_c, c_minus_n);
  auto abs_is_lessthanoreq_thresh = _mm256_cmple_epu8(absdiff, threshold);
  // count. increment when true. We simply sub the mask value 00 (0) or FF (-1)
  cnt = _mm256_sub_epi8(cnt, abs_is_lessthanoreq_thresh);
  // increase sum elements by neighbour where true, that is mask is FF
  // sum is 16 bits
  auto masked_neighbour = _mm256_and_si256(abs_is_lessthanoreq_thresh, neighbour);
  auto zero = _mm256_setzero_si256();
  auto masked_neighbour_lo = _mm256_unpacklo_epi8(masked_neighbour, zero);
  auto masked_neighbour_hi = _mm256_unpackhi_epi8(masked_neighbour, zero);
  sum_lo = _mm256_add_epi16(sum_lo, masked_neighbour_lo);
  sum_hi = _mm256_add_epi16(sum_hi, masked_neighbour_hi);

  /*
  if (std::abs(neighbour - center) <= threshold)
  {
    sum += neighbour;
    ++cnt;
  }
  */
}

__forceinline void check_neighbour_simd_uint16(__m256i &neighbour, __m256i &center, __m256i &threshold,
  __m256i &sum_lo, __m256i &sum_hi, __m256i &cnt, const __m256i &make_signed_word)
{
  // threshold is shifted to the "signed" int16 domain
  auto n_minus_c = _mm256_subs_epu16(neighbour, center);
  auto c_minus_n = _mm256_subs_epu16(center, neighbour);
  auto absdiff = _mm256_or_si256(n_minus_c, c_minus_n);
  // absdiff <= threshold ==> !(absdiff > threshold)
  // FIXME make it a bit faster: cmpgt and later: andnot, and count in a reverse way (instead of increase-when-match use decrease-by-non-match)
  auto abs_is_lessthanoreq_thresh = _mm256_cmple_epi16(_mm256_add_epi16(absdiff, make_signed_word), threshold);
  // count. increment when true. We simply sub the mask value 0000 (0) or FFFF (-1)
  cnt = _mm256_sub_epi16(cnt, abs_is_lessthanoreq_thresh);
  // increase sum elements by neighbour where true, that is mask is FF
  // sum is 16 bits
  auto masked_neighbour = _mm256_and_si256(abs_is_lessthanoreq_thresh, neighbour);
  auto zero = _mm256_setzero_si256();
  auto masked_neighbour_lo = _mm256_unpacklo_epi16(masked_neighbour, zero);
  auto masked_neighbour_hi = _mm256_unpackhi_epi16(masked_neighbour, zero);
  sum_lo = _mm256_add_epi32(sum_lo, masked_neighbour_lo);
  sum_hi = _mm256_add_epi32(sum_hi, masked_neighbour_hi);

  /*
  if (std::abs(neighbour - center) <= threshold)
  {
    sum += neighbour;
    ++cnt;
  }
  */
}

/************************************
// Temporal only AVX2, 8 bit
************************************/

__forceinline void fluxT_core_avx2(const BYTE * currp, const long long src_pitch,
  const BYTE * prevp, const BYTE * nextp,
  BYTE * destp, int x,
  __m256i &temporal_threshold_vector,
  __m256i &scaletab_lut_lsbs,
  __m256i &scaletab_lut_msbs
)
{
  auto b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x));
  auto pbt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prevp + x));
  auto nbt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(nextp + x));
  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm256_cmpgt_epu8(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm256_cmpgt_epu8(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm256_cmpgt_epu8(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm256_cmpgt_epu8(nbt, b); // FF where nbt > b
  auto both_less = _mm256_and_si256(pbt_lessthan_b, nbt_lessthan_b);
  auto both_greater = _mm256_and_si256(pbt_greaterthan_b, nbt_greaterthan_b);
  auto mask_either_is_true = _mm256_or_si256(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

  // int sum = b, cnt = 1;
  auto zero = _mm256_setzero_si256();
  auto sum_lo = _mm256_unpacklo_epi8(b, zero);
  auto sum_hi = _mm256_unpackhi_epi8(b, zero);
  auto cnt = _mm256_set1_epi8(1);

  check_neighbour_simd(pbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  check_neighbour_simd(nbt, b, temporal_threshold_vector, sum_lo, sum_hi, cnt);
  // (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);

  // factor1 = sum*2 + cnt, sum elements are 16 bits
  auto cnt_lo = _mm256_unpacklo_epi8(cnt, zero);
  auto cnt_hi = _mm256_unpackhi_epi8(cnt, zero);
  auto factor1_lo = _mm256_add_epi16(_mm256_add_epi16(sum_lo, sum_lo), cnt_lo);
  auto factor1_hi = _mm256_add_epi16(_mm256_add_epi16(sum_hi, sum_hi), cnt_hi);
  // factor2 = scaletab[cnt]
  auto factor2_lsb = _mm256_shuffle_epi8(scaletab_lut_lsbs, cnt);
  auto factor2_msb = _mm256_shuffle_epi8(scaletab_lut_msbs, cnt);
  auto factor2_lo = _mm256_unpacklo_epi8(factor2_lsb, factor2_msb);
  auto factor2_hi = _mm256_unpackhi_epi8(factor2_lsb, factor2_msb);
  // finally mul and shift
  auto mulres_lo = _mm256_mulhi_epi16(factor1_lo, factor2_lo); // upper 16 bit of mul result, no need for >> 16
  auto mulres_hi = _mm256_mulhi_epi16(factor1_hi, factor2_hi); // upper 16 bit of mul result, no need for >> 16
  // move back to 16x8 bits
  auto result = _mm256_packus_epi16(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm256_blendv_epi8(b, result, mask_either_is_true); // true: second param, false: 1st param

  _mm256_storeu_si256(reinterpret_cast<__m256i *>(destp + x), finalres);
}


// Temporal only
void fluxT_avx2(const uint8_t* currp, const int src_pitch,
  const uint8_t * prevp, const int prv_pitch,
  const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch,
  const int width, int height,
  int temporal_threshold,
  short *scaletab)
{
  __m256i scaletab_lut_lsbs;
  __m256i scaletab_lut_msbs;
  for (int i = 0; i < 16; i++) {
    ((uint8_t*)&scaletab_lut_lsbs)[i] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i] = (scaletab[i] >> 8) & 0xFF;
    // same for hi 128
    ((uint8_t*)&scaletab_lut_lsbs)[i+16] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i+16] = (scaletab[i] >> 8) & 0xFF;
  }

  const int xcnt = width;

  __m256i temporal_threshold_vector = _mm256_set1_epi8(temporal_threshold);

  const int wmod32 = xcnt / 32 * 32;
  const int rest = xcnt - wmod32;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < wmod32; x += 32)
      fluxT_core_avx2(currp, src_pitch, prevp, nextp, destp, x, temporal_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);
    // do rest
    if (rest > 0)
      fluxT_core_avx2(currp, src_pitch, prevp, nextp, destp, xcnt - 32, temporal_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
  _mm256_zeroupper();
}

/************************************
// Temporal only AVX2, 16 bit
************************************/

__forceinline void fluxT_core_avx2_uint16(const uint8_t * currp, const int src_pitch, const uint8_t* prevp, const uint8_t *nextp, uint8_t *destp, int x,
  __m256i &temporal_threshold_vector // already shifted to "signed" domain
)
{
  const auto make_signed_word = _mm256_set1_epi16(0x8000); // int16 support is better than of uint16 (cmp, etc...)

  auto b_orig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x));
  auto pbt_orig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prevp + x));
  auto nbt_orig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(nextp + x));

  auto b = _mm256_add_epi16(b_orig, make_signed_word);
  auto pbt = _mm256_add_epi16(pbt_orig, make_signed_word);
  auto nbt = _mm256_add_epi16(nbt_orig, make_signed_word);
  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm256_cmpgt_epi16(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm256_cmpgt_epi16(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm256_cmpgt_epi16(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm256_cmpgt_epi16(nbt, b); // FF where nbt > b
  auto both_less = _mm256_and_si256(pbt_lessthan_b, nbt_lessthan_b);
  auto both_greater = _mm256_and_si256(pbt_greaterthan_b, nbt_greaterthan_b);
  auto mask_either_is_true = _mm256_or_si256(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

  // int sum = b, cnt = 1;
  auto zero = _mm256_setzero_si256();
  auto sum_lo = _mm256_unpacklo_epi16(b_orig, zero);
  auto sum_hi = _mm256_unpackhi_epi16(b_orig, zero);
  auto cnt = _mm256_set1_epi16(1);

  check_neighbour_simd_uint16(pbt_orig, b_orig, temporal_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  check_neighbour_simd_uint16(nbt_orig, b_orig, temporal_threshold_vector, sum_lo, sum_hi, cnt, make_signed_word);
  // (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);

  auto cnt_lo = _mm256_unpacklo_epi16(cnt, zero);
  auto cnt_hi = _mm256_unpackhi_epi16(cnt, zero);
  // Difference from SSE4.1 and C: floating point division
  // SSE2: sum / count -> (int)((float)sum * 1.0f/(float)count + 0.5f)
  const auto rounder_half = _mm256_set1_ps(0.5f);
  // lower 4 pixels
  auto fcnt_lo = _mm256_cvtepi32_ps(cnt_lo);
  auto fsum_lo = _mm256_cvtepi32_ps(sum_lo);
  auto mulres_lo = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(fsum_lo, _mm256_rcp_ps(fcnt_lo)), rounder_half));
  // upper 4 pixels
  auto fcnt_hi = _mm256_cvtepi32_ps(cnt_hi);
  auto fsum_hi = _mm256_cvtepi32_ps(sum_hi);
  auto mulres_hi = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(fsum_hi, _mm256_rcp_ps(fcnt_hi)), rounder_half));

  // move back to 8x16 bits
  auto result = _mm256_packus_epi32(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm256_blendv_epi8(b_orig, result, mask_either_is_true); // true: second param, false: 1st param

  _mm256_storeu_si256(reinterpret_cast<__m256i *>(destp + x), finalres);
}

// Temporal only
void fluxT_avx2_uint16(const uint8_t* currp, const int src_pitch,
  const uint8_t * prevp, const int prv_pitch,
  const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch,
  const int width, int height,
  int temporal_threshold,
  short *scaletab)
{
  const int xcnt = width;

  __m256i temporal_threshold_vector = _mm256_set1_epi16(temporal_threshold - 0x8000); // move to signed int16 domain

  const int wmod16 = xcnt / 16 * 16;
  const int rest = xcnt - wmod16;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < wmod16; x += 16)
      fluxT_core_avx2_uint16(currp, src_pitch, prevp, nextp, destp, x * sizeof(uint16_t), temporal_threshold_vector);
    // do rest
    if (rest > 0)
      fluxT_core_avx2_uint16(currp, src_pitch, prevp, nextp, destp, (xcnt - 16) * sizeof(uint16_t), temporal_threshold_vector);

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
  _mm256_zeroupper();
}

/************************************
// Spatial Temporal AVX2, 8 bit
************************************/

__forceinline void fluxST_core_avx2(const BYTE * currp, const long long src_pitch,
  const BYTE * prevp, const BYTE * nextp,
  BYTE * destp, int x,
  __m256i &temporal_threshold_vector,
  __m256i &spatial_threshold_vector,
  __m256i &scaletab_lut_lsbs,
  __m256i &scaletab_lut_msbs
)
{
  // +1: center of 3x3 pixels [+0,+1,+2]
  auto b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + 1));
  auto pbt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prevp + x + 1));
  auto nbt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(nextp + x + 1));

  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm256_cmpgt_epu8(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm256_cmpgt_epu8(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm256_cmpgt_epu8(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm256_cmpgt_epu8(nbt, b); // FF where nbt > b
  auto both_less = _mm256_and_si256(pbt_lessthan_b, nbt_lessthan_b);
  auto both_greater = _mm256_and_si256(pbt_greaterthan_b, nbt_greaterthan_b);
  auto mask_either_is_true = _mm256_or_si256(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

    // int pb1 = currp[-src_pitch - 1], pb2 = currp[-src_pitch], pb3 = currp[-src_pitch + 1];
    // int b1 = currp[-1], /*b = currp[0], */b2 = currp[1];
    // int nb1 = currp[src_pitch - 1], nb2 = currp[src_pitch], nb3 = currp[src_pitch + 1];

  auto pb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x - src_pitch + 0));
  auto pb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x - src_pitch + 1));
  auto pb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x - src_pitch + 2));

  auto b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + 0));
  auto b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + 2));

  auto nb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + src_pitch + 0));
  auto nb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + src_pitch + 1));
  auto nb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + src_pitch + 2));

  // int sum = b, cnt = 1;
  auto zero = _mm256_setzero_si256();
  auto sum_lo = _mm256_unpacklo_epi8(b, zero);
  auto sum_hi = _mm256_unpackhi_epi8(b, zero);
  auto cnt = _mm256_set1_epi8(1);

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
  auto cnt_lo = _mm256_unpacklo_epi8(cnt, zero);
  auto cnt_hi = _mm256_unpackhi_epi8(cnt, zero);
  auto factor1_lo = _mm256_add_epi16(_mm256_add_epi16(sum_lo, sum_lo), cnt_lo);
  auto factor1_hi = _mm256_add_epi16(_mm256_add_epi16(sum_hi, sum_hi), cnt_hi);
  // factor2 = scaletab[cnt]
  auto factor2_lsb = _mm256_shuffle_epi8(scaletab_lut_lsbs, cnt);
  auto factor2_msb = _mm256_shuffle_epi8(scaletab_lut_msbs, cnt);
  auto factor2_lo = _mm256_unpacklo_epi8(factor2_lsb, factor2_msb);
  auto factor2_hi = _mm256_unpackhi_epi8(factor2_lsb, factor2_msb);
  // finally mul and shift
  auto mulres_lo = _mm256_mulhi_epi16(factor1_lo, factor2_lo); // upper 16 bit of mul result, no need for >> 16
  auto mulres_hi = _mm256_mulhi_epi16(factor1_hi, factor2_hi); // upper 16 bit of mul result, no need for >> 16
  // move back to 16x8 bits
  auto result = _mm256_packus_epi16(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm256_blendv_epi8(b, result, mask_either_is_true); // true: second param, false: 1st param

  _mm256_storeu_si256(reinterpret_cast<__m256i *>(destp + x + 1), finalres);
}

// Spatial Temporal
void fluxST_avx2(const uint8_t* currp, const int src_pitch, const uint8_t * prevp, const int prv_pitch, const uint8_t * nextp, const int nxt_pitch,
  uint8_t* destp, const int dst_pitch, const int width, int height, int temporal_threshold, int spatial_threshold, short *scaletab)
{
  __m256i scaletab_lut_lsbs;
  __m256i scaletab_lut_msbs;
  for (int i = 0; i < 16; i++) {
    ((uint8_t*)&scaletab_lut_lsbs)[i] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i] = (scaletab[i] >> 8) & 0xFF;
    // same for upper 128
    ((uint8_t*)&scaletab_lut_lsbs)[i+16] = scaletab[i] & 0xFF;
    ((uint8_t*)&scaletab_lut_msbs)[i+16] = (scaletab[i] >> 8) & 0xFF;
  }

  // spatial: because of previous and next line involved, function is called 
  // starting with the 2nd line and with height = (real_height - 2) 
  const int xcnt = width - 2; // leftmost/rightmost column safety

  __m256i temporal_threshold_vector = _mm256_set1_epi8(temporal_threshold);
  __m256i spatial_threshold_vector = _mm256_set1_epi8(spatial_threshold);

  const int wmod32 = xcnt / 32 * 32;
  const int rest = xcnt - wmod32;

  for (int y = 0; y < height; y++)
  {
    destp[0] = currp[0]; // Copy left edge

    for (int x = 0; x < wmod32; x += 32)
      fluxST_core_avx2(currp, src_pitch, prevp, nextp, destp, x, temporal_threshold_vector, spatial_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);
    // do rest
    if (rest > 0)
      fluxST_core_avx2(currp, src_pitch, prevp, nextp, destp, xcnt - 32, temporal_threshold_vector, spatial_threshold_vector, scaletab_lut_lsbs, scaletab_lut_msbs);

    destp[width - 1] = currp[width - 1]; // Copy right edge

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
  _mm256_zeroupper();
}

/************************************
// Spatial Temporal AVX2, 16 bit
************************************/
__forceinline void fluxST_core_avx2_uint16(const uint8_t * currp, const int src_pitch, const uint8_t* prevp, const uint8_t *nextp, uint8_t *destp, int x,
  __m256i &temporal_threshold_vector, // already shifted to "signed" domain
  __m256i &spatial_threshold_vector // already shifted to "signed" domain
)
{
  const auto make_signed_word = _mm256_set1_epi16(0x8000); // int16 support is better than of uint16 (cmp, etc...)
  // +1: center of 3x3 pixels [+0,+1,+2]
  auto b_orig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + 1 * sizeof(uint16_t)));
  auto pbt_orig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prevp + x + 1 * sizeof(uint16_t)));
  auto nbt_orig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(nextp + x + 1 * sizeof(uint16_t)));

  auto b = _mm256_add_epi16(b_orig, make_signed_word);
  auto pbt = _mm256_add_epi16(pbt_orig, make_signed_word);
  auto nbt = _mm256_add_epi16(nbt_orig, make_signed_word);
  // int b = *currp, pbt = *prevp++, nbt = *nextp++;
  // int pdiff = pbt - b, ndiff = nbt - b;
  // if ((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
  // --> if ((pbt < b && nbt < b) || (pbt > b && nbt > b))
  auto pbt_lessthan_b = _mm256_cmpgt_epi16(b, pbt); // FF where b > pbt. No lt --> gt with exchanged parameters
  auto nbt_lessthan_b = _mm256_cmpgt_epi16(b, nbt); // FF where b > nbt. No lt --> gt with exchanged parameters
  auto pbt_greaterthan_b = _mm256_cmpgt_epi16(pbt, b); // FF where pbt > b
  auto nbt_greaterthan_b = _mm256_cmpgt_epi16(nbt, b); // FF where nbt > b
  auto both_less = _mm256_and_si256(pbt_lessthan_b, nbt_lessthan_b);
  auto both_greater = _mm256_and_si256(pbt_greaterthan_b, nbt_greaterthan_b);
  auto mask_either_is_true = _mm256_or_si256(both_less, both_greater);
  // mask will be used at the final decision. Where FF: keep computed result. 00: keep original pixel (dst=curr)

    // int pb1 = currp[-src_pitch - 1], pb2 = currp[-src_pitch], pb3 = currp[-src_pitch + 1];
    // int b1 = currp[-1], /*b = currp[0], */b2 = currp[1];
    // int nb1 = currp[src_pitch - 1], nb2 = currp[src_pitch], nb3 = currp[src_pitch + 1];

  auto pb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x - src_pitch + 0 * sizeof(uint16_t)));
  auto pb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x - src_pitch + 1 * sizeof(uint16_t)));
  auto pb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x - src_pitch + 2 * sizeof(uint16_t)));

  auto b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + 0 * sizeof(uint16_t)));
  auto b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + 2 * sizeof(uint16_t)));

  auto nb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + src_pitch + 0 * sizeof(uint16_t)));
  auto nb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + src_pitch + 1 * sizeof(uint16_t)));
  auto nb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currp + x + src_pitch + 2 * sizeof(uint16_t)));

  // int sum = b, cnt = 1;
  auto zero = _mm256_setzero_si256();
  auto sum_lo = _mm256_unpacklo_epi16(b_orig, zero);
  auto sum_hi = _mm256_unpackhi_epi16(b_orig, zero);
  auto cnt = _mm256_set1_epi16(1);

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

  auto cnt_lo = _mm256_unpacklo_epi16(cnt, zero);
  auto cnt_hi = _mm256_unpackhi_epi16(cnt, zero);
  // Difference from SSE4.1 and C: floating point division
  // SSE2: sum / count -> (int)((float)sum * 1.0f/(float)count + 0.5f)
  const auto rounder_half = _mm256_set1_ps(0.5f);
  // lower 4 pixels
  auto fcnt_lo = _mm256_cvtepi32_ps(cnt_lo);
  auto fsum_lo = _mm256_cvtepi32_ps(sum_lo);
  auto mulres_lo = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(fsum_lo, _mm256_rcp_ps(fcnt_lo)), rounder_half));
  // upper 4 pixels
  auto fcnt_hi = _mm256_cvtepi32_ps(cnt_hi);
  auto fsum_hi = _mm256_cvtepi32_ps(sum_hi);
  auto mulres_hi = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(fsum_hi, _mm256_rcp_ps(fcnt_hi)), rounder_half));

  // move back to 8x16 bits
  auto result = _mm256_packus_epi32(mulres_lo, mulres_hi);

  // decide if original pixel is kept
  auto finalres = _mm256_blendv_epi8(b_orig, result, mask_either_is_true); // true: second param, false: 1st param

  _mm256_storeu_si256(reinterpret_cast<__m256i *>(destp + x + 1 * sizeof(uint16_t)), finalres);
}

// Spatial Temporal
void fluxST_avx2_uint16(const uint8_t* currp, const int src_pitch,
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

  __m256i temporal_threshold_vector = _mm256_set1_epi16(temporal_threshold - 0x8000); // move to signed int16 domain;
  __m256i spatial_threshold_vector = _mm256_set1_epi16(spatial_threshold - 0x8000); // move to signed int16 domain;

  const int wmod16 = xcnt / 16 * 16;
  const int rest = xcnt - wmod16;

  for (int y = 0; y < height; y++)
  {
    reinterpret_cast<uint16_t*>(destp)[0] = reinterpret_cast<const uint16_t*>(currp)[0]; // Copy left edge

    for (int x = 0; x < wmod16; x += 16)
      fluxST_core_avx2_uint16(currp, src_pitch, prevp, nextp, destp, x * sizeof(uint16_t), temporal_threshold_vector, spatial_threshold_vector);
    // do rest
    if (rest > 0)
      fluxST_core_avx2_uint16(currp, src_pitch, prevp, nextp, destp, (xcnt - 16) * sizeof(uint16_t), temporal_threshold_vector, spatial_threshold_vector);

    reinterpret_cast<uint16_t*>(destp)[width - 1] = reinterpret_cast<const uint16_t*>(currp)[width - 1]; // Copy right edge

    currp += src_pitch;
    prevp += prv_pitch;
    nextp += nxt_pitch;
    destp += dst_pitch;
  } // for y
  _mm256_zeroupper();
}


