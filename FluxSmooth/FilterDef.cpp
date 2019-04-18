// FluxSmooth
// Avisynth filter for spatio-temporal smoothing of fluctuations
//
// By Ross Thomas <ross@grinfinity.com>
//
// There is no copyright on this code, and there are no conditions
// on its distribution or use. Do with it what you will.

#include <windows.h>
#include <cassert>
#include "avisynth.h"
#include "FluxSmooth.h"
#include <algorithm>

FluxSmoothST::FluxSmoothST(PClip _child, int _temporal_threshold, int _spatial_threshold, bool _luma, bool _chroma, int _opt, IScriptEnvironment * env)
  : GenericVideoFilter(_child),
  spatial_threshold(_spatial_threshold),
  temporal_threshold(_temporal_threshold),
  opt(_opt)
{
  assert(temporal_threshold >= -1);
  assert(spatial_threshold >= -1);
  assert(!((-1 == temporal_threshold) && (-1 == spatial_threshold)));
  assert(env);

  // division table 1/1, 1/2, ... 1/11
  // only 1..11 is valid
  scaletab[0] = 0;
  scaletab[1] = 32767;
  for (int i = 2; i < 16; ++i)
    scaletab[i] = (int)(32768.0 / i + 0.5);

  const bool goodAVX512 = ((env->GetCPUFlags() & CPUF_AVX512F) == CPUF_AVX512F) && (env->GetCPUFlags() & CPUF_AVX512BW) == CPUF_AVX512BW;

#ifndef FLUXSMOOTH_AVX512_ENABLED
  if (opt == USE_OPT_AVX512)
    env->ThrowError("FluxSmoothST: cannot apply opt: this DLL version does not support AVX512");
#endif

  if (opt == USE_OPT_AVX512 && !goodAVX512)
    env->ThrowError("FluxSmoothST: cannot apply opt: AVX512F and AVX512BW is needed");
  if (opt == USE_OPT_AVX2 && !(env->GetCPUFlags() & CPUF_AVX2))
    env->ThrowError("FluxSmoothST: cannot apply opt: AVX2 is not supported");
  if (opt == USE_OPT_SSE41 && !(env->GetCPUFlags() & CPUF_SSE4_1))
    env->ThrowError("FluxSmoothST: cannot apply opt: SSE4.1 is not supported");
  if (opt == USE_OPT_SSE2 && !(env->GetCPUFlags() & CPUF_SSE2))
    env->ThrowError("FluxSmoothST: cannot apply opt: SSE2 is not supported");

  const int *current_planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
  int planecount = min(vi.NumComponents(), 3);
  int bits_per_pixel = vi.BitsPerComponent();

  for (int i = 0; i < planecount; i++) {
    if (vi.IsRGB())
      processPlane[i] = true;
    else if (i == 0) // Y
      processPlane[i] = _luma;
    else
      processPlane[i] = _chroma;

    const int actual_width = vi.width >> vi.GetPlaneWidthSubsampling(current_planes[i]);
    if (bits_per_pixel == 8) {
#ifdef FLUXSMOOTH_AVX512_ENABLED
      if ((actual_width >= 1 + 64 + 1) && ((goodAVX512 && opt < 0) || opt >= USE_OPT_AVX512))
        proc_ST[i] = fluxST_avx512;
      else
#endif
      if ((actual_width >= 1 + 32 + 1) && (((env->GetCPUFlags() & CPUF_AVX2) == CPUF_AVX2 && opt < 0) || opt >= USE_OPT_AVX2))
        proc_ST[i] = fluxST_avx2;
      else if ((actual_width >= 1 + 16 + 1) && (((env->GetCPUFlags() & CPUF_SSE4_1) == CPUF_SSE4_1 && opt < 0) || opt >= USE_OPT_SSE41))
        proc_ST[i] = fluxST_sse41;
      else if ((actual_width >= 1 + 16 + 1) && (((env->GetCPUFlags() & CPUF_SSE2) == CPUF_SSE2 && opt < 0) || opt >= USE_OPT_SSE2))
        proc_ST[i] = fluxST_sse2;
      else
        proc_ST[i] = fluxST_C<uint8_t>;
    }
    else {
#ifdef FLUXSMOOTH_AVX512_ENABLED
      if ((actual_width >= 1 + 32 + 1) && ((goodAVX512 && opt < 0) || opt >= USE_OPT_AVX512))
        proc_ST[i] = fluxST_avx512_uint16;
      else
#endif
      if ((actual_width >= 1 + 16 + 1) && (((env->GetCPUFlags() & CPUF_AVX2) == CPUF_AVX2 && opt < 0) || opt >= USE_OPT_AVX2))
        proc_ST[i] = fluxST_avx2_uint16;
      else if ((actual_width >= 1 + 8 + 1) && (((env->GetCPUFlags() & CPUF_SSE4_1) == CPUF_SSE4_1 && opt < 0) || opt >= USE_OPT_SSE41))
        proc_ST[i] = fluxST_sse41_uint16;
      else
        proc_ST[i] = fluxST_C<uint16_t>;
    }
  }
}

static void copy_plane(PVideoFrame &destf, PVideoFrame &currf, int plane, IScriptEnvironment *env) {
  const uint8_t* srcp = currf->GetReadPtr(plane);
  int src_pitch = currf->GetPitch(plane);
  int height = currf->GetHeight(plane);
  int row_size = currf->GetRowSize(plane);
  uint8_t* destp = destf->GetWritePtr(plane);
  int dst_pitch = destf->GetPitch(plane);
  env->BitBlt(destp, dst_pitch, srcp, src_pitch, row_size, height);
}

PVideoFrame __stdcall FluxSmoothST::GetFrame(int n, IScriptEnvironment * env)
{
  const uint8_t* srcp;
  const uint8_t* prevp;
  const uint8_t* nextp;
  uint8_t* destp;
  int src_pitch, dst_pitch, prv_pitch, nxt_pitch, row_size, height;

  PVideoFrame currf = child->GetFrame(n, env);
  PVideoFrame destf = env->NewVideoFrame(vi);

  const int *current_planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

  if (n == 0 || n == vi.num_frames - 1)
  {
    // 1st or last: not temporal
    for (int i = 0; i < vi.NumComponents(); i++)
    {
      const int plane = current_planes[i];
      copy_plane(destf, currf, plane, env);
    }
    return destf;
  }

  PVideoFrame prevf = child->GetFrame(n - 1, env);
  PVideoFrame nextf = child->GetFrame(n + 1, env);

  int planecount = min(vi.NumComponents(), 3);

  for (int i = 0; i < planecount; i++)
  {
    const int plane = current_planes[i];
    if (processPlane[i]) {
      dst_pitch = destf->GetPitch(plane);
      src_pitch = currf->GetPitch(plane);
      prv_pitch = prevf->GetPitch(plane);
      nxt_pitch = nextf->GetPitch(plane);
      row_size = currf->GetRowSize(plane);
      const int width = row_size / vi.ComponentSize();
      height = currf->GetHeight(plane);
      srcp = currf->GetReadPtr(plane);
      prevp = prevf->GetReadPtr(plane);
      nextp = nextf->GetReadPtr(plane);
      destp = destf->GetWritePtr(plane);

      // copy top and bottom lines
      memcpy(destp + dst_pitch * (height - 1), srcp + src_pitch * (height - 1), row_size);
      memcpy(destp, srcp, row_size);
      // skip to 2nd line
      srcp += src_pitch;
      prevp += prv_pitch;
      nextp += nxt_pitch;
      destp += dst_pitch;
      height -= 2; // two lines less

      const int bits_per_pixel = vi.BitsPerComponent();

      proc_ST[i](srcp, src_pitch, prevp, prv_pitch, nextp, nxt_pitch, destp, dst_pitch, width, height, temporal_threshold << (bits_per_pixel - 8), spatial_threshold << (bits_per_pixel - 8), scaletab);
    }
    else {
      copy_plane(destf, currf, plane, env);
    }
  }
  // copy alpha
  if (vi.NumComponents() == 4) {
    const int plane = PLANAR_A;
    copy_plane(destf, currf, plane, env);
  }

  return destf;
}

FluxSmoothT::FluxSmoothT(PClip _child, int _temporal_threshold, bool _luma, bool _chroma, int _opt, IScriptEnvironment * env)
  : GenericVideoFilter(_child), temporal_threshold(_temporal_threshold),
  opt(_opt)
{
  assert(temporal_threshold >= -1);
  assert(!((-1 == temporal_threshold)));
  assert(env);

  // division table 1/1, 1/2, ... 1/11
  // only 1..11 is valid
  scaletab[0] = 0;
  scaletab[1] = 32767;
  for (int i = 2; i < 16; ++i)
    scaletab[i] = (int)(32768.0 / i + 0.5);

  const bool goodAVX512 = ((env->GetCPUFlags() & CPUF_AVX512F) == CPUF_AVX512F) && (env->GetCPUFlags() & CPUF_AVX512BW) == CPUF_AVX512BW;

#ifndef FLUXSMOOTH_AVX512_ENABLED
  if (opt == USE_OPT_AVX512)
    env->ThrowError("FluxSmoothT: cannot apply opt: this DLL version does not support AVX512");
#endif

  if (opt == USE_OPT_AVX512 && !goodAVX512)
    env->ThrowError("FluxSmoothT: cannot apply opt: AVX512F and AVX512BW is needed");
  if (opt == USE_OPT_AVX2 && !(env->GetCPUFlags() & CPUF_AVX2))
    env->ThrowError("FluxSmoothT: cannot apply opt: AVX2 is not supported");
  if (opt == USE_OPT_SSE41 && !(env->GetCPUFlags() & CPUF_SSE4_1))
    env->ThrowError("FluxSmoothT: cannot apply opt: SSE4.1 is not supported");
  if (opt == USE_OPT_SSE2 && !(env->GetCPUFlags() & CPUF_SSE2))
    env->ThrowError("FluxSmoothT: cannot apply opt: SSE2 is not supported");

  const int *current_planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
  int planecount = min(vi.NumComponents(), 3);
  int bits_per_pixel = vi.BitsPerComponent();

  for (int i = 0; i < planecount; i++) {
    if (vi.IsRGB())
      processPlane[i] = true;
    else if (i == 0) // Y
      processPlane[i] = _luma;
    else
      processPlane[i] = _chroma;

    const int actual_width = vi.width >> vi.GetPlaneWidthSubsampling(current_planes[i]);

    if (bits_per_pixel == 8) {
#ifdef FLUXSMOOTH_AVX512_ENABLED
      if ((actual_width >= 64) && ((goodAVX512 && opt < 0) || opt >= USE_OPT_AVX512))
        proc_T[i] = fluxT_avx512;
      else
#endif
      if ((actual_width >= 32) && (((env->GetCPUFlags() & CPUF_AVX2) == CPUF_AVX2 && opt < 0) || opt >= USE_OPT_AVX2))
        proc_T[i] = fluxT_avx2;
      else if ((actual_width >= 16) && (((env->GetCPUFlags() & CPUF_SSE4_1) == CPUF_SSE4_1 && opt < 0) || opt >= USE_OPT_SSE41))
        proc_T[i] = fluxT_sse41;
      else if ((actual_width >= 16) && (((env->GetCPUFlags() & CPUF_SSE2) == CPUF_SSE2 && opt < 0) || opt >= USE_OPT_SSE2))
        proc_T[i] = fluxT_sse2;
      else
        proc_T[i] = fluxT_C<uint8_t>;
    }
    else {
#ifdef FLUXSMOOTH_AVX512_ENABLED
      if ((actual_width >= 32) && ((goodAVX512 && opt < 0) || opt >= USE_OPT_AVX512))
        proc_T[i] = fluxT_avx512_uint16;
      else
#endif
      if ((actual_width >= 16) && (((env->GetCPUFlags() & CPUF_AVX2) == CPUF_AVX2 && opt < 0) || opt >= USE_OPT_AVX2))
        proc_T[i] = fluxT_avx2_uint16;
      else if ((actual_width >= 8) && (((env->GetCPUFlags() & CPUF_SSE4_1) == CPUF_SSE4_1 && opt < 0) || opt >= USE_OPT_SSE41))
        proc_T[i] = fluxT_sse41_uint16;
      else
        proc_T[i] = fluxT_C<uint16_t>;
    }
  }
}

PVideoFrame __stdcall FluxSmoothT::GetFrame(int n, IScriptEnvironment * env)
{
  const uint8_t* srcp;
  const uint8_t* prevp;
  const uint8_t* nextp;
  uint8_t* destp;
  int src_pitch, dst_pitch, prv_pitch, nxt_pitch, row_size, height;

  PVideoFrame currf = child->GetFrame(n, env);
  PVideoFrame destf = env->NewVideoFrame(vi);

  const int *current_planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

  if (n == 0 || n == vi.num_frames - 1)
  {
    // 1st or last: simple copy
    for (int i = 0; i < vi.NumComponents(); i++)
    {
      const int plane = current_planes[i];
      copy_plane(destf, currf, plane, env);
    }
    return destf;
  }

  PVideoFrame prevf = child->GetFrame(n - 1, env);
  PVideoFrame nextf = child->GetFrame(n + 1, env);

  int planecount = min(vi.NumComponents(), 3);

  for (int i = 0; i < planecount; i++)
  {
    const int plane = current_planes[i];
    if (processPlane[i]) {
      dst_pitch = destf->GetPitch(plane);
      src_pitch = currf->GetPitch(plane);
      prv_pitch = prevf->GetPitch(plane);
      nxt_pitch = nextf->GetPitch(plane);
      row_size = currf->GetRowSize(plane);
      const int width = row_size / vi.ComponentSize();
      height = currf->GetHeight(plane);
      srcp = currf->GetReadPtr(plane);
      prevp = prevf->GetReadPtr(plane);
      nextp = nextf->GetReadPtr(plane);
      destp = destf->GetWritePtr(plane);

      const int bits_per_pixel = vi.BitsPerComponent();

      proc_T[i](srcp, src_pitch, prevp, prv_pitch, nextp, nxt_pitch, destp, dst_pitch, width, height, temporal_threshold << (bits_per_pixel - 8), scaletab);
    }
    else {
      copy_plane(destf, currf, plane, env);
    }
  }
  // copy alpha
  if (vi.NumComponents() == 4) {
    const int plane = PLANAR_A;
    copy_plane(destf, currf, plane, env);
  }

  return destf;
}

AVSValue __cdecl Create_FluxSmoothT(AVSValue args, void * user_data, IScriptEnvironment * env)
{
  enum ARGS { CLIP, TEMPORAL_THRESHOLD, LUMA, CHROMA, OPT };

  PClip clip = args[CLIP].AsClip();
  int temporal_threshold = args[TEMPORAL_THRESHOLD].AsInt(7);
  bool luma = args[LUMA].AsBool(true);
  bool chroma = args[CHROMA].AsBool(true);
  int opt = args[OPT].AsInt(-1);

  if (temporal_threshold < 0)
    env->ThrowError("FluxSmoothT: temporal_threshold must be >= 0");

  const VideoInfo & vi = clip->GetVideoInfo();

  // YUY2 support only through YV16 autoconversion
  if (vi.IsYUY2()) {
    AVSValue new_args[1] = { clip };
    clip = env->Invoke("ConvertToYV16", AVSValue(new_args, 1)).AsClip();
    clip = new FluxSmoothT(clip, temporal_threshold, luma, chroma, opt, env);
    AVSValue new_args2[1] = { clip };
    clip = env->Invoke("ConvertToYUY2", AVSValue(new_args2, 1)).AsClip();
    return clip;
  }

  if (vi.BitsPerComponent() == 32)
    env->ThrowError("FluxSmoothT: 32 bit float formats not supported");

  if (vi.IsY() || vi.IsYV411() || vi.Is420() || vi.Is422() || vi.Is444() || vi.IsPlanarRGB() || vi.IsPlanarRGBA())
    return new FluxSmoothT(clip, temporal_threshold, luma, chroma, opt, env);
  else
    env->ThrowError("FluxSmoothT: Clip must be in Y or planar YUV(A), RGB(A) or YUY2 format (8-16 bits)");

  return 0; // Unreached
}

AVSValue __cdecl Create_FluxSmoothST(AVSValue args, void * user_data, IScriptEnvironment * env)
{
  enum ARGS { CLIP, TEMPORAL_THRESHOLD, SPATIAL_THRESHOLD, LUMA, CHROMA, OPT };

  PClip clip = args[CLIP].AsClip();
  int temporal_threshold = args[TEMPORAL_THRESHOLD].AsInt(7);
  int spatial_threshold = args[SPATIAL_THRESHOLD].AsInt(7);
  bool luma = args[LUMA].AsBool(true);
  bool chroma = args[CHROMA].AsBool(true);
  int opt = args[OPT].AsInt(-1);

  if (temporal_threshold < -1)
    env->ThrowError("FluxSmoothST: temporal_threshold must be >= -1");
  if (spatial_threshold < -1)
    env->ThrowError("FluxSmoothST: spatial_threshold must be >= -1");
  if (-1 == temporal_threshold && -1 == spatial_threshold)
    env->ThrowError("FluxSmoothST: Both thresholds cannot be -1");

  const VideoInfo & vi = clip->GetVideoInfo();

  // YUY2 support only through YV16 autoconversion
  if (vi.IsYUY2()) {
    AVSValue new_args[1] = { clip };
    clip = env->Invoke("ConvertToYV16", AVSValue(new_args, 1)).AsClip();
    clip = new FluxSmoothST(clip, temporal_threshold, spatial_threshold, luma, chroma, opt, env);
    AVSValue new_args2[1] = { clip };
    clip = env->Invoke("ConvertToYUY2", AVSValue(new_args2, 1)).AsClip();
    return clip;
  }

  if (vi.BitsPerComponent() == 32)
    env->ThrowError("FluxSmoothST: 32 bit float formats not supported");

  if (vi.IsY() || vi.IsYV411() || vi.Is420() || vi.Is422() || vi.Is444() || vi.IsPlanarRGB() || vi.IsPlanarRGBA())
    return new FluxSmoothST(clip, temporal_threshold, spatial_threshold, luma, chroma, opt, env);
  else
    env->ThrowError("FluxSmoothST: Clip must be in Y or planar YUV(A), RGB(A) or YUY2 format (8-16 bits)");

  return 0; // Unreached
}

/* New 2.6 requirement!!! */
// Declare and initialise server pointers static storage.
const AVS_Linkage *AVS_linkage = 0;

/* New 2.6 requirement!!! */
// DLL entry point called from LoadPlugin() to setup a user plugin.
extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
  /* New 2.6 requirement!!! */
  // Save the server pointers.
  AVS_linkage = vectors;
  env->AddFunction("FluxSmoothT", "c[temporal_threshold]i[luma]b[chroma]b[opt]i", Create_FluxSmoothT, 0);
  env->AddFunction("FluxSmoothST", "c[temporal_threshold]i[spatial_threshold]i[luma]b[chroma]b[opt]i", Create_FluxSmoothST, 0);
  return "FluxSmooth";
}
