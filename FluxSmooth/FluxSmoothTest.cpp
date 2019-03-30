// FluxSmooth
// Avisynth filter for spatio-temporal smoothing of fluctuations
//
// By Ross Thomas <ross@grinfinity.com>
//
// There is no copyright on this code, and there are no conditions
// on its distribution or use. Do with it what you will.
//
// Overhauled and new optimized 64-bit versions by Devin Gardner - 2010/11/30

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <cassert>
#include "avisynth.h"

#define INCLUDE_MMX_VER
#define INCLUDE_SSE2_VER
#define INCLUDE_SSSE3_VER
//#define USE_SSSE3_FP_METHOD

#define DO_SPATIAL
#include "YUY2.h"
#include "YV12-Test.h"

#undef DO_SPATIAL
#include "YUY2.h"
#include "YV12-Test.h"

AVSValue __cdecl Create_FluxSmoothT(AVSValue args, void * user_data, IScriptEnvironment * env)
{
	enum ARGS { CLIP, TEMPORAL_THRESHOLD, OPT };

	PClip clip = args[CLIP].AsClip();
	int temporal_threshold = args[TEMPORAL_THRESHOLD].AsInt(7);
	int opt = args[OPT].AsInt(0);

	if (temporal_threshold < 0)
		env->ThrowError("FluxSmoothT: temporal_threshold must be >= 0");

	const VideoInfo & vi = clip->GetVideoInfo();

	if (vi.IsYUY2())
		return new FluxSmoothT_YUY2(clip, temporal_threshold, env);
	else if (vi.IsYV12())
		return new FluxSmoothT_YV12(clip, temporal_threshold, opt, env);
	else
		env->ThrowError("FluxSmoothT: Clip must be in YUY2 or YV12 format");

	return 0; // Unreached
}

AVSValue __cdecl Create_FluxSmoothST(AVSValue args, void * user_data, IScriptEnvironment * env)
{
	enum ARGS { CLIP, TEMPORAL_THRESHOLD, SPATIAL_THRESHOLD, OPT };

	PClip clip = args[CLIP].AsClip();
	int temporal_threshold = args[TEMPORAL_THRESHOLD].AsInt(7);
	int spatial_threshold = args[SPATIAL_THRESHOLD].AsInt(7);
	int opt = args[OPT].AsInt(0);

	if (temporal_threshold < -1)
		env->ThrowError("FluxSmoothST: temporal_threshold must be >= -1");
	if (spatial_threshold < -1)
		env->ThrowError("FluxSmoothST: spatial_threshold must be >= -1");
	if (-1 == temporal_threshold && -1 == spatial_threshold)
		env->ThrowError("FluxSmoothST: Both thresholds cannot be -1");

	const VideoInfo & vi = clip->GetVideoInfo();

	if (vi.IsYUY2())
		return new FluxSmoothST_YUY2(clip, temporal_threshold, spatial_threshold, env);
	else if (vi.IsYV12())
		return new FluxSmoothST_YV12(clip, temporal_threshold, spatial_threshold, opt, env);
	else
		env->ThrowError("FluxSmoothST: Clip must be in YUY2 or YV12 format");

	return 0; // Unreached
}

extern "C" __declspec(dllexport) const char * __stdcall AvisynthPluginInit2(IScriptEnvironment * env)
{
	env->AddFunction("FluxSmoothT", "c[temporal_threshold]i[opt]i",
		Create_FluxSmoothT, 0);
	env->AddFunction("FluxSmoothST", "c[temporal_threshold]i[spatial_threshold]i[opt]i",
		Create_FluxSmoothST, 0);
	return "FluxSmooth: Testing version";
}
