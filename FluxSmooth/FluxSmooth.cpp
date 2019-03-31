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

//Please only define one of these symbols.
#define COMPILE_SSE2_VERSION
//#define COMPILE_SSSE3_VERSION

#define DO_SPATIAL
#include "YUY2.h"
#include "YV12-SSE-only.h"

#undef DO_SPATIAL
#include "YUY2.h"
#include "YV12-SSE-only.h"


AVSValue __cdecl Create_FluxSmoothT(AVSValue args, void * user_data, IScriptEnvironment * env)
{
	enum ARGS { CLIP, TEMPORAL_THRESHOLD };

	PClip clip = args[CLIP].AsClip();
	int temporal_threshold = args[TEMPORAL_THRESHOLD].AsInt(7);

	if (temporal_threshold < 0)
		env->ThrowError("FluxSmoothT: temporal_threshold must be >= 0");

	const VideoInfo & vi = clip->GetVideoInfo();

	if (vi.IsYUY2())
		return new FluxSmoothT_YUY2(clip, temporal_threshold, env);
	else if (vi.IsYV12())
		return new FluxSmoothT_YV12(clip, temporal_threshold, env);
	else
		env->ThrowError("FluxSmoothT: Clip must be in YUY2 or YV12 format");

	return 0; // Unreached
}

AVSValue __cdecl Create_FluxSmoothST(AVSValue args, void * user_data, IScriptEnvironment * env)
{
	enum ARGS { CLIP, TEMPORAL_THRESHOLD, SPATIAL_THRESHOLD };

	PClip clip = args[CLIP].AsClip();
	int temporal_threshold = args[TEMPORAL_THRESHOLD].AsInt(7);
	int spatial_threshold = args[SPATIAL_THRESHOLD].AsInt(7);

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
		return new FluxSmoothST_YV12(clip, temporal_threshold, spatial_threshold, env);
	else
		env->ThrowError("FluxSmoothST: Clip must be in YUY2 or YV12 format");

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
#ifdef COMPILE_SSE2_VERSION
	if (!(env->GetCPUFlags() & CPUF_SSE2)) {
		env->ThrowError("This version of FluxSmooth requires a CPU with SSE2");
	}
#endif
#ifdef COMPILE_SSSE3_VERSION
	if (!(env->GetCPUFlags() & CPUF_SSSE3)) {
		env->ThrowError("This version of FluxSmooth requires a CPU with SSSE3 (not just SSE3)");
	}
#endif

	env->AddFunction("FluxSmoothT", "c[temporal_threshold]i",
		Create_FluxSmoothT, 0);
	env->AddFunction("FluxSmoothST", "c[temporal_threshold]i[spatial_threshold]i",
		Create_FluxSmoothST, 0);
	return "FluxSmooth: Smoothing of fluctuations";
}
