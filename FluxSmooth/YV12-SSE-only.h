// FluxSmooth
// Avisynth filter for smoothing of fluctuations
//
// By Ross Thomas <ross@grinfinity.com>
//
// There is no copyright on this code, and there are no conditions
// on its distribution or use. Do with it what you will.
//
// Overhauled and new optimized 64-bit versions by Devin Gardner - 2010/11/30
//
// YV12 version designed to use just one of the SSE-optimized versions.

#include <emmintrin.h>		//needed for __m128i

#ifdef DO_SPATIAL
#define CLASS_NAME		FluxSmoothST_YV12
#else
#define CLASS_NAME		FluxSmoothT_YV12
#endif

#ifdef COMPILE_SSSE3_VERSION
#define CALL_FILTER		DoFilter_SSSE3
#else
#ifdef COMPILE_SSE2_VERSION
#define CALL_FILTER		DoFilter_SSE2
#endif
#endif


class CLASS_NAME : public GenericVideoFilter
{
public:
#ifdef DO_SPATIAL
	CLASS_NAME(PClip _child, int temporal_threshold,
			   int spatial_threshold, IScriptEnvironment * env);
#else
	CLASS_NAME(PClip _child, int temporal_threshold, IScriptEnvironment * env);
#endif

	~CLASS_NAME();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment * env);


private:
#ifdef COMPILE_SSE2_VERSION
	void DoFilter_SSE2(const BYTE * currp, const long long src_pitch, const BYTE * prevp,
					   const BYTE * nextp, const long long prv_pitch, const long long nxt_pitch,
					   BYTE * destp, const long long dst_pitch, const int row_size, int height);
#endif
#ifdef COMPILE_SSSE3_VERSION
	void DoFilter_SSSE3(const BYTE * currp, const long long src_pitch, const BYTE * prevp,
						const BYTE * nextp, const long long prv_pitch, const long long nxt_pitch,
						BYTE * destp, const long long dst_pitch, const int row_size, int height);
#endif

private:
	int spatial_threshold;
	int temporal_threshold;
	short scaletab[16];
	__m128i spat_thresh;		//filled with 16-bit copies of spatial_threshold
	__m128i temp_thresh;		//filled with 16-bit copies of temporal_threshold
#ifdef COMPILE_SSSE3_VERSION
	__m128i scaletab_low;		//elements 0->7 of scaletab; element 0 must have a value of 0.
	__m128i scaletab_high;		//elements 8->15 of scaletab (12->15 will never be used)
#endif
};




#ifdef DO_SPATIAL
CLASS_NAME::CLASS_NAME(PClip _child, int _temporal_threshold,
					   int _spatial_threshold, IScriptEnvironment * env)
	: GenericVideoFilter(_child), temporal_threshold(_temporal_threshold),
	  spatial_threshold(_spatial_threshold)
{
	assert(_child);
	assert(temporal_threshold >= -1);
	assert(spatial_threshold >= -1);
	assert(!((-1 == temporal_threshold) && (-1 == spatial_threshold)));
#else
CLASS_NAME::CLASS_NAME(PClip _child, int _temporal_threshold, IScriptEnvironment * env)
	: GenericVideoFilter(_child), temporal_threshold(_temporal_threshold),
	  spatial_threshold(-1)
{
	assert(_child);
	assert(temporal_threshold >= 0);
#endif
	assert(env);

	scaletab[0] = 0;
	scaletab[1] = 32767;
	for(int i = 2; i < 16; ++i)
		scaletab[i] = (int)(32768.0 / i + 0.5);

	for(int i = 0; i < 8; i++) {
		((short*)&spat_thresh)[i] = (short)spatial_threshold;
		((short*)&temp_thresh)[i] = (short)temporal_threshold;
	}
#ifdef COMPILE_SSSE3_VERSION
	for(int i = 0; i < 8; i++)	((short*)&scaletab_low)[i] = scaletab[i];	//Important: the first word here MUST be 0!
	for(int i = 0; i < 4; i++)	((short*)&scaletab_high)[i] = scaletab[i + 8];
	for(int i = 4; i < 8; i++)	((short*)&scaletab_high)[i] = 0;
#endif

	SetCacheHints(CACHE_RANGE, 1);
}


CLASS_NAME::~CLASS_NAME() {

}


PVideoFrame __stdcall CLASS_NAME::GetFrame(int n, IScriptEnvironment * env)
{
	const BYTE * srcp;
	const BYTE * prevp;
	const BYTE * nextp;
	BYTE * destp;
	int src_pitch, dst_pitch, prv_pitch, nxt_pitch, row_size, height;

	assert(n >= 0 && n < vi.num_frames);
	assert(env);

	PVideoFrame currf = child->GetFrame(n, env);
	assert(currf);

	PVideoFrame destf = env->NewVideoFrame(vi);
	assert(destf);

	if (n == 0 || n == vi.num_frames - 1)
	{
		srcp = currf->GetReadPtr(PLANAR_Y);
		src_pitch = currf->GetPitch(PLANAR_Y);
		height = currf->GetHeight(PLANAR_Y);
		row_size = currf->GetRowSize(PLANAR_Y);
		destp = destf->GetWritePtr(PLANAR_Y);
		dst_pitch = destf->GetPitch(PLANAR_Y);
		env->BitBlt(destp, dst_pitch, srcp, src_pitch, row_size, height);

		srcp = currf->GetReadPtr(PLANAR_U);
		src_pitch = currf->GetPitch(PLANAR_U);
		height = currf->GetHeight(PLANAR_U);
		row_size = currf->GetRowSize(PLANAR_U);
		destp = destf->GetWritePtr(PLANAR_U);
		dst_pitch = destf->GetPitch(PLANAR_U);
		env->BitBlt(destp, dst_pitch, srcp, src_pitch, row_size, height);

		srcp = currf->GetReadPtr(PLANAR_V);
		src_pitch = currf->GetPitch(PLANAR_V);
		destp = destf->GetWritePtr(PLANAR_V);
		dst_pitch = destf->GetPitch(PLANAR_V);
		env->BitBlt(destp, dst_pitch, srcp, src_pitch, row_size, height);

		return destf;
	}

	PVideoFrame prevf = child->GetFrame(n - 1, env);
	assert(prevf);

	PVideoFrame nextf = child->GetFrame(n + 1, env);
	assert(nextf);

	//The compiler will unroll this loop if it's faster
	for(int c = 0; c < 3; c++)
	{
		switch (c) {
			case 0:
				dst_pitch = destf->GetPitch(PLANAR_Y);
				src_pitch = currf->GetPitch(PLANAR_Y);
				prv_pitch = prevf->GetPitch(PLANAR_Y);
				nxt_pitch = nextf->GetPitch(PLANAR_Y);
				row_size = currf->GetRowSize(PLANAR_Y);
				height = currf->GetHeight(PLANAR_Y);
				srcp = currf->GetReadPtr(PLANAR_Y);
				prevp = prevf->GetReadPtr(PLANAR_Y);
				nextp = nextf->GetReadPtr(PLANAR_Y);
				destp = destf->GetWritePtr(PLANAR_Y);
				break;
			case 1:
				dst_pitch = destf->GetPitch(PLANAR_U);
				src_pitch = currf->GetPitch(PLANAR_U);
				prv_pitch = prevf->GetPitch(PLANAR_U);
				nxt_pitch = nextf->GetPitch(PLANAR_U);
				row_size = currf->GetRowSize(PLANAR_U);
				height = currf->GetHeight(PLANAR_U);
				srcp = currf->GetReadPtr(PLANAR_U);
				prevp = prevf->GetReadPtr(PLANAR_U);
				nextp = nextf->GetReadPtr(PLANAR_U);
				destp = destf->GetWritePtr(PLANAR_U);
				break;
			case 2:
				dst_pitch = destf->GetPitch(PLANAR_V);
				src_pitch = currf->GetPitch(PLANAR_V);
				prv_pitch = prevf->GetPitch(PLANAR_V);
				nxt_pitch = nextf->GetPitch(PLANAR_V);
				row_size = currf->GetRowSize(PLANAR_V);
				height = currf->GetHeight(PLANAR_V);
				srcp = currf->GetReadPtr(PLANAR_V);
				prevp = prevf->GetReadPtr(PLANAR_V);
				nextp = nextf->GetReadPtr(PLANAR_V);
				destp = destf->GetWritePtr(PLANAR_V);
				break;
			default:
				__assume(0);
		}

#ifdef DO_SPATIAL
		memcpy(destp + dst_pitch * (height - 1),
			srcp + src_pitch * (height - 1), row_size);
		memcpy(destp, srcp, row_size);

		srcp  += src_pitch;
		prevp += prv_pitch;
		nextp += nxt_pitch;
		destp += dst_pitch;
		height -= 2;
#endif

		CALL_FILTER(srcp, src_pitch, prevp, nextp, prv_pitch,
					nxt_pitch, destp, dst_pitch, row_size, height);
	}

	return destf;
}




#ifdef COMPILE_SSSE3_VERSION
#include "YV12_SSSE3.h"
#endif

#ifdef COMPILE_SSE2_VERSION
#include "YV12_SSE2.h"
#endif


#undef CLASS_NAME
#undef CALL_FILTER
