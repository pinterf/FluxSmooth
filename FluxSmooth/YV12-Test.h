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
// YV12 Test version with universal optimizations.

#include <emmintrin.h>		//needed for __m128i

#ifdef INCLUDE_MMX_VER
#include <malloc.h>
#endif

#ifdef DO_SPATIAL
#define CLASS_NAME FluxSmoothST_YV12
#else
#define CLASS_NAME FluxSmoothT_YV12
#endif


class CLASS_NAME : public GenericVideoFilter
{
public:
#ifdef DO_SPATIAL
	CLASS_NAME(PClip _child, int temporal_threshold,
			   int spatial_threshold, int opt, IScriptEnvironment * env);
#else
	CLASS_NAME(PClip _child, int temporal_threshold, int opt, IScriptEnvironment * env);
#endif
	//Constructor sets member "optimization" to setting specified by "opt" if it is available.
	// If that specific version isn't available, the default C code is used; there is no falling
	// back to the next-best optimization.

	~CLASS_NAME();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment * env);


	//Settings to choose optimized version
	enum { USE_OPT_C = 0, USE_OPT_MMX = 1, USE_OPT_SSE2 = 2, USE_OPT_SSSE3 = 3, USE_OPT_SSE41 = 4 };

private:
	void DoFilter_C(const BYTE * currp, const long long src_pitch, const BYTE * prevp,
					const BYTE * nextp, const long long prv_pitch, const long long nxt_pitch,
					BYTE * destp, const long long dst_pitch, const int row_size, int height);
#ifdef INCLUDE_MMX_VER
	void DoFilter_MMX(const BYTE * currp, const long long src_pitch, const BYTE * prevp,
					  const BYTE * nextp, const long long prv_pitch, const long long nxt_pitch,
					  BYTE * destp, const long long dst_pitch, const int row_size, int height);
#endif
#ifdef INCLUDE_SSE2_VER
	void DoFilter_SSE2(const BYTE * currp, const long long src_pitch, const BYTE * prevp,
					   const BYTE * nextp, const long long prv_pitch, const long long nxt_pitch,
					   BYTE * destp, const long long dst_pitch, const int row_size, int height);
#endif
#ifdef INCLUDE_SSSE3_VER
	void DoFilter_SSSE3(const BYTE * currp, const long long src_pitch, const BYTE * prevp,
						const BYTE * nextp, const long long prv_pitch, const long long nxt_pitch,
						BYTE * destp, const long long dst_pitch, const int row_size, int height);
#endif

private:
	int spatial_threshold;
	int temporal_threshold;
	int optimization;
	short scaletab[16];
	__m128i spat_thresh;		//filled with 16-bit copies of spatial_threshold
	__m128i temp_thresh;		//filled with 16-bit copies of temporal_threshold
#ifdef INCLUDE_SSSE3_VER
	__m128i scaletab_low;		//elements 0->7 of scaletab; element 0 must have a value of 0.
	__m128i scaletab_high;		//elements 8->15 of scaletab (12->15 will never be used)
#endif
#ifdef INCLUDE_MMX_VER
	__int64 * scaletab_MMX;
#endif
};




#ifdef DO_SPATIAL
CLASS_NAME::CLASS_NAME(PClip _child, int _temporal_threshold,
					   int _spatial_threshold, int opt, IScriptEnvironment * env)
	: GenericVideoFilter(_child), temporal_threshold(_temporal_threshold),
	  spatial_threshold(_spatial_threshold), optimization(USE_OPT_C)
{
	assert(_child);
	assert(temporal_threshold >= -1);
	assert(spatial_threshold >= -1);
	assert(!((-1 == temporal_threshold) && (-1 == spatial_threshold)));
#else
CLASS_NAME::CLASS_NAME(PClip _child, int _temporal_threshold, int opt, IScriptEnvironment * env)
	: GenericVideoFilter(_child), temporal_threshold(_temporal_threshold),
	  spatial_threshold(-1), optimization(USE_OPT_C)
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
#ifdef INCLUDE_SSSE3_VER
	for(int i = 0; i < 8; i++)	((short*)&scaletab_low)[i] = scaletab[i];	//Important: the first word here MUST be 0!
	for(int i = 0; i < 4; i++)	((short*)&scaletab_high)[i] = scaletab[i + 8];
	for(int i = 4; i < 8; i++)	((short*)&scaletab_high)[i] = 0;
#endif

#ifdef INCLUDE_MMX_VER
	scaletab_MMX = NULL;
	if ((env->GetCPUFlags() & CPUF_INTEGER_SSE) && (opt == USE_OPT_MMX))
	{
		optimization = USE_OPT_MMX;
		scaletab_MMX = (__int64*) _aligned_malloc(8*65536, 64);		// sh0: Use aligned memory
		for(long long i = 0; i < 65536; i++)
		{
			scaletab_MMX[i] = ( (UINT64)scaletab[ i        & 15]       ) |
							  (((UINT64)scaletab[(i >>  4) & 15]) << 16) |
							  (((UINT64)scaletab[(i >>  8) & 15]) << 32) |
							  (((UINT64)scaletab[(i >> 12) & 15]) << 48);
		}
	}
#endif
	if (env->GetCPUFlags() & CPUF_SSE2)
	{
#ifdef INCLUDE_SSE2_VER
		if (opt == USE_OPT_SSE2)
			optimization = USE_OPT_SSE2;
#endif
#ifdef INCLUDE_SSSE3_VER
		if (env->GetCPUFlags() & CPUF_SSSE3)
		{
			if (opt == USE_OPT_SSSE3)
				optimization = USE_OPT_SSSE3;
		}
#endif
	}

	SetCacheHints(CACHE_RANGE, 1);
}


CLASS_NAME::~CLASS_NAME() {
#ifdef INCLUDE_MMX_VER
	if (scaletab_MMX)
	    _aligned_free(scaletab_MMX);
#endif
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

		switch (optimization)
		{
#ifdef INCLUDE_SSSE3_VER
			case USE_OPT_SSSE3:
				DoFilter_SSSE3(srcp, src_pitch, prevp, nextp, prv_pitch,
							   nxt_pitch, destp, dst_pitch, row_size, height);
				break;
#endif
#ifdef INCLUDE_SSE2_VER
			case USE_OPT_SSE2:
				DoFilter_SSE2(srcp, src_pitch, prevp, nextp, prv_pitch,
							  nxt_pitch, destp, dst_pitch, row_size, height);
				break;
#endif
#ifdef INCLUDE_MMX_VER
			case USE_OPT_MMX:
				DoFilter_MMX(srcp, src_pitch, prevp, nextp, prv_pitch,
							 nxt_pitch, destp, dst_pitch, row_size, height);
				break;
#endif
			case USE_OPT_C:
			default:
				DoFilter_C(srcp, src_pitch, prevp, nextp, prv_pitch,
						   nxt_pitch, destp, dst_pitch, row_size, height);
				break;
		}
	}

	return destf;
}




void CLASS_NAME::DoFilter_C( const BYTE * currp, const long long src_pitch,
							 const BYTE * prevp, const BYTE * nextp,
							 const long long prv_pitch, const long long nxt_pitch,
							 BYTE * destp, const long long dst_pitch,
							 const int row_size, int height )
{
#ifdef _DEBUG
	const BYTE *currp_end = currp + src_pitch * height,
		*prevp_end = prevp + prv_pitch * height,
		*nextp_end = nextp + nxt_pitch * height;
	BYTE* destp_end = destp + dst_pitch * height;
#endif

#ifdef DO_SPATIAL
	const int sskip = src_pitch - row_size + 1,
		pskip = prv_pitch - row_size + 1,
		nskip = nxt_pitch - row_size + 1,
		dskip = dst_pitch - row_size + 1;
#else
	const int sskip = src_pitch - row_size,
		pskip = prv_pitch - row_size,
		nskip = nxt_pitch - row_size,
		dskip = dst_pitch - row_size;
#endif
	int ycnt = height;

	do
	{
		int xcnt = row_size;
#ifdef DO_SPATIAL
		xcnt -= 2;

		*destp = *currp; // Copy left edge

		++currp;
		++prevp;
		++nextp;
		++destp;
#endif

		do
		{
			assert(currp < currp_end);
			assert(prevp < prevp_end);
			assert(nextp < nextp_end);
			assert(destp < destp_end);

			int b = *currp, pbt = *prevp++, nbt = *nextp++;
			int pdiff = pbt - b, ndiff = nbt - b;
			if((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
			{
#ifdef DO_SPATIAL
				int pb1 = currp[-src_pitch - 1], pb2 = currp[-src_pitch],
					pb3 = currp[-src_pitch + 1], b1 = currp[-1], b2 = currp[1],
					nb1 = currp[src_pitch - 1], nb2 = currp[src_pitch],
					nb3 = currp[src_pitch + 1];
#endif
				int sum = b, cnt = 1;

				if(abs(pbt - b) <= temporal_threshold)
				{
					sum += pbt;
					++cnt;
				}
				if(abs(nbt - b) <= temporal_threshold)
				{
					sum += nbt;
					++cnt;
				}

#ifdef DO_SPATIAL
				if(abs(pb1 - b) <= spatial_threshold)
				{
					sum += pb1;
					++cnt;
				}
				if(abs(pb2 - b) <= spatial_threshold)
				{
					sum += pb2;
					++cnt;
				}
				if(abs(pb3 - b) <= spatial_threshold)
				{
					sum += pb3;
					++cnt;
				}
				if(abs(b1 - b) <= spatial_threshold)
				{
					sum += b1;
					++cnt;
				}
				if(abs(b2 - b) <= spatial_threshold)
				{
					sum += b2;
					++cnt;
				}
				if(abs(nb1 - b) <= spatial_threshold)
				{
					sum += nb1;
					++cnt;
				}
				if(abs(nb2 - b) <= spatial_threshold)
				{
					sum += nb2;
					++cnt;
				}
				if(abs(nb3 - b) <= spatial_threshold)
				{
					sum += nb3;
					++cnt;
				}

				assert(sum < 2806);
				assert(cnt < 12);
#else
				assert(sum < 766);
				assert(cnt < 4);
#endif
				assert(sum >= 0);
				assert(cnt > 0);

				*destp++ = (BYTE)(((sum * 2 + cnt) * scaletab[cnt]) >> 16);
				++currp;
			}
			else
			{
				*destp++ = *currp++;
			}
		} while(--xcnt);
		assert(xcnt == 0);

#ifdef DO_SPATIAL
		*destp = *currp; // Copy right edge
#endif

		currp += sskip;
		prevp += pskip;
		nextp += nskip;
		destp += dskip;

	} while(--ycnt);
	assert(ycnt == 0);

	assert(currp == currp_end);
	assert(prevp == prevp_end);
	assert(nextp == nextp_end);
	assert(destp == destp_end);
}




#ifdef INCLUDE_SSE2_VER
#include "YV12_SSE2.h"
#endif

#ifdef INCLUDE_SSSE3_VER
#include "YV12_SSSE3.h"
#endif

#ifdef INCLUDE_MMX_VER
#include "YV12_MMX.h"
#endif


#undef CLASS_NAME
