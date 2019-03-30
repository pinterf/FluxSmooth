
// FluxSmooth
// Avisynth filter for smoothing of fluctuations
//
// By Ross Thomas <ross@grinfinity.com>
//
// There is no copyright on this code, and there are no conditions
// on its distribution or use. Do with it what you will.

// YUY2 version

#include <malloc.h>

#ifdef DO_SPATIAL
#define FULL_CLASS_NAME FluxSmoothST_YUY2
#else
#define FULL_CLASS_NAME FluxSmoothT_YUY2
#endif

class FULL_CLASS_NAME : public GenericVideoFilter
{
public:
#ifdef DO_SPATIAL
	FULL_CLASS_NAME(PClip _child, int temporal_threshold,
		int spatial_threshold, IScriptEnvironment* env);
#else
	FULL_CLASS_NAME(PClip _child, int temporal_threshold, IScriptEnvironment* env);
#endif
	virtual ~FULL_CLASS_NAME();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

private:
	void DoFilter_C(const BYTE * currp, const BYTE * prevp, const BYTE * nextp,
					const long long prv_pitch, const long long src_pitch, const long long nxt_pitch, 
					BYTE * destp, const long long dst_pitch, const int row_size, const int height);
	void DoFilter_MMX(const BYTE * currp, const BYTE * prevp, const BYTE * nextp,
					  const long long prv_pitch, const long long src_pitch, const long long nxt_pitch, 
					  BYTE * destp, const long long dst_pitch, const int row_size, const int height);
	
private:
	int temporal_threshold;
#ifdef DO_SPATIAL
	int spatial_threshold;
#endif
	bool use_SSE;
	short scaletab[16];
	__int64* scaletab_MMX;
};



#ifdef DO_SPATIAL
FULL_CLASS_NAME::FULL_CLASS_NAME(PClip _child, int _temporal_threshold,
					   int _spatial_threshold, IScriptEnvironment* env)
	: GenericVideoFilter(_child), temporal_threshold(_temporal_threshold),
	  spatial_threshold(_spatial_threshold), scaletab_MMX(0)
#else
FULL_CLASS_NAME::FULL_CLASS_NAME(PClip _child, int _temporal_threshold, IScriptEnvironment* env)
	: GenericVideoFilter(_child), temporal_threshold(_temporal_threshold), scaletab_MMX(0)
#endif
{
	assert(_child);
#ifdef DO_SPATIAL
	assert(temporal_threshold >= -1);
	assert(spatial_threshold >= -1);
	assert(!((-1 == temporal_threshold) && (-1 == spatial_threshold)));
#else
	assert(temporal_threshold >= 0);
#endif
	assert(env);

	scaletab[1] = 32767;
	for(int i = 2; i < 16; ++i)
		scaletab[i] = (int)(32768.0 / i + 0.5);

	if (env->GetCPUFlags() & CPUF_INTEGER_SSE)
	{
		use_SSE = true;
		scaletab_MMX = (__int64*)_aligned_malloc(8*65536,64);	// sh0: We use aligned memory.
		for(long long i = 0; i < 65536; ++i)
		{
			scaletab_MMX[i] = ( (__int64)scaletab[ i        & 15]       ) |
							  (((__int64)scaletab[(i >>  4) & 15]) << 16) |
							  (((__int64)scaletab[(i >>  8) & 15]) << 32) |
							  (((__int64)scaletab[(i >> 12) & 15]) << 48);
		}
	} else
		use_SSE = false;
}

FULL_CLASS_NAME::~FULL_CLASS_NAME()
{
	// sh0: Use aligned malloc
	if(scaletab_MMX)
	    _aligned_free(scaletab_MMX);
}



PVideoFrame __stdcall FULL_CLASS_NAME::GetFrame(int n, IScriptEnvironment* env)
{
	assert(n >= 0 && n < vi.num_frames);
	assert(env);

	PVideoFrame currf = child->GetFrame(n, env);
	assert(currf);
	const BYTE* currp = currf->GetReadPtr();
	assert(currp);

	const int src_pitch = currf->GetPitch(), row_size = currf->GetRowSize(),
		height = currf->GetHeight();

	PVideoFrame destf = env->NewVideoFrame(vi);
	assert(destf);
	BYTE* destp = destf->GetWritePtr();
	assert(destp);

	const int dst_pitch = destf->GetPitch();

	if(n == 0 || n == vi.num_frames - 1)
	{
		env->BitBlt(destp, dst_pitch, currp, src_pitch, row_size, height);
		return destf;
	}

	PVideoFrame prevf = child->GetFrame(n - 1, env);
	assert(prevf);
	const BYTE* prevp = prevf->GetReadPtr();
	assert(prevp);
	const int prv_pitch = prevf->GetPitch();

	PVideoFrame nextf = child->GetFrame(n + 1, env);
	assert(nextf);
	const BYTE* nextp = nextf->GetReadPtr();
	assert(nextp);
	const int nxt_pitch = nextf->GetPitch();

	// Copy first and last rows

	memcpy(destp, currp, row_size);
	memcpy(destp + dst_pitch * (height - 1),
		currp + src_pitch * (height - 1), row_size);

	currp += src_pitch;
	prevp += prv_pitch;
	nextp += nxt_pitch;
	destp += dst_pitch;

	if(use_SSE)
	{
		DoFilter_MMX(currp, prevp, nextp, prv_pitch, src_pitch,
			nxt_pitch, destp, dst_pitch, row_size, height - 2);
	}
	else
	{
		DoFilter_C(currp, prevp, nextp, prv_pitch, src_pitch,
			nxt_pitch, destp, dst_pitch, row_size, height - 2);
	}

	return destf;
}




void FULL_CLASS_NAME::DoFilter_C(const BYTE * currp, const BYTE * prevp,
								 const BYTE * nextp, const long long prv_pitch,
								 const long long src_pitch, const long long nxt_pitch,
								 BYTE * destp, const long long dst_pitch,
								 const int row_size, const int height)
{
#ifdef _DEBUG
	const BYTE *currp_end = currp + src_pitch * height,
		*prevp_end = prevp + prv_pitch * height,
		*nextp_end = nextp + nxt_pitch * height;
	BYTE* destp_end = destp + dst_pitch * height;
#endif

	const long long pskip = prv_pitch - row_size + 4,
		sskip = src_pitch - row_size + 4,
		nskip = nxt_pitch - row_size + 4,
		dskip = dst_pitch - row_size + 4;
	int ycnt = height;

	enum PIXEL_TYPE { LUMA, CHROMA };

	do
	{
		*(DWORD*)destp = *(DWORD*)currp; // Copy left edge

		currp += 4;
		prevp += 4;
		nextp += 4;
		destp += 4;

		PIXEL_TYPE pt = LUMA;
		int xcnt = row_size - 8;

		do
		{
			assert(currp < currp_end);
			assert(prevp < prevp_end);
			assert(nextp < nextp_end);
			assert(destp < destp_end);

			int pbt = *prevp++, b = *currp, nbt = *nextp++;
			int pdiff = pbt - b, ndiff = nbt - b;
			if((pdiff < 0 && ndiff < 0) || (pdiff > 0 && ndiff > 0))
			{
#ifdef DO_SPATIAL
				int pb1, pb2 = currp[-src_pitch], pb3, b1, b2, nb1,
					nb2 = currp[src_pitch], nb3;
				if(LUMA == pt)
				{
					pb1 = currp[-src_pitch - 2];
					pb3 = currp[-src_pitch + 2];
					b1 = currp[-2];
					b2 = currp[2];
					nb1 = currp[src_pitch - 2];
					nb3 = currp[src_pitch + 2];
					pt = CHROMA;
				} else if(CHROMA == pt)
				{
					pb1 = currp[-src_pitch - 4];
					pb3 = currp[-src_pitch + 4];
					b1 = currp[-4];
					b2 = currp[4];
					nb1 = currp[src_pitch - 4];
					nb3 = currp[src_pitch + 4];
					pt = LUMA;
				}
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
			} else
				*destp++ = *currp++;
		} while(--xcnt);
		assert(xcnt == 0);

		*(DWORD*)destp = *(DWORD*)currp; // Copy right edge

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





// MMX helper macros

#define LEFT_FIX() __asm { \
	__asm movq mm2, mm1 \
	__asm pand mm1, luma_mask			/* Mask out all but Y2 and Y3 */ \
	__asm psrlq mm1, 16					/* Y2 and Y3 now in Y1 and Y2's place */ \
	__asm pand mm2, left_mask			/* Mask out Y1 and Y2 */ \
	__asm por mm1, mm2 \
	__asm punpcklbw mm1, mm7			/* mm1 = Y2 U1 Y3 V1 */ \
}

#define RIGHT_FIX() __asm { \
	__asm movq mm2, mm1 \
	__asm pand mm1, luma_mask			/* Mask out all but Y4 and Y5 */ \
	__asm psllq mm1, 16					/* Y4 and Y5 now in Y5 and Y6's place */ \
	__asm pand mm2, right_mask			/* Mask out Y5 and Y6 */ \
	__asm por mm1, mm2 \
	__asm punpckhbw mm1, mm7			/* mm1 = Y4 U3 Y5 V3 */ \
}

#define CHECK_AND_ADD(threshold) __asm { \
	__asm movq mm2, mm0 \
	__asm movq mm3, mm1 \
	__asm psubusw mm2, mm1 \
	__asm psubusw mm3, mm0 \
	__asm por mm2, mm3					/* mm2 = abs diff */ \
	__asm pcmpgtw mm2, threshold		/* Compare with threshold */ \
	__asm paddw mm6, mm2				/* -1 from counter if not within */ \
	__asm pandn mm2, mm1 \
	__asm paddw mm5, mm2				/* Add to sum */ \
}

void FULL_CLASS_NAME::DoFilter_MMX( const BYTE * currp, const BYTE * prevp,
									const BYTE * nextp, const long long prv_pitch,
									const long long src_pitch, const long long nxt_pitch,
									BYTE * destp, const long long dst_pitch,
									const int row_size, const int height)
{
#ifdef DO_SPATIAL
	__declspec(align(16)) static const __int64 counter_init = 0x000b000b000b000b,
#else
	__declspec(align(16)) static const __int64 counter_init = 0x0003000300030003,
#endif
		luma_mask = 0x000000ff00ff0000, left_mask = 0xffffffffff00ff00,
		right_mask = 0xff00ff00ffffffff, indexer = 0x1000010000100001;

	const __int64
#ifdef DO_SPATIAL
		spat_thresh = ((__int64)spatial_threshold << 48) |
					  ((__int64)spatial_threshold << 32) |
					  ((__int64)spatial_threshold << 16) |
					   (__int64)spatial_threshold,
#endif
		temp_thresh = ((__int64)temporal_threshold << 48) |
					  ((__int64)temporal_threshold << 32) |
					  ((__int64)temporal_threshold << 16) |
					   (__int64)temporal_threshold,
		*scaletabp = scaletab_MMX;

	
	const int xmax = row_size - 4;
	int ycnt = height;
	__int64 prev_pels = 0, next_pels = 0;

	__asm
	{
		// Register usage:
		//
		// ecx - x loop
		// esi - current frame
		// edi - destination frame
		// mm0 = curr pels
		// mm5 - sum
		// mm6 - counter
		// mm7 - zero

		mov rsi, currp
		mov rdi, destp
		pxor mm7, mm7

yloop:
		// Copy first two pels

		mov eax, dword ptr [rsi]
		mov dword ptr [rdi], eax

		mov ecx, 4

xloop:
		// Get current, init sum and counter

		movq mm0, [rsi + rcx - 4]
		movq mm1, mm0
		punpckhbw mm0, mm7				// mm0 = Y3 U2 Y4 V2
		movq mm5, mm0					// Current bytes to sum
		movq mm6, counter_init			// Init counter

#ifdef DO_SPATIAL
		// Middle left

		LEFT_FIX()
		CHECK_AND_ADD(spat_thresh)

		// Middle right

		movq mm1, [rsi + rcx]
		RIGHT_FIX()
		CHECK_AND_ADD(spat_thresh)

		// Top left

		mov rax, rsi
		sub rax, src_pitch
		movq mm1, [rax + rcx - 4]
		LEFT_FIX()
		CHECK_AND_ADD(spat_thresh)

		// Top centre

		movd mm1, [rax + rcx]
		punpcklbw mm1, mm7				// mm1 = Y3 U2 Y4 V2
		CHECK_AND_ADD(spat_thresh)

		// Top right

		movq mm1, [rax + rcx]
		RIGHT_FIX()
		CHECK_AND_ADD(spat_thresh)

		// Bottom left

		mov rax, rsi
		add rax, src_pitch
		movq mm1, [rax + rcx - 4]
		LEFT_FIX()
		CHECK_AND_ADD(spat_thresh)

		// Bottom centre

		movd mm1, [rax + rcx]
		punpcklbw mm1, mm7				// mm1 = Y3 U2 Y4 V2
		CHECK_AND_ADD(spat_thresh)

		// Bottom right

		movq mm1, [rax + rcx]
		RIGHT_FIX()
		CHECK_AND_ADD(spat_thresh)
#endif

		// Previous frame

		mov rax, prevp
		movd mm1, [rax + rcx]
		punpcklbw mm1, mm7				// mm1 = Y3 U2 Y4 V2
		movq prev_pels, mm1				// For later
		CHECK_AND_ADD(temp_thresh)

		// Next frame

		mov rax, nextp
		movd mm1, [rax + rcx]
		punpcklbw mm1, mm7				// mm1 = Y3 U2 Y4 V2
		movq next_pels, mm1				// For later
		CHECK_AND_ADD(temp_thresh)

		// Average

		psllw mm5, 1					// sum *= 2
		paddw mm5, mm6					// sum += count

		pmaddwd mm6, indexer			// Make index into lookup
		movq mm1, mm6
		punpckhdq mm6, mm6
		mov rax, scaletabp
		paddd mm1, mm6
		xor rdx, rdx
		movd edx, mm1

	    movq mm2, [rax + rdx * 8]		// Do lookup
	    pmulhw mm5, mm2					// mm5 = average

		// Apply smoothing only to fluctuating pels

		movq mm1, mm0
		movq mm2, prev_pels
		movq mm3, mm0
		movq mm4, next_pels

		pcmpgtw mm1, mm2				// curr > prev
		pcmpgtw mm3, mm4				// curr > next
		pcmpgtw mm2, mm0				// prev > curr
		pcmpgtw mm4, mm0				// next > curr

		pand mm1, mm3					// (curr > prev) and (curr > next)
		pand mm2, mm4					// (prev > curr) and (next > curr)
		por mm1, mm2					// mm1 = FFh if fluctuating, else 00h

		movq mm2, mm1
		pand mm1, mm5					// mm1 = smoothed pels
		pandn mm2, mm0					// mm2 = unsmoothed pels
		por mm1, mm2					// mm1 = result

		// Store

		packuswb mm1, mm7
	    // sh0: Streamed store is not faster here!
		movq [rdi + rcx], mm1

		// Advance

		add ecx, 4
		cmp ecx, xmax
		jl xloop

		// Copy last two pels

		mov eax, dword ptr [rsi + rcx]
		mov dword ptr [rdi + rcx], eax

		// Next row

		add rsi, src_pitch
		mov rax, prevp
		add rax, prv_pitch
		mov prevp, rax
		mov rax, nextp
		add rax, nxt_pitch
		mov nextp, rax
		add rdi, dst_pitch

		dec ycnt
		jnz yloop

		sfence
		emms
	}
}

#undef FULL_CLASS_NAME
