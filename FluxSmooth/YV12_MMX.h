// "YV12_MMX.h"
// YV12 version with the original MMXSSE optimizations.
// No changes made to assembly code except those needed to port it to 64-bit.



#define CHECK_AND_ADD(threshold) __asm { \
	__asm movq mm2, mm0 \
	__asm movq mm3, mm1 \
	__asm psubusw mm2, mm1 \
	__asm psubusw mm3, mm0 \
	__asm por mm2, mm3				/* mm2 = abs diff */ \
	__asm pcmpgtw mm2, threshold	/* Compare with threshold */ \
	__asm paddw mm6, mm2			/* -1 from counter if not within */ \
	__asm pandn mm2, mm1 \
	__asm paddw mm5, mm2			/* Add to sum */ \
}

void CLASS_NAME::DoFilter_MMX( const BYTE * currp, const long long src_pitch,
							   const BYTE * prevp, const BYTE * nextp,
							   const long long prv_pitch, const long long nxt_pitch,
							   BYTE * destp, const long long dst_pitch,
							   const int row_size, int height )
{
	const int xmax = row_size - 4;
	int ycnt = height;

	__int64 prev_pels = 0, next_pels = 0;
#ifdef DO_SPATIAL
	__declspec(align(16)) static const __int64 counter_init = 0x000b000b000b000b;
#else
	__declspec(align(16)) static const __int64 counter_init = 0x0003000300030003;
#endif
	__declspec(align(16)) static const __int64 indexer = 0x1000010000100001;

#ifdef DO_SPATIAL
	const __int64 spat_thresh =
					  ((__int64)spatial_threshold << 48) |
					  ((__int64)spatial_threshold << 32) |
					  ((__int64)spatial_threshold << 16) |
					   (__int64)spatial_threshold;
#endif
	const __int64 temp_thresh = 
					  ((__int64)temporal_threshold << 48) |
					  ((__int64)temporal_threshold << 32) |
					  ((__int64)temporal_threshold << 16) |
					   (__int64)temporal_threshold;
	const __int64 * scaletabp = scaletab_MMX;

	__asm
	{
		mov rsi, currp
		mov rdi, destp
		pxor mm7, mm7

yloop:
		// Copy first dword

		mov eax, dword ptr [rsi]
		mov dword ptr [rdi], eax

		mov rcx, 4

xloop:
		// Get current pels, init sum and counter

		movd mm0, [rsi + rcx]
		punpcklbw mm0, mm7
		movq mm5, mm0
		movq mm6, counter_init

#ifdef DO_SPATIAL
		// Middle left

		movq mm1, mm0
		psllq mm1, 16
		movd mm2, [rsi + rcx - 4]
		punpcklbw mm2, mm7
		psrlq mm2, 48
		por mm1, mm2

		CHECK_AND_ADD(spat_thresh)

		// Middle right

		movq mm1, mm0
		psrlq mm1, 16
		movd mm2, [rsi + rcx + 4]
		punpcklbw mm2, mm7
		psllq mm2, 48
		por mm1, mm2

		CHECK_AND_ADD(spat_thresh)

		// Top left

		mov rax, rsi
		sub rax, src_pitch
		movd mm1, [rax + rcx]
		punpcklbw mm1, mm7
		psllq mm1, 16
		movd mm2, [rax + rcx - 4]
		punpcklbw mm2, mm7
		psrlq mm2, 48
		por mm1, mm2

		CHECK_AND_ADD(spat_thresh)

		// Top centre

		movd mm1, [rax + rcx]
		punpcklbw mm1, mm7

		CHECK_AND_ADD(spat_thresh)

		// Top right

		psrlq mm1, 16
		movd mm2, [rax + rcx + 4]
		punpcklbw mm2, mm7
		psllq mm2, 48
		por mm1, mm2

		CHECK_AND_ADD(spat_thresh)

		// Bottom left

		mov rax, rsi
		add rax, src_pitch
		movd mm1, [rax + rcx]
		punpcklbw mm1, mm7
		psllq mm1, 16
		movd mm2, [rax + rcx - 4]
		punpcklbw mm2, mm7
		psrlq mm2, 48
		por mm1, mm2

		CHECK_AND_ADD(spat_thresh)

		// Bottom centre

		movd mm1, [rax + rcx]
		punpcklbw mm1, mm7

		CHECK_AND_ADD(spat_thresh)

		// Bottom right

		psrlq mm1, 16
		movd mm2, [rax + rcx + 4]
		punpcklbw mm2, mm7
		psllq mm2, 48
		por mm1, mm2

		CHECK_AND_ADD(spat_thresh)
#endif

		// Previous frame

		mov rax, prevp
		movd mm1, [rax + rcx]
		punpcklbw mm1, mm7
		movq prev_pels, mm1

		CHECK_AND_ADD(temp_thresh)

		// Next frame

		mov rax, nextp
		movd mm1, [rax + rcx]
		punpcklbw mm1, mm7
		movq next_pels, mm1

		CHECK_AND_ADD(temp_thresh)

		// Average

		psllw mm5, 1					// sum *= 2
		paddw mm5, mm6					// sum += count

		pmaddwd mm6, indexer			// Make index into lookup
		movq mm1, mm6
		punpckhdq mm6, mm6
		mov rax, scaletabp
		paddd mm1, mm6
		xor edx, edx
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

		add rcx, 4
		cmp ecx, xmax
		jl xloop

		// Copy last dword

		mov eax, dword ptr [rsi + rcx]
		mov dword ptr [rdi + rcx], eax

		// Next row

		add rsi, src_pitch
		mov rax, prevp
		add rax, prv_pitch
		mov prevp, rax
		mov rdx, nextp
		add rdx, nxt_pitch
		mov nextp, rdx
		add rdi, dst_pitch

		dec ycnt
		jnz yloop

		sfence
		emms
	}
}


