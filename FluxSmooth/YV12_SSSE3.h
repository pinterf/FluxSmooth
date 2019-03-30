// "YV12_SSSE3.h"
// Devin Gardner - 2010/11/30
//
// YV12 version with SSSE3 optimizations targeting Intel Core 2 65nm (Conroe/Kentsfield).


#undef _2BYTES
#undef _REPEAT_8_WORDS
#define _2BYTES( _word )		((BYTE)(_word)), ((BYTE)((_word) >> 8))
#define _REPEAT_8_WORDS( _w )	_2BYTES(_w), _2BYTES(_w), _2BYTES(_w), _2BYTES(_w), _2BYTES(_w), _2BYTES(_w), _2BYTES(_w), _2BYTES(_w)




#ifdef DO_SPATIAL

void CLASS_NAME::DoFilter_SSSE3( const BYTE * currp, const long long src_pitch,
								 const BYTE * prevp, const BYTE * nextp,
								 const long long prv_pitch, const long long nxt_pitch,
								 BYTE * destp, const long long dst_pitch,
								 const int row_size, int height )
{
#ifdef USE_SSSE3_FP_METHOD
	static const __m128  half = { 0.501f, 0.501f, 0.501f, 0.501f };	//for rounding to int
#endif
	static const __m128i eights = { 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8 };
	static const __m128i shufb_adjuster = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };	//used in making word indexes into byte indexes
	static const __m128i counter_init = { _REPEAT_8_WORDS(11) };	//0x000B000B000B000B000B000B000B000B
	const __m128i spat_thresh = this->spat_thresh;
	const __m128i temp_thresh = this->temp_thresh;
	const __m128i scaletab_low  = this->scaletab_low;
	const __m128i scaletab_high = this->scaletab_high;
	const __m128i l_edge_mask = { _2BYTES(0),  _2BYTES(-1), _2BYTES(-1), _2BYTES(-1),
								  _2BYTES(-1), _2BYTES(-1), _2BYTES(-1), _2BYTES(-1) };  //0xFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000
	__m128i r_edge_mask = { _REPEAT_8_WORDS(-1) };					//0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

	//The "right edge mask" is used to exclude the right-most pixel from being altered, as well as any pad
	// bytes included in the last QWORD of each row when the image width is not a multiple of 8. It will always
	// have at least the right-most 16-bits cleared, and will have all of them cleared when row_size % 8 == 1.
	for(long long i = -(8 - (((long long)row_size - 1) & 7)); i < 0; i++)
		((short*)&r_edge_mask)[8 + i] = 0;		//exclude this pixel

	__asm
	{
		//rsi = current source row
		//rdi = current destination row
		//r10 = top pixel row
		//r11 = bottom pixel row
		//r12 = next frame row
		//r13 = previous frame row
		//r9  = src_pitch (constant)
		//rdx = row_size (constant)
		//rcx = main loop counter
		//rax = temp
		//rbx is never used
		//All 16 xmm registers are used.
		//xmm15 is always all 0s, for unpacking bytes into zero-extended words.

		mov		rsi, currp
		mov		r9, src_pitch
		movsxd	rdx, row_size
		mov		r12, prevp
		mov		r13, nextp
		mov		rdi, destp
		mov		r10, rsi
		mov		r11, rsi
		sub		r10, r9					//r10 = top pixel row
		add		r11, r9					//r11 = bottom pixel row
		sub		rdx, 8					//subtract right-edge pixels; they're handled in the loop's unrolled final iteration

		jna  JUST_ONE_QWORD

		align(16)


Y_LOOP:

		//-----------------------------------
		// First iteration of loop is unrolled to handle special case of left frame edge.
		// Skip down to the comments for the code inside the main loop for a better idea of what's going on here.
		// The Middle Left, Top Left, and Bottom Left pixels can't be done for the first pixel, but they still
		//  must be done for the rest of the pixels in the first QWORD. The ML, TL, and BL pixels must be constructed
		//  from the Middle Center, Top Center, and Bottom Center pixels; 0 is inserted into the left-most pixel
		//  position for each of them.

		// Do Middle Left and Right, and prepare Top pixels
		movq	xmm0, [rsi]				//load first 8 pixels
		movq	xmm2, [rsi + 8]			//Middle Right
		pxor	xmm15, xmm15
		mov		ecx, 8					//loop counter (not used until main loop)

		movq	xmm6, [r10]				//Start loading Top pixels
		movq	xmm7, [r10 + 8]

		punpcklbw	xmm0, xmm15			//xmm0 = {p7, p6, p5, p4, p3, p2, p1, p0}
		movdqa		xmm14, spat_thresh
		movdqa		xmm1, xmm0
		punpcklbw	xmm2, xmm15			//xmm2 = {p15, p14, p13, p12, p11, p10, p9, p8}
		palignr		xmm1, xmm15, 14		//xmm1 = {p6, p5, p4, p3, p2, p1, p0, 0}
		movdqa		xmm9, xmm0			//xmm9 = sum of all included pixel values
		palignr		xmm2, xmm0, 2		//xmm2 = {p8, p7, p6, p5, p4, p3, p2, p1}
		movdqa		xmm10, counter_init	//xmm10 = counter of neighbor pixels included in the average

		movdqa     xmm3, xmm1
		 movdqa     xmm4, xmm2
		  punpcklbw  xmm6, xmm15		//xmm6 = {t7, t6, t5, t4, t3, t2, t1, t0}
		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  punpcklbw  xmm7, xmm15		//xmm7 = {t15, t14, t13, t12, t11, t10, t9, t8}
		pcmpgtw    xmm1, xmm14			//compare with spatial threshold
		 pcmpgtw    xmm2, xmm14
		  movdqa     xmm5, xmm6
		paddw      xmm10, xmm1			//subtract 1 from counter if not within threshold
		pandn      xmm1, xmm3
		  palignr    xmm5, xmm15, 14	//xmm5 = {t6, t5, t4, t3, t2, t1, t0, 0}
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm4
		  palignr    xmm7, xmm6, 2		//xmm7 = {t8, t7, t6, t5, t4, t3, t2, t1}
		paddw      xmm9, xmm1			//add included pixels to sum
		 paddw      xmm9, xmm2

		// Do Top pixels, and prepare Bottom pixels
		movq	xmm12, [r11]			//Start loading Bottom pixels
		movq	xmm13, [r11 + 8]
		movdqa	xmm1, xmm5
		movdqa	xmm2, xmm6
		movdqa	xmm3, xmm7

		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		  psubw      xmm3, xmm0
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  pabsw      xmm3, xmm3
		   punpcklbw  xmm12, xmm15		//xmm12 = {b7, b6, b5, b4, b3, b2, b1, b0}
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		paddw      xmm10, xmm1
		pandn      xmm1, xmm5
		   punpcklbw  xmm13, xmm15		//xmm13 = {b15, b14, b13, b12, b11, b10, b9, b8}
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm6
		   movdqa     xmm11, xmm12
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm7
		   palignr    xmm11, xmm15, 14	//xmm11 = {b6, b5, b4, b3, b2, b1, b0, 0}
		paddw      xmm9, xmm1
		   palignr    xmm13, xmm12, 2	//xmm13 = {b8, b7, b6, b5, b4, b3, b2, b1}
		 paddw      xmm9, xmm2
		  paddw      xmm9, xmm3

		// Do Bottom pixels, and prepare Prev/Next Frame pixels
		movq	xmm5, [r12] 			//Start loading Previous and Next Frame
		movq	xmm6, [r13]
		movdqa	xmm1, xmm11
		movdqa	xmm2, xmm12
		movdqa	xmm3, xmm13
		movdqa	xmm7, temp_thresh

		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		  psubw      xmm3, xmm0
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  pabsw      xmm3, xmm3
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		paddw      xmm10, xmm1
		pandn      xmm1, xmm11
		   punpcklbw  xmm5, xmm15
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm12
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm13
		   punpcklbw  xmm6, xmm15
		paddw      xmm9, xmm1
		 paddw      xmm9, xmm2
		  paddw      xmm9, xmm3

		// Do Previous and Next Frame pixels, and determine which pixels were fluctuating
		movdqa     xmm12, xmm5			//Previous
		movdqa     xmm1, xmm0
		movdqa     xmm13, xmm6			//Next
		movdqa     xmm2, xmm0

#ifndef USE_SSSE3_FP_METHOD
		movdqa	xmm8, eights
		movdqa	xmm11, shufb_adjuster
#else
		movdqa	xmm11, half
#endif

		psubw      xmm5, xmm0
		 psubw      xmm6, xmm0
		  pcmpgtw    xmm1, xmm12		//curr > prev
		pabsw      xmm5, xmm5
		 pabsw      xmm6, xmm6
		  pcmpgtw    xmm2, xmm13		//curr > next
		pcmpgtw    xmm5, xmm7			//compare with temporal threshold
		 pcmpgtw    xmm6, xmm7
		  pand       xmm1, xmm2			//(curr > prev) and (curr > next)
		paddw      xmm10, xmm5
		pandn      xmm5, xmm12
		  pcmpgtw    xmm12, xmm0		//prev > curr
		 paddw      xmm10, xmm6
		 pandn      xmm6, xmm13
		  pcmpgtw    xmm13, xmm0		//next > curr
		paddw      xmm9, xmm5
		 paddw      xmm9, xmm6
		  pand       xmm12, xmm13 		//(prev > curr) and (next > curr)
		  por        xmm12, xmm1		//xmm12 = mask set if pixel is fluctuating
		  pand       xmm12, l_edge_mask	//exclude the left-edge pixel from being smoothed

		// Average and store results, and pre-load next iteration's Middle pixels

#ifndef USE_SSSE3_FP_METHOD
		pcmpeqb		xmm7, xmm7			//xmm7 = all -1s
		movdqa		xmm3, xmm10
		mov		eax, 16					//generate an address ahead of time for loading next iteration's Middle Right pixels
		cmp		edx, 8					//test if row pitch consists of only 2 QWORDS (8 was already subbed from rdx)
		movdqa		xmm1, scaletab_low	//xmm1 = {7, 6, 5, 4, 3, 2, 1, 0}
		movdqa		xmm2, scaletab_high	//xmm2 = {-, -, -, -, 11, 10, 9, 8}
		cmovle	eax, ecx				//if so, load from Middle Center pixels instead; the main loop is gonna get skipped.
		paddb		xmm7, xmm8			//xmm7 = all 7s

		psllw       xmm3, 8
		paddw       xmm9, xmm9			//sum *= 2
		por         xmm3, xmm10			//low & high byte of each word both hold count; from here on xmm3 is bytes
		 movdqa      xmm6, xmm12
		movdqa      xmm4, xmm3
		movdqa      xmm5, xmm3
		 pandn       xmm6, xmm0 		//xmm6 = pixels that will remain unsmoothed

		movdqa	xmm13, xmm0 			//save for use as Middle Left pixels for next iteration
		movq	xmm0, [rsi + 8]			//pre-load main pixels for next iteration

		pcmpgtb     xmm3, xmm7			//LO: xmm3 = mask set for elements that are 8->11, clear for ones that are 1->7
		 psubb       xmm4, xmm8			//HI: subtract 8: elements are either 0->3 or negative numbers
		movq	xmm7, [rsi + rax]		//pre-load Middle Right pixels for next iteration
		pandn       xmm3, xmm5			//LO: xmm3 = mixture of valid index numbers 1->7 and 0s
		 paddb       xmm4, xmm4			//HI: mul by two; turn word indexes to byte indexes
		paddb       xmm3, xmm3			//LO: mul by two; turn word indexes to byte indexes
		 paddb       xmm4, xmm11		//HI: high bytes of each word now index high bytes of scaletab words (also, negative elements will remain negative)
		paddb       xmm3, xmm11			//LO: high bytes of each word now index high bytes of scaletab word data
		 pshufb      xmm2, xmm4			//HI: xmm2 = mixture of scaletab[8->11] and 0s as words
		paddw       xmm9, xmm10			//sum += count
		pshufb      xmm1, xmm3			//LO: xmm1 = scaletab[0->7] as words; scaletab[0] value is conveniently 0.

		  punpcklbw   xmm0, xmm15		//prep next iter's pixels
		por			xmm1, xmm2
		pmulhw		xmm1, xmm9			//xmm1 = averages
		  punpcklbw   xmm7, xmm15
		pand		xmm1, xmm12			//xmm1 = pixels that will be smoothed
		  movdqa      xmm3, xmm13
		  movdqa      xmm2, xmm7
		por			xmm1, xmm6			//xmm1 = result pixels

#else

		movdqa		xmm4, xmm10			//copy count for high data
		punpcklwd	xmm10, xmm15
		mov		eax, 16					//generate an address ahead of time for loading next iteration's Middle Right pixels
		cmp		edx, 8					//test if row pitch is only 2 QWORDS (8 was already subbed from rdx)
		cvtdq2ps	xmm3, xmm10			//xmm3 = low data for count
		cmovle	eax, ecx				//if so, load from Middle Center pixels instead; the main loop is gonna get skipped.
		punpckhwd	xmm4, xmm15
		cvtdq2ps	xmm4, xmm4			//xmm4 = high data for count

		rcpps		xmm1, xmm3
		 movdqa      xmm10, xmm9		//copy sum for high data
		rcpps		xmm2, xmm4
		movdqa		xmm6, xmm1
		 punpcklwd   xmm9, xmm15
		movdqa		xmm7, xmm2
		 cvtdq2ps    xmm9, xmm9 		//xmm9 = low data for sum

		mulps		xmm6, xmm6
		addps		xmm1, xmm1
		 punpckhwd   xmm10, xmm15
		mulps		xmm7, xmm7
		addps		xmm2, xmm2
		movq	xmm8, [rsi + rcx]		//pre-load main pixels for next iteration
		mulps		xmm3, xmm6
		subps		xmm1, xmm3
		 cvtdq2ps    xmm10, xmm10		//xmm10 = high data for sum
		mulps		xmm4, xmm7
		subps		xmm2, xmm4
		movq	xmm4, [rsi + rax]		//pre-load Middle Right pixels for next iteration

		mulps		xmm1, xmm9			//multiply sum by reciprocal of count to get averages
		 punpcklbw   xmm8, xmm15		//prep next iter's pixels
		mulps		xmm2, xmm10
		addps		xmm1, xmm11			//add 0.5 to round
		cvttps2dq	xmm1, xmm1			//convert to int with rounding (round-to-even)
		addps		xmm2, xmm11
		 movdqa      xmm3, xmm12
		cvttps2dq	xmm2, xmm2
		 punpcklbw   xmm4, xmm15
		 pandn       xmm3, xmm0			//xmm3 = pixels that will remain unsmoothed
		packssdw	xmm1, xmm2			//covert back to 8 words - the smoothed pixels

		pand		xmm1, xmm12			//xmm1 = pixels that will be smoothed
		movdqa		xmm2, xmm4
		por			xmm1, xmm3			//xmm1 = result pixels
		movdqa		xmm3, xmm0 			//save for use as Middle Left pixels for next iteration
		movdqa		xmm0, xmm8
#endif

		packuswb	xmm1, xmm15			//convert back to bytes in low QWORD
		movq	[rdi], xmm1				//store processed pixels

		//xmm0 = next iter's pixels, unpacked to words
		//xmm3 = next iter's Middle Left pixels, unpacked
		//xmm2 = next iter's Middle Right pixels, unpacked


		//Skip main loop if row_size <= 16; row is just a left-edge QWORD and a right-edge QWORD.
		//If not, branch prediction will eliminate this conditional jump.
		cmp		edx, 8				//8 was already subtracted from rdx at beginning of function
		jle  X_POST_ITER				

		align(16)





X_LOOP:

		//-----------------------------------
		// Main Loop - filter 8 pixels from a QWORD - loop never handles the first or last QWORD in a row.
		// xmm0 is prepared with current 8 pixels zero-extended to words
		// xmm3 & xmm2 are prepared with the Middle Left and Middle Right pixels zero-extended to words
		// xmm0 = {p7, p6, p5, p4, p3, p2, p1, p0}
		// xmm3 = {p-1, p-2, p-3, p-4, p-5, p-6, p-7, p-8}			
		// xmm2 = {p15, p14, p13, p12, p11, p10, p9, p8}

		// Do Middle Left and Middle Right, and prepare Top pixels

		movq	xmm8, [r10 + rcx - 8]	//Start loading Top pixels
		movq	xmm6, [r10 + rcx]
		movq	xmm7, [r10 + rcx + 8]

		movdqa		xmm1, xmm0
//		movdqa		xmm14, xmm14		//Core 2 likes to have all input operands already in the execution pipeline, so just refresh this one before it's used again.
		palignr		xmm1, xmm3, 14		//xmm1 = {p6, p5, p4, p3, p2, p1, p0, p-1}
		movdqa		xmm9, xmm0			//xmm9 = sum of all included pixel values
		palignr		xmm2, xmm0, 2		//xmm2 = {p8, p7, p6, p5, p4, p3, p2, p1}
		movdqa		xmm10, counter_init	//xmm10 = counter of neighbor pixels included in the average

		//check pixels and add to sum and count
		movdqa     xmm3, xmm1
		 movdqa     xmm4, xmm2
		  punpcklbw  xmm6, xmm15		//xmm6 = {t7, t6, t5, t4, t3, t2, t1, t0}
		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  punpcklbw  xmm8, xmm15		//xmm8 = {t-1, t-2, t-3, t-4, t-5, t-6, t-7, t-8}
		pcmpgtw    xmm1, xmm14			//compare absolute difference with spatial threshold
		 pcmpgtw    xmm2, xmm14
		  movdqa     xmm5, xmm6
		paddw      xmm10, xmm1			//subtract 1 from counter if not within threshold
		pandn      xmm1, xmm3
		  punpcklbw  xmm7, xmm15		//xmm7 = {t15, t14, t13, t12, t11, t10, t9, t8}
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm4
		  palignr    xmm5, xmm8, 14		//xmm5 = {t6, t5, t4, t3, t2, t1, t0, t-1}
		paddw      xmm9, xmm1			//add included pixel values to sum
		  palignr    xmm7, xmm6, 2		//xmm7 = {t8, t7, t6, t5, t4, t3, t2, t1}
		 paddw      xmm9, xmm2

		// Do Top Pixels, and prepare Bottom pixels

		movq	xmm8,  [r11 + rcx - 8]	//Start loading Bottom pixels
		movq	xmm12, [r11 + rcx]
		movq	xmm13, [r11 + rcx + 8]
		//the instructions to unpack and prepare these pixels is simply placed where the decoder can reach them
		// reasonably early and queue them up for whenever the inputs are ready and execution resources are available.

		movdqa	xmm1, xmm5
		movdqa	xmm2, xmm6
		movdqa	xmm3, xmm7

		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		  psubw      xmm3, xmm0
		   punpcklbw  xmm8, xmm15		//xmm8  = {b-1, b-2, b-3, b-4, b-5, b-6, b-7, b-8}
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  pabsw      xmm3, xmm3
		   punpcklbw  xmm12, xmm15		//xmm12 = {b7, b6, b5, b4, b3, b2, b1, b0}
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		paddw      xmm10, xmm1			//sub from counter
		pandn      xmm1, xmm5
		   punpcklbw  xmm13, xmm15		//xmm13 = {b15, b14, b13, b12, b11, b10, b9, b8}
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm6
		   movdqa     xmm11, xmm12
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm7
		   palignr    xmm11, xmm8, 14	//xmm11 = {b6, b5, b4, b3, b2, b1, b0, b-1}
		paddw      xmm9, xmm1			//add to sum
		   palignr    xmm13, xmm12, 2	//xmm13 = {b8, b7, b6, b5, b4, b3, b2, b1}
		 paddw      xmm9, xmm2
		  paddw      xmm9, xmm3

		// Do Bottom Pixels, and prepare Prev/Next Frame pixels

		movq	xmm5, [r12 + rcx]		//Start loading Previous Frame and Next Frame pixels
		movq	xmm6, [r13 + rcx]

		movdqa	xmm1, xmm11
		movdqa	xmm2, xmm12
		movdqa	xmm3, xmm13
		movdqa	xmm7, temp_thresh

		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		  psubw      xmm3, xmm0
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  pabsw      xmm3, xmm3
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		paddw      xmm10, xmm1
		pandn      xmm1, xmm11
		   punpcklbw  xmm5, xmm15
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm12
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm13
		   punpcklbw  xmm6, xmm15
		paddw      xmm9, xmm1
		 paddw      xmm9, xmm2
		  paddw      xmm9, xmm3

		// Do Previous and Next Frame Pixels
		// Also determine which pixels were fluctuating

		movdqa     xmm12, xmm5			//Previous
		movdqa     xmm1, xmm0
		movdqa     xmm13, xmm6			//Next
		movdqa     xmm2, xmm0

#ifndef USE_SSSE3_FP_METHOD
		movdqa	xmm8, eights			//xmm8 = all 8s
		movdqa	xmm11, shufb_adjuster	//adds +1 to byte index in high byte of each word, to prepare for 'pshufb'
#else
		movdqa	xmm11, half
#endif

		psubw      xmm5, xmm0
		 psubw      xmm6, xmm0
		  pcmpgtw    xmm1, xmm12		//curr > prev
		pabsw      xmm5, xmm5
		 pabsw      xmm6, xmm6
		  pcmpgtw    xmm2, xmm13		//curr > next
		pcmpgtw    xmm5, xmm7			//compare with temporal threshold
		 pcmpgtw    xmm6, xmm7
		  pand       xmm1, xmm2			//(curr > prev) and (curr > next)
		paddw      xmm10, xmm5
		pandn      xmm5, xmm12
		  pcmpgtw    xmm12, xmm0		//prev > curr
		 paddw      xmm10, xmm6
		 pandn      xmm6, xmm13
		  pcmpgtw    xmm13, xmm0		//next > curr
		paddw      xmm9, xmm5
		 paddw      xmm9, xmm6
		  pand       xmm12, xmm13 		//(prev > curr) and (next > curr)
		  por        xmm12, xmm1		//xmm12 = mask set if pixel is fluctuating

		// Average and store results, and pre-load next iteration's Middle pixels
		// Elements of xmm10 can only be 1->11 at this point.

#ifndef USE_SSSE3_FP_METHOD
		//Build up a control value for pshufb from count to choose the correct multipliers
		// from "scaletab" for each sum value.

		pcmpeqb		xmm7, xmm7			//xmm7 = all -1s
		movdqa		xmm3, xmm10
		lea		eax, [rcx + 16]		//generate an address ahead of time for loading next iteration's Middle Right pixels
		add		ecx, 8				//inc loop counter early
		cmp		ecx, edx			//do loop test way early
		movdqa		xmm1, scaletab_low	//xmm1 = {7, 6, 5, 4, 3, 2, 1, 0}
		cmovge	eax, ecx			//if this is last iteration, load from Middle Center pixels instead - this also avoids potentially loading from mem we don't own.
		movdqa		xmm2, scaletab_high	//xmm2 = {-, -, -, -, 11, 10, 9, 8}
		paddb		xmm7, xmm8			//xmm7 = all 7s

		psllw       xmm3, 8
		 movdqa      xmm6, xmm12
		por         xmm3, xmm10			//low & high byte of each word both hold count; from here on xmm3 is bytes
		paddw       xmm9, xmm9			//sum *= 2
		 pandn       xmm6, xmm0			//xmm6 = pixels that will remain unsmoothed
		movdqa      xmm4, xmm3
		movdqa      xmm5, xmm3

		movdqa	xmm13, xmm0 			//save for use as Middle Left pixels for next iteration
		movq	xmm0, [rsi + rcx]		//pre-load main pixels for next iteration

		pcmpgtb     xmm3, xmm7			//LO: xmm3 = mask set for elements that are 8->11, clear for ones that are 1->7
		 psubb       xmm4, xmm8			//HI: subtract 8: elements are either 0->3 or negative numbers
		movq	xmm7, [rsi + rax]		//pre-load Middle Right pixels for next iteration
		pandn       xmm3, xmm5			//LO: xmm3 = mixture of valid index numbers 1->7 and 0s
		 paddb       xmm4, xmm4			//HI: mul by two; turn word indexes to byte indexes
		paddb       xmm3, xmm3			//LO: mul by two; turn word indexes to byte indexes
		 paddb       xmm4, xmm11		//HI: high bytes of each word now index high bytes of scaletab words (also, negative elements will remain negative)
		paddb       xmm3, xmm11			//LO: high bytes of each word now index high bytes of scaletab word data
		 pshufb      xmm2, xmm4			//HI: xmm2 = mixture of scaletab[8->11] and 0s as words
		  paddw       xmm9, xmm10		//sum += count
		pshufb      xmm1, xmm3			//LO: xmm1 = scaletab[0->7] as words; scaletab[0] value is conveniently 0.

		  punpcklbw   xmm0, xmm15		//prep next iter's pixels
		por			xmm1, xmm2
		pmulhw		xmm1, xmm9			//xmm1 = averages
		  punpcklbw   xmm7, xmm15
		pand		xmm1, xmm12			//xmm1 = pixels that will be smoothed
		  movdqa      xmm3, xmm13
		  movdqa      xmm2, xmm7
		por			xmm1, xmm6			//xmm1 = result pixels

#else
		//xmm1/xmm2 = low & high data for reciprocal of count
		//xmm3/xmm4 = low & high for count
		//xmm6/xmm7 = temps for N-R calc; later, pre-loaded left/right pixels for next loop iter
		//xmm9/xmm10 will be low & high data for sum

		movdqa		xmm4, xmm10			//copy count for high data
		punpcklwd	xmm10, xmm15
		lea		eax, [rcx + 16]			//generate an address ahead of time for loading next iteration's Middle Right pixels
		add		ecx, 8					//inc loop counter early
		cvtdq2ps	xmm3, xmm10			//xmm3 = low data for count
		punpckhwd	xmm4, xmm15
		cmp		ecx, edx				//do loop test way early
		cmovge	eax, ecx				//if this is last iteration, load from Middle Center pixels instead - this also avoids potentially loading from mem we don't own.
		cvtdq2ps	xmm4, xmm4			//xmm4 = high data for count

		//Get reciprocals and do a Newton-Raphson refinement of them to make them more accurate
		//  result = 2r - (d * r^2)		(d = divisor; r = fast-reciprocal of d)
		//xmm3 = low d;			xmm4 = high d
		//xmm1/xmm6 = low r;	xmm2/xmm7 = high r
		rcpps		xmm1, xmm3
		 movdqa      xmm10, xmm9		//copy sum for high data
		rcpps		xmm2, xmm4
		movdqa		xmm6, xmm1
		 punpcklwd   xmm9, xmm15
		movdqa		xmm7, xmm2
		 cvtdq2ps    xmm9, xmm9 		//xmm9 = low data for sum

		mulps		xmm6, xmm6
		addps		xmm1, xmm1
		 punpckhwd   xmm10, xmm15
		mulps		xmm7, xmm7
		addps		xmm2, xmm2
		movq	xmm8, [rsi + rcx]		//pre-load main pixels for next iteration
		mulps		xmm3, xmm6
		subps		xmm1, xmm3
		 cvtdq2ps    xmm10, xmm10		//xmm10 = high data for sum
		mulps		xmm4, xmm7
		subps		xmm2, xmm4
		movq	xmm4, [rsi + rax]		//pre-load Middle Right pixels for next iteration

		mulps		xmm1, xmm9			//multiply sum by reciprocal of count to get averages
		 punpcklbw   xmm8, xmm15		//prep next iter's pixels
		mulps		xmm2, xmm10
		addps		xmm1, xmm11			//add 0.5 to round
		cvttps2dq	xmm1, xmm1			//convert to int with rounding (round-to-even)
		addps		xmm2, xmm11
		 movdqa      xmm3, xmm12
		cvttps2dq	xmm2, xmm2
		 punpcklbw   xmm4, xmm15
		 pandn       xmm3, xmm0			//xmm3 = pixels that will remain unsmoothed
		packssdw	xmm1, xmm2			//covert back to 8 words - the smoothed pixels

		pand		xmm1, xmm12			//xmm1 = pixels that will be smoothed
		movdqa		xmm2, xmm4
		por			xmm1, xmm3			//xmm1 = result pixels
		movdqa		xmm3, xmm0 			//save for use as Middle Left pixels for next iteration
		movdqa		xmm0, xmm8
#endif

		packuswb	xmm1, xmm15			//convert pixels back to bytes in low QWORD
		movq	[rdi + rcx - 8], xmm1	// sh0: Streamed store is not faster here!

		//xmm0 = next iter's pixels, unpacked to words
		//xmm3 = next iter's Middle Left pixels, unpacked
		//xmm2 = next iter's Middle Right pixels, unpacked

		jl  X_LOOP





X_POST_ITER:

		//-----------------------------------
		// Last iteration of loop is also unrolled, for the right-edge pixels.
		// xmm0 is prepared with current 8 pixels zero-extended to words
		// xmm3 is prepared with Middle Left pixels and xmm2 has the Middle Center pixels
		// xmm0 = {p7, p6, p5, p4, p3, p2, p1, p0}
		// xmm3 = {p-1, p-2, p-3, p-4, p-5, p-6, p-7, p-8}
		// xmm2 = {p7, p6, p5, p4, p3, p2, p1, p0}

		// Do Middle Left and Middle Right, and prepare Top pixels
		movq	xmm8, [r10 + rcx - 8]	//Start loading Top pixels
		movq	xmm6, [r10 + rcx]

		movdqa		xmm1, xmm0
		psrldq		xmm2, 2				//xmm2 = {0, p7, p6, p5, p4, p3, p2, p1}
//		movdqa		xmm14, xmm14		//Core 2 likes to have all input operands already in the execution pipeline, so just refresh this one before it's used again.
		palignr		xmm1, xmm3, 14		//xmm1 = {p6, p5, p4, p3, p2, p1, p0, p-1}
		movdqa		xmm9, xmm0			//xmm9 = sum
		movdqa		xmm10, counter_init	//xmm10 = counter
		pxor		xmm7, xmm7

		movdqa     xmm3, xmm1
		 movdqa     xmm4, xmm2
		  punpcklbw  xmm8, xmm15		//xmm8 = {t-1, t-2, t-3, t-4, t-5, t-6, t-7, t-8}
		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  punpcklbw  xmm6, xmm15		//xmm6 = {t7, t6, t5, t4, t3, t2, t1, t0}
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		paddw      xmm10, xmm1
		pandn      xmm1, xmm3
		  movdqa     xmm5, xmm6
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm4
		  palignr    xmm5, xmm8, 14		//xmm5 = {t6, t5, t4, t3, t2, t1, t0, t-1}
		paddw      xmm9, xmm1
		  palignr    xmm7, xmm6, 2		//xmm7 = {0, t7, t6, t5, t4, t3, t2, t1}
		 paddw      xmm9, xmm2

		// Do Top pixels, and prepare Bottom pixels
		movq	xmm8,  [r11 + rcx - 8]	//Start loading Bottom pixels
		movq	xmm12, [r11 + rcx]
		movdqa	xmm1, xmm5
		movdqa	xmm2, xmm6
		movdqa	xmm3, xmm7

		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		  psubw      xmm3, xmm0
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		   pxor       xmm13, xmm13
		  pabsw      xmm3, xmm3
		   punpcklbw  xmm8, xmm15		//xmm8 = {b-1, b-2, b-3, b-4, b-5, b-6, b-7, b-8}
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		paddw      xmm10, xmm1
		pandn      xmm1, xmm5
		   punpcklbw  xmm12, xmm15		//xmm12 = {b7, b6, b5, b4, b3, b2, b1, b0}
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm6
		   movdqa     xmm11, xmm12
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm7
		   palignr    xmm11, xmm8, 14	//xmm11 = {b6, b5, b4, b3, b2, b1, b0, b-1}
		paddw      xmm9, xmm1
		   palignr    xmm13, xmm12, 2	//xmm13 = {0, b7, b6, b5, b4, b3, b2, b1}
		 paddw      xmm9, xmm2
		  paddw      xmm9, xmm3

		// Do Bottom pixels, and prepare Prev/Next Frame pixels
		movq	xmm5, [r12 + rcx] 		//Start loading Previous and Next Frame
		movq	xmm6, [r13 + rcx]
		movdqa	xmm1, xmm11
		movdqa	xmm2, xmm12
		movdqa	xmm3, xmm13
		movdqa	xmm7, temp_thresh

		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		  psubw      xmm3, xmm0
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  pabsw      xmm3, xmm3
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		paddw      xmm10, xmm1
		pandn      xmm1, xmm11
		   punpcklbw  xmm5, xmm15
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm12
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm13
		   punpcklbw  xmm6, xmm15
		paddw      xmm9, xmm1
		 paddw      xmm9, xmm2
		  paddw      xmm9, xmm3

		/// Do Previous and Next Frame pixels, and determine which pixels were fluctuating
		movdqa     xmm12, xmm5			//Previous
		movdqa     xmm1, xmm0
		movdqa     xmm13, xmm6			//Next
		movdqa     xmm2, xmm0

#ifndef USE_SSSE3_FP_METHOD
		movdqa	xmm8, eights
		movdqa	xmm11, shufb_adjuster
#else
		movdqa	xmm11, half
#endif

		psubw      xmm5, xmm0
		 psubw      xmm6, xmm0
		  pcmpgtw    xmm1, xmm12		//curr > prev
		pabsw      xmm5, xmm5
		 pabsw      xmm6, xmm6
		  pcmpgtw    xmm2, xmm13		//curr > next
		pcmpgtw    xmm5, xmm7			//compare with temporal threshold
		 pcmpgtw    xmm6, xmm7
		  pand       xmm1, xmm2			//(curr > prev) and (curr > next)
		paddw      xmm10, xmm5
		pandn      xmm5, xmm12
		  pcmpgtw    xmm12, xmm0		//prev > curr
		 paddw      xmm10, xmm6
		 pandn      xmm6, xmm13
		  pcmpgtw    xmm13, xmm0		//next > curr
		paddw      xmm9, xmm5
		 paddw      xmm9, xmm6
		  pand       xmm12, xmm13 		//(prev > curr) and (next > curr)
		  por        xmm12, xmm1		//xmm12 = mask set if pixel is fluctuating
		  pand       xmm12, r_edge_mask	//exclude the right-edge pixel and any pad bytes from being smoothed

		// Average and store results, and pre-load next iteration's Middle pixels

#ifndef USE_SSSE3_FP_METHOD
		pcmpeqb		xmm7, xmm7			//xmm7 = all -1s
		movdqa		xmm3, xmm10
		movdqa		xmm1, scaletab_low	//xmm1 = {7, 6, 5, 4, 3, 2, 1, 0}
		movdqa		xmm2, scaletab_high	//xmm2 = {-, -, -, -, 11, 10, 9, 8}
		paddb		xmm7, xmm8			//xmm7 = all 7s

		psllw       xmm3, 8
		  paddw       xmm9, xmm9		//sum *= 2
		por         xmm3, xmm10			//low & high byte of each word both hold count; from here on xmm3 is bytes
		 movdqa      xmm6, xmm12
		movdqa      xmm4, xmm3
		movdqa      xmm5, xmm3
		 pandn       xmm6, xmm0			//xmm6 = pixels that will remain unsmoothed

		pcmpgtb     xmm3, xmm7			//LO: xmm3 = mask set for elements that are 8->11, clear for ones that are 1->7
		 psubb       xmm4, xmm8			//HI: subtract 8: elements are either 0->3 or negative numbers
		pandn       xmm3, xmm5			//LO: xmm3 = mixture of valid index numbers 1->7 and 0s
		 paddb       xmm4, xmm4			//HI: mul by two; turn word indexes to byte indexes
		paddb       xmm3, xmm3			//LO: mul by two; turn word indexes to byte indexes
		 paddb       xmm4, xmm11		//HI: high bytes of each word now index high bytes of scaletab words (also, negative elements will remain negative)
		paddb       xmm3, xmm11			//LO: high bytes of each word now index high bytes of scaletab word data
		 pshufb      xmm2, xmm4			//HI: xmm2 = mixture of scaletab[8->11] and 0s as words
		  paddw       xmm9, xmm10		//sum += count
		pshufb      xmm1, xmm3			//LO: xmm1 = scaletab[0->7] as words; scaletab[0] value is conveniently 0.

		mov			eax, height			//pre-load y-counter

		por			xmm1, xmm2
		pmulhw		xmm1, xmm9			//xmm1 = averages
		pand		xmm1, xmm12			//xmm1 = pixels that will be smoothed
		por			xmm1, xmm6			//xmm1 = result pixels

#else

		movdqa		xmm4, xmm10			//copy count for high data
		punpcklwd	xmm10, xmm15
		cvtdq2ps	xmm3, xmm10			//xmm3 = low data for count
		punpckhwd	xmm4, xmm15
		cvtdq2ps	xmm4, xmm4			//xmm4 = high data for count

		rcpps		xmm1, xmm3
		 movdqa      xmm10, xmm9		//copy sum for high data
		rcpps		xmm2, xmm4
		movdqa		xmm6, xmm1
		 punpcklwd   xmm9, xmm15
		movdqa		xmm7, xmm2
		 cvtdq2ps    xmm9, xmm9 		//xmm9 = low data for sum

		mulps		xmm6, xmm6
		addps		xmm1, xmm1
		 punpckhwd   xmm10, xmm15
		mulps		xmm7, xmm7
		addps		xmm2, xmm2
		mulps		xmm3, xmm6
		subps		xmm1, xmm3
		 cvtdq2ps    xmm10, xmm10		//xmm10 = high data for sum
		mulps		xmm4, xmm7
		subps		xmm2, xmm4

		mov			eax, height			//pre-load y-counter

		mulps		xmm1, xmm9			//multiply sum by reciprocal of count to get averages
		 movdqa      xmm3, xmm12
		mulps		xmm2, xmm10
		addps		xmm1, xmm11
		cvttps2dq	xmm1, xmm1			//convert to int with rounding (round-to-even)
		addps		xmm2, xmm11
		cvttps2dq	xmm2, xmm2
		 pandn       xmm3, xmm0			//xmm3 = pixels that will remain unsmoothed
		packssdw	xmm1, xmm2			//covert back to 8 words - the smoothed pixels
		pand		xmm1, xmm12			//xmm1 = pixels that will be smoothed
		por			xmm1, xmm3			//xmm1 = result pixels
#endif

		//eax is pre-loaded with "height"

		packuswb	xmm1, xmm15			//convert back to bytes in low QWORD
		movq	[rdi + rcx], xmm1


		// Advance to next row

		add		rsi, r9
		add		r12, prv_pitch
		add		r10, r9
		add		r13, nxt_pitch
		add		r11, r9
		add		rdi, dst_pitch
		sub		eax, 1
		mov		height, eax

		jnz  Y_LOOP

		nop
		sfence
		jmp  FINISHED







JUST_ONE_QWORD:

		//////////////////////////////////////////
		//Because this code path is for an extremely rare and unimportant case, its
		// speed is not too important. Concessions can be made to make the code more
		// compact. No matter what, it's still going to be way faster than the C code.

		movdqa		xmm3, l_edge_mask
		movdqa		xmm4, r_edge_mask
		movq		xmm0, [rsi]			//load first 8 pixels
		pxor		xmm15, xmm15
		movq		xmm6, [r10]			//load Top pixels
		pand		xmm3, xmm4
		punpcklbw	xmm0, xmm15
		movdqa		r_edge_mask, xmm3	//now excludes both left and right edge
		movdqa		xmm1, xmm0
		movdqa		xmm2, xmm0
		movdqa		xmm14, spat_thresh
		mov			r8, dst_pitch

		align(16)


QWORD_LOOP:
		//xmm0 = {p7, p6, p5, p4, p3, p2, p1, p0}
		//xmm1 = xmm0
		//xmm2 = xmm0
		//xmm6 = Top pixels pre-loaded in low QWORD

		palignr    xmm1, xmm15, 14		//xmm1 = {p6, p5, p4, p3, p2, p1, p0, 0}
		movdqa     xmm9, xmm0			//xmm9 = sum
		psrldq     xmm2, 2				//xmm2 = {0, p7, p6, p5, p4, p3, p2, p1}
		movdqa     xmm10, counter_init	//xmm10 = counter
		punpcklbw  xmm6, xmm15			//xmm6 = {t7, t6, t5, t4, t3, t2, t1, t0}

		movdqa     xmm3, xmm1
		 movdqa     xmm4, xmm2
		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		  movq       xmm12, [r11]		//load Bottom pixels
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  movdqa     xmm5, xmm6
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pslldq     xmm5, 2			//xmm5 = {t6, t5, t4, t3, t2, t1, t0, 0}		  
		paddw      xmm10, xmm1
		pandn      xmm1, xmm3
		  movdqa     xmm7, xmm6
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm4
		  psrldq     xmm7, 2			//xmm7 = {0, t7, t6, t5, t4, t3, t2, t1}
		paddw      xmm9, xmm1
		 paddw      xmm9, xmm2
		  punpcklbw  xmm12, xmm15		//xmm12 = {b7, b6, b5, b4, b3, b2, b1, b0}

		movdqa	xmm1, xmm5
		movdqa	xmm2, xmm6
		movdqa	xmm3, xmm7

		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		  psubw      xmm3, xmm0
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  pabsw      xmm3, xmm3
		   movdqa     xmm11, xmm12
		   movdqa     xmm13, xmm12
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		paddw      xmm10, xmm1
		pandn      xmm1, xmm5
		   pslldq     xmm11, 2			//xmm11 = {b6, b5, b4, b3, b2, b1, b0, 0}
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm6
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm7
		   psrldq     xmm13, 2			//xmm13 = {0, b7, b6, b5, b4, b3, b2, b1}
		paddw      xmm9, xmm1
		 paddw      xmm9, xmm2
		  paddw      xmm9, xmm3

		movq    xmm6, [r13]				//Next Frame
		movq    xmm5, [r12] 			//Previous frame
		movdqa	xmm1, xmm11
		movdqa	xmm2, xmm12
		movdqa	xmm3, xmm13

		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		  psubw      xmm3, xmm0
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  pabsw      xmm3, xmm3
		   movdqa  xmm7, temp_thresh
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		paddw      xmm10, xmm1
		pandn      xmm1, xmm11
		   punpcklbw  xmm5, xmm15
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm12
		add        r13, nxt_pitch
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm13
		   punpcklbw  xmm6, xmm15
		paddw      xmm9, xmm1
		add        r12, prv_pitch
		 paddw      xmm9, xmm2
		  paddw      xmm9, xmm3

		movdqa     xmm12, xmm5			//Previous
		movdqa     xmm1, xmm0
		movdqa     xmm13, xmm6			//Next
		movdqa     xmm2, xmm0
		movdqa	xmm8, eights
		movdqa	xmm11, shufb_adjuster

		psubw      xmm5, xmm0
		 psubw      xmm6, xmm0
		  pcmpgtw    xmm1, xmm12		//curr > prev
		pabsw      xmm5, xmm5
		 pabsw      xmm6, xmm6
		  pcmpgtw    xmm2, xmm13		//curr > next
		pcmpgtw    xmm5, xmm7
		 pcmpgtw    xmm6, xmm7
		  pand       xmm1, xmm2			//(curr > prev) and (curr > next)
		paddw      xmm10, xmm5
		pandn      xmm5, xmm12
		  pcmpgtw    xmm12, xmm0		//prev > curr
		 paddw      xmm10, xmm6
		 pandn      xmm6, xmm13
		  pcmpgtw    xmm13, xmm0		//next > curr
		paddw      xmm9, xmm5
		add        rsi, r9				//inc source ptr early for pre-load for next iter
		 paddw      xmm9, xmm6
		  pand       xmm12, xmm13 		//(prev > curr) and (next > curr)
		movdqa     xmm3, xmm10
		  por        xmm12, xmm1		//xmm12 = mask set if pixel is fluctuating
		pcmpeqb    xmm7, xmm7			//xmm7 = all -1s
		  pand       xmm12, r_edge_mask	//exclude left-edge and right-edge pixels

		//Average and store results, and pre-load next iteration's Middle and Top pixels
		psllw       xmm3, 8
		paddb		xmm7, xmm8			//xmm7 = all 7s
		movdqa		xmm1, scaletab_low	//xmm1 = {7, 6, 5, 4, 3, 2, 1, 0}
		movdqa		xmm2, scaletab_high	//xmm2 = {-, -, -, -, 11, 10, 9, 8}
		por         xmm3, xmm10			//low & high byte of each word both hold count; from here on xmm3 is bytes
		paddw       xmm9, xmm9			//sum *= 2
		 movdqa      xmm5, xmm12
		movdqa      xmm4, xmm3
		movdqa      xmm13, xmm3
		 pandn       xmm5, xmm0 		//xmm5 = pixels that will remain unsmoothed

		movq	xmm0, [rsi]				//pre-load pixels for next iteration

		pcmpgtb     xmm3, xmm7			//LO: xmm3 = mask set for elements that are 8->11, clear for ones that are 1->7
		 psubb       xmm4, xmm8			//HI: subtract 8: elements are either 0->3 or negative numbers
		  add         r10, r9 
		pandn       xmm3, xmm13			//LO: xmm3 = mixture of valid index numbers 1->7 and 0s
		 paddb       xmm4, xmm4			//HI: mul by two; turn word indexes to byte indexes
		  add         r11, r9
		paddb       xmm3, xmm3			//LO: mul by two; turn word indexes to byte indexes
		 paddb       xmm4, xmm11		//HI: high bytes of each word now index high bytes of scaletab words (also, negative elements will remain negative)
		  mov         rdx, rdi
		paddb       xmm3, xmm11			//LO: high bytes of each word now index high bytes of scaletab word data
		 pshufb      xmm2, xmm4			//HI: xmm2 = mixture of scaletab[8->11] and 0s as words
		  add         rdi, r8
		paddw       xmm9, xmm10			//sum += count
		pshufb      xmm1, xmm3			//LO: xmm1 = scaletab[0->7] as words; scaletab[0] value is conveniently 0.
		por			xmm1, xmm2
		  punpcklbw   xmm0, xmm15		//prep next iter's pixels
		  movdqa      xmm6, [r10]		//pre-load next iteration's Top pixels
		pmulhw		xmm1, xmm9			//xmm1 = averages

		dec			DWORD PTR height
		pand		xmm1, xmm12			//xmm1 = pixels that will be smoothed
		por			xmm5, xmm1			//xmm5 = result pixels
		packuswb	xmm5, xmm15			//convert back to bytes in low QWORD
		movdqa		xmm1, xmm0
		movdqa		xmm2, xmm0
		movq	[rdx], xmm5				//store processed pixels

		jnz  QWORD_LOOP

		sfence



FINISHED:

	}
}






#else	//defined DO_SPATIAL



void CLASS_NAME::DoFilter_SSSE3( const BYTE * currp, const long long src_pitch,
								 const BYTE * prevp, const BYTE * nextp,
								 const long long prv_pitch, const long long nxt_pitch,
								 BYTE * destp, const long long dst_pitch,
								 const int row_size, int height )
{
	static const __m128i multiplier3 = { _REPEAT_8_WORDS(10923) };	//multiplier from scaletab[3]
	static const __m128i counter_init = { _REPEAT_8_WORDS(3) };		//0x00030003000300030003000300030003
	const __m128i temp_thresh = this->temp_thresh;

	__asm
	{
		//rsi = current source row
		//rdi = current destination row
		//r8  = next frame row
		//r9  = previous frame row
		//r10 = src_pitch (const)
		//r11 = prv_pitch (const)
		//r12 = nxt_pitch (const)
		//r13 = dst_pitch (const)
		//rdx = row_size (const)
		//rcx = main loop counter
		//rax = height
		//rbx is never used

		mov		rsi, currp
		mov		r8, prevp
		mov		r9, nextp
		mov		r10, src_pitch
		xor		ecx, ecx
		movdqa	xmm14, temp_thresh
		pxor	xmm15, xmm15			//xmm15 is used for unpacking bytes into zero-extended words
		movdqa	xmm13, multiplier3
		mov		r11, prv_pitch
		mov		r12, nxt_pitch
		mov		rdi, destp
		mov		r13, dst_pitch
		movsxd	rdx, row_size
		movsxd	rax, height

		align(16)

Y_LOOP:

		//align(16)

X_LOOP:

		movq	xmm0, [rsi + rcx]		//xmm0 = Current 8 pixels in low QWORD
		movq	xmm1, [r8 + rcx]		//xmm1 = Previous frame pixels
		movq	xmm2, [r9 + rcx]		//xmm2 = Next frame pixels

		punpcklbw  xmm0, xmm15			//zero-extend to words
		movdqa     xmm14, xmm14			//Core 2 likes to have all input operands already in the execution pipeline, so just refresh this one before it's used again.
		movdqa     xmm10, counter_init	//counter is only needed to add to sum to complete FluxSmooth's original fixed-point avg calc.
		punpcklbw  xmm1, xmm15
		movdqa     xmm5, xmm0
		movdqa     xmm6, xmm0
		movdqa     xmm7, xmm0			//init sum
		punpcklbw  xmm2, xmm15
		movdqa     xmm8, xmm1
		movdqa     xmm9, xmm2

		//xmm0 = {p7, p6, p5, p4, p3, p2, p1, p0}
		//xmm1 = check and add for previous frame
		//xmm2 = check and add for next frame
		//xmm3 = mask set if next and prev pixel both outside thresh (count=1)
		//xmm4 = mask set if one but not both pixels outside thresh (count=2)
		//xmm7 = sum of included pixel values
		//xmm10 = counter of neighbor pixels included in the average
		//xmm5/6/8/9 = used for checking if pixels are fluctuating; xmm5 will be the combined result
		//xmm13 = multiplier3 (const)
		//xmm14 = temp_thresh (const)

		psubw      xmm1, xmm0
		 psubw      xmm2, xmm0
		  pcmpgtw    xmm5, xmm8			//curr > prev
		pabsw      xmm1, xmm1
		 pabsw      xmm2, xmm2
		  pcmpgtw    xmm6, xmm9			//curr > next
		pcmpgtw    xmm1, xmm14			//compare with temporal threshold (mask set if value is outside thresh)
		 pcmpgtw    xmm2, xmm14
		  movdqa     xmm3, xmm1
		  movdqa     xmm4, xmm1
		  pand       xmm3, xmm2			//xmm3 = mask set if neither pixel is included
		  pxor       xmm4, xmm2			//xmm4 = mask set if one but not both pixels are included
		paddw      xmm10, xmm1			//subtract 1 from counter if not within threshold
		pandn      xmm1, xmm8
		  pcmpgtw    xmm8, xmm0			//prev > curr
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm9
		  pcmpgtw    xmm9, xmm0			//next > curr
		paddw      xmm7, xmm1			//add included pixels to sum
		  pand       xmm5, xmm6			//(curr > prev) and (curr > next)
		 paddw      xmm7, xmm2
		  pand       xmm8, xmm9			//(prev > curr) and (next > curr)

		add		ecx, 8

		//Average - Since there's only 3 possible divisors, calculate for all of them and choose using masks.
		 movdqa     xmm1, xmm3
		paddw      xmm7, xmm7
		 por        xmm3, xmm4			//xmm3 = mask set for pixels with count=1 or 2; inversed later to get count=3 mask
		paddw      xmm7, xmm10			//xmm7 = sum * 2 + count
		 por        xmm5, xmm8			//xmm5 = mask set if pixel is fluctuating
		 pand       xmm1, xmm0			//xmm1 = pixels with count=1 (no avg needed)
		movdqa     xmm2, xmm7
		 movdqa     xmm6, xmm5
		pmulhw     xmm7, xmm13			//xmm7 = averages for count=3
		psrlw      xmm2, 2				//xmm2 = averages for count=2
		 pandn      xmm6, xmm0			//xmm6 = non-fluctuating pixels
		 pand       xmm4, xmm5			//xmm4 = count=2 mask with non-fluctuating pixels removed
		 pandn      xmm3, xmm5			//xmm3 = count=3 mask with non-fluctuating pixels removed
		 por        xmm1, xmm6			//xmm1 = non-fluctuating pixels and ones with count=1
		pand       xmm4, xmm2
		pand       xmm3, xmm7

		//Combine pixels into final result and store
		cmp        ecx, edx
		por        xmm1, xmm4
		pxor       xmm15, xmm15			//refresh xmm15 in the execution pipeline before the start of next iteration
		por        xmm1, xmm3			//xmm1 = result
		packuswb   xmm1, xmm15			//convert back to 8 bytes in low QWORD

		movq	[rdi + rcx - 8], xmm1	// sh0: Streamed store is not faster here!

		jl  X_LOOP


		// Advance to next row

		add		rsi, r10
		add		r8, r11
		add		r9, r12
		xor		ecx, ecx
		add		rdi, r13
		sub		eax, 1
		mov		height, eax

		jnz  Y_LOOP


		sfence
	}
}

#endif	//defined DO_SPATIAL


