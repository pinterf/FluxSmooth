// "YV12_SSE2.h"
// Devin Gardner - 2010/11/30
//
// YV12 version with SSE2 optimizations targeting Athlon 64.



#undef _2BYTES
#undef _REPEAT_8_WORDS
#define _2BYTES( _word )		((BYTE)(_word)), ((BYTE)((_word) >> 8))
#define _REPEAT_8_WORDS( _w )	_2BYTES(_w), _2BYTES(_w), _2BYTES(_w), _2BYTES(_w), _2BYTES(_w), _2BYTES(_w), _2BYTES(_w), _2BYTES(_w)



#ifdef DO_SPATIAL

void CLASS_NAME::DoFilter_SSE2( const BYTE * currp, const long long src_pitch,
								const BYTE * prevp, const BYTE * nextp,
								const long long prv_pitch, const long long nxt_pitch,
								BYTE * destp, const long long dst_pitch,
								const int row_size, int height )
{
	static const __m128  half = { 0.501f, 0.501f, 0.501f, 0.501f };	//for rounding to int
	static const __m128i counter_init = { _REPEAT_8_WORDS(11) };	//0x000B000B000B000B000B000B000B000B
	const __m128i spat_thresh = this->spat_thresh;
	const __m128i temp_thresh = this->temp_thresh;
	const __m128i l_edge_mask = { _2BYTES(0),  _2BYTES(-1), _2BYTES(-1), _2BYTES(-1),
								  _2BYTES(-1), _2BYTES(-1), _2BYTES(-1), _2BYTES(-1) };  //0xFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000
	__m128i r_edge_mask = { _REPEAT_8_WORDS(-1) };					//0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
	__m128i sum_temp;		//temp swap space

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

		mov		rsi, currp
		mov		r9, src_pitch
		movsxd	rdx, row_size
		mov		r12, prevp
		mov		r13, nextp
		mov		rdi, destp
		mov		r10, rsi
		mov		r11, rsi
		sub		r10, r9				//r10 = top pixel row
		add		r11, r9				//r11 = bottom pixel row
		sub		rdx, 8				//subtract right-edge pixels; they're handled in the loop's unrolled final iteration

		jna  JUST_ONE_QWORD

		align(16)


Y_LOOP:

		//-----------------------------------
		// First iteration of x loop is unrolled to handle special case of left frame edge.
		// Skip down to the comments for the code inside the main loop for a better idea of what's going on here.
		// The Middle Left, Top Left, and Bottom Left can't be done for the first pixel, but they still
		//  must be done for the rest of the pixels in the first QWORD.

		// Do Middle Left and Right, and prepare Top pixels
		movq	xmm0, [rsi]				//load first 8 pixels
		movq	xmm2, [rsi + 8]			//Middle Right
		pxor	xmm15, xmm15			//xmm15 is always 0, for unpacking bytes into zero-extended words
		mov		ecx, 8					//loop counter

		movq	xmm12, [r10]			//Start loading Top pixels
		movq	xmm13, [r10 + 8]

		punpcklbw	xmm0, xmm15			//xmm0 = {p7, p6, p5, p4, p3, p2, p1, p0}
		punpcklbw	xmm2, xmm15			//xmm2 = {p15, p14, p13, p12, p11, p10, p9, p8}
		movdqa		xmm1, xmm0			//xmm1 = {p7, p6, p5, p4, p3, p2, p1, p0}
		movdqa		xmm6, xmm0			//used to build Middle Right pixels

		pslldq     xmm1, 2				//xmm1 = {p6, p5, p4, p3, p2, p1, p0, 0}
		 pslldq     xmm2, 14			//xmm2 = {p8, 0, 0, 0, 0, 0, 0, 0}
		 psrldq     xmm6, 2				//xmm6 = {0, p7, p6, p5, p4, p3, p2, p1}

		movdqa	xmm14, spat_thresh
		movdqa	xmm7, xmm0				//initialize sum of included pixel values
		movdqa	xmm10, counter_init		//xmm10 = counter of neighbor pixels included in the average

		 por        xmm2, xmm6			//xmm2 = {p8, p7, p6, p5, p4, p3, p2, p1}
		  punpcklbw  xmm12, xmm15		//xmm12 = {t7, t6, t5, t4, t3, t2, t1, t0}
		  punpcklbw  xmm13, xmm15		//xmm13 = {t15, t14, t13, t12, t11, t10, t9, t8}
		movdqa     xmm3, xmm0
		movdqa     xmm5, xmm1
		 movdqa     xmm4, xmm0
		 movdqa     xmm6, xmm2
		psubusw    xmm3, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm4, xmm6
		 psubusw    xmm2, xmm0
		  movdqa     xmm11, xmm12
		por        xmm1, xmm3			//get absolute difference
		 por        xmm2, xmm4
		  movdqa     xmm4, xmm12
		  pslldq     xmm11, 2			//xmm11 = {t6, t5, t4, t3, t2, t1, t0, 0}
		pcmpgtw    xmm1, xmm14			//compare with spatial threshold
		 pcmpgtw    xmm2, xmm14
		  pslldq     xmm13, 14
		paddw      xmm10, xmm1			//subtract 1 from counter if not within threshold
		pandn      xmm1, xmm5
		  psrldq     xmm4, 2
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm6
		  por        xmm13, xmm4		//xmm13 = {t8, t7, t6, t5, t4, t3, t2, t1}
		paddw      xmm7, xmm1			//add included pixels to sum
		 paddw      xmm7, xmm2
		movdqa	sum_temp, xmm7

		// Do Top pixels, and prepare Bottom pixels
		movdqa	xmm7, xmm11
		movdqa	xmm8, xmm12
		movdqa	xmm9, xmm13
		movq	xmm12, [r11]			//Start loading Bottom pixels
		movq	xmm13, [r11 + 8]

		movdqa     xmm4, xmm0
		movdqa     xmm1, xmm7
		 movdqa     xmm5, xmm0
		 movdqa     xmm2, xmm8
		  movdqa     xmm6, xmm0
		  movdqa     xmm3, xmm9
		psubusw    xmm4, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm5, xmm2
		 psubusw    xmm2, xmm0
		  psubusw    xmm6, xmm3
		  psubusw    xmm3, xmm0
		   punpcklbw  xmm12, xmm15		//xmm12 = {b7, b6, b5, b4, b3, b2, b1, b0}
		por        xmm1, xmm4
		 por        xmm2, xmm5
		  por        xmm3, xmm6
		   punpcklbw  xmm13, xmm15		//xmm13 = {b15, b14, b13, b12, b11, b10, b9, b8}
		   movdqa     xmm11, xmm12
		movdqa	xmm6, sum_temp
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		   movdqa     xmm5, xmm12
		paddw      xmm10, xmm1
		pandn      xmm1, xmm7
		   pslldq     xmm13, 14
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm8
		   psrldq     xmm5, 2
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm9
		paddw      xmm6, xmm1
		   pslldq     xmm11, 2			//xmm11 = {b6, b5, b4, b3, b2, b1, b0, 0}
		   por        xmm13, xmm5		//xmm13 = {b8, b7, b6, b5, b4, b3, b2, b1}
		 paddw      xmm6, xmm2
		  paddw      xmm6, xmm3

		// Do Bottom pixels, and prepare Prev/Next Frame pixels
		movdqa	xmm7, xmm11
		movdqa	xmm8, xmm12
		movdqa	xmm9, xmm13
		movq	xmm12, [r12]			//Previous Frame
		movq	xmm13, [r13]			//Next Frame
		movdqa	xmm11, xmm6				//xmm11 = sum

		movdqa     xmm4, xmm0
		movdqa     xmm1, xmm7
		 movdqa     xmm5, xmm0
		 movdqa     xmm2, xmm8
		  movdqa     xmm6, xmm0
		  movdqa     xmm3, xmm9
		psubusw    xmm4, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm5, xmm2
		 psubusw    xmm2, xmm0
		  psubusw    xmm6, xmm3
		  psubusw    xmm3, xmm0
		por        xmm1, xmm4
		 por        xmm2, xmm5
		  por        xmm3, xmm6
   		   punpcklbw  xmm12, xmm15
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		   punpcklbw  xmm13, xmm15
		paddw      xmm10, xmm1
		pandn      xmm1, xmm7
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm8
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm9
		paddw      xmm11, xmm1
		 paddw      xmm11, xmm2
		  paddw      xmm11, xmm3

		// Do Previous and Next Frame pixels, and determine which pixels were fluctuating
		movdqa     xmm9, temp_thresh
		movdqa     xmm3, xmm0
		movdqa     xmm1, xmm12
		 movdqa     xmm4, xmm0
		 movdqa     xmm2, xmm13
		  movdqa     xmm5, xmm0			//begin testing for which pixels are fluctuating
		  movdqa     xmm6, xmm0
		psubusw    xmm3, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm4, xmm2
		 psubusw    xmm2, xmm0
		  movdqa     xmm7, xmm12
		  movdqa     xmm8, xmm13
		por        xmm1, xmm3			//get abs diff
		 por        xmm2, xmm4
		  pcmpgtw    xmm5, xmm12		//curr > prev
		  pcmpgtw    xmm6, xmm13		//curr > next
		  pcmpgtw    xmm7, xmm0 		//prev > curr
		  pcmpgtw    xmm8, xmm0 		//next > curr
		pcmpgtw    xmm1, xmm9			//compare with temporal threshold
		 pcmpgtw    xmm2, xmm9
		  pand       xmm5, xmm6			//(curr > prev) and (curr > next)
		  pand       xmm7, xmm8 		//(prev > curr) and (next > curr)
		paddw      xmm10, xmm1			//sub from count
		pandn      xmm1, xmm12
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm13
		paddw      xmm11, xmm1			//add to sum
		  por        xmm5, xmm7			//xmm5 = mask set if pixel is fluctuating
		 paddw      xmm11, xmm2
		  pand       xmm5, l_edge_mask	//exclude the left-edge pixel from being smoothed

		// Average and store results
		movdqa		xmm4, xmm10			//copy count for high data
		punpcklwd	xmm10, xmm15
		mov		eax, 16					//generate an address ahead of time for loading next iteration's Middle Right pixels
		cmp		edx, 8					//test if row pitch is only 2 QWORDS (8 was already subbed from rdx)
		punpckhwd	xmm4, xmm15
		cvtdq2ps	xmm3, xmm10			//xmm3 = low data for count
		cmovle	eax, ecx				//if so, load from Middle Center pixels instead; the main loop is gonna get skipped.
		 movdqa      xmm10, xmm11		//copy sum for high data
		cvtdq2ps	xmm4, xmm4			//xmm4 = high data for count

		rcpps		xmm1, xmm3
		 punpcklwd   xmm11, xmm15
		 punpckhwd   xmm10, xmm15
		rcpps		xmm2, xmm4
		 cvtdq2ps    xmm11, xmm11 		//xmm11 = low data for sum
		movdqa		xmm6, xmm1
		movdqa		xmm7, xmm2

		mulps		xmm6, xmm6
		addps		xmm1, xmm1
		 movdqa      xmm12, half
		 cvtdq2ps    xmm10, xmm10		//xmm10 = high data for sum
		mulps		xmm7, xmm7
		addps		xmm2, xmm2
		movq	xmm8, [rsi + 8]			//pre-load main pixels for next iteration
		mulps		xmm3, xmm6
		subps		xmm1, xmm3
		mulps		xmm4, xmm7
		subps		xmm2, xmm4
		movq	xmm7, [rsi + rax]		//pre-load Middle Right pixels for next iteration

		mulps		xmm1, xmm11			//multiply sum by reciprocal of count to get averages
		addps		xmm1, xmm12			//add 0.5 to round
		mulps		xmm2, xmm10
		cvttps2dq	xmm1, xmm1			//convert to int with truncation
		addps		xmm2, xmm12
		 movdqa      xmm3, xmm5
		cvttps2dq	xmm2, xmm2
		 movdqa      xmm6, xmm0 		//save for use as Middle Left pixels for next iteration
		 pandn       xmm3, xmm0			//xmm3 = pixels that will remain unsmoothed
		packssdw	xmm1, xmm2			//covert back to 8 words - the smoothed pixels

		pand		xmm1, xmm5			//xmm1 = pixels that will be smoothed
		 punpcklbw   xmm8, xmm15		//prep next iteration's pixels
		por			xmm1, xmm3			//xmm1 = result pixels
		 punpcklbw   xmm7, xmm15		//prep next iter's Middle Right pixels
		 movdqa      xmm0, xmm8
		packuswb	xmm1, xmm15			//convert back to 8 bytes in low QWORD
		movq	[rdi], xmm1				//store processed pixels

		//xmm0 = next iter's pixels, unpacked to words
		//xmm6 = next iter's Middle Left pixels, unpacked
		//xmm7 = next iter's Middle Right pixels, unpacked


		test	edx, edx			//Test if each row is just a single QWORD.
		jle  NEXT_ROW

		//Skip main loop if row_size <= 16; row is just a left-edge QWORD and a right-edge QWORD.
		//If not, branch prediction will eliminate this conditional jump.
		cmp		edx, 8				//8 was already subtracted from rdx at beginning of function
		jle  X_POST_ITER				

		align(16)





X_LOOP:

		//-----------------------------------
		// Main Loop - filter 8 pixels from a QWORD - never handles the first or last QWORD in a row.
		// xmm0 is prepared with current 8 pixels zero-extended to words
		// xmm6 & xmm7 are prepared with the Middle Left and Middle Right pixels zero-extended to words
		// xmm0 = {p7, p6, p5, p4, p3, p2, p1, p0}


		// Do Middle Left and Middle Right, and prepare Top pixels

		movq	xmm11, [r10 + rcx - 8]	//Start loading Top pixels
		movq	xmm12, [r10 + rcx]
		movq	xmm13, [r10 + rcx + 8]

		movdqa	xmm1, xmm6				//xmm1 = {p-1, p-2, p-3, p-4, p-5, p-6, p-7, p-8}
		movdqa	xmm2, xmm7				//xmm2 = {p15, p14, p13, p12, p11, p10, p9, p8}
		movdqa	xmm5, xmm0				//used to build Middle Left pixels
		movdqa	xmm6, xmm0				//used to build Middle Right pixels

		psrldq     xmm1, 14				//xmm1 = {0, 0, 0, 0, 0, 0, 0, p-1}
		 pslldq     xmm2, 14			//xmm2 = {p8, 0, 0, 0, 0, 0, 0, 0}
		pslldq     xmm5, 2				//xmm5 = {p6, p5, p4, p3, p2, p1, p0, 0}
		 psrldq     xmm6, 2				//xmm6 = {0, p7, p6, p5, p4, p3, p2, p1}
		movdqa     xmm10, counter_init	//xmm10 = counter of neighbor pixels included in the average
		movdqa     xmm7, xmm0			//initialize sum of included pixel values
		por        xmm1, xmm5			//xmm1 = {p6, p5, p4, p3, p2, p1, p0, p-1}
		 por        xmm2, xmm6			//xmm2 = {p8, p7, p6, p5, p4, p3, p2, p1}
		  punpcklbw  xmm11, xmm15		//xmm11 = {t-1, t-2, t-3, t-4, t-5, t-6, t-7, t-8}
		  punpcklbw  xmm12, xmm15		//xmm12 = {t7, t6, t5, t4, t3, t2, t1, t0}
		  punpcklbw  xmm13, xmm15		//xmm13 = {t15, t14, t13, t12, t11, t10, t9, t8}
		movdqa     xmm3, xmm0
		movdqa     xmm5, xmm1
		 movdqa     xmm4, xmm0
		 movdqa     xmm6, xmm2
		psubusw    xmm1, xmm0
		psubusw    xmm3, xmm5
		 psubusw    xmm2, xmm0
		 psubusw    xmm4, xmm6
		  psrldq     xmm11, 14			//xmm11 = {0, 0, 0, 0, 0, 0, 0, t-1}
		  pslldq     xmm13, 14			//xmm13 = {t8, 0, 0, 0, 0, 0, 0, 0}
		por        xmm1, xmm3			//get absolute difference
		 por        xmm2, xmm4
		  movdqa     xmm3, xmm12		//start building Top Left and Top Right pixels
		  movdqa     xmm4, xmm12
		pcmpgtw    xmm1, xmm14			//compare with spatial threshold
		 pcmpgtw    xmm2, xmm14
		  pslldq     xmm3, 2			//xmm3 = {t6, t5, t4, t3, t2, t1, t0, 0}
		  psrldq     xmm4, 2			//xmm3 = {0, t7, t6, t5, t4, t3, t2, t1}
		paddw      xmm10, xmm1			//subtract 1 from counter if not within threshold
		pandn      xmm1, xmm5
		  por        xmm11, xmm3		//xmm11 = {t6, t5, t4, t3, t2, t1, t0, t-1}
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm6
		  por        xmm13, xmm4		//xmm13 = {t8, t7, t6, t5, t4, t3, t2, t1}
		paddw      xmm7, xmm1			//add included pixels to sum
		 paddw      xmm7, xmm2
		movdqa	sum_temp, xmm7

		// Do Top pixels, and prepare Bottom pixels

		movdqa	xmm7, xmm11
		movdqa	xmm8, xmm12
		movdqa	xmm9, xmm13
		movq	xmm11, [r11 + rcx - 8]	//Start loading Bottom pixels
		movq	xmm12, [r11 + rcx]
		movq	xmm13, [r11 + rcx + 8]

		movdqa     xmm4, xmm0
		movdqa     xmm1, xmm7
		 movdqa     xmm5, xmm0
		 movdqa     xmm2, xmm8
		  movdqa     xmm6, xmm0
		  movdqa     xmm3, xmm9
		psubusw    xmm4, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm5, xmm2
		 psubusw    xmm2, xmm0
		  psubusw    xmm6, xmm3
		  psubusw    xmm3, xmm0
		   punpcklbw  xmm11, xmm15		//xmm11 = {b-1, b-2, b-3, b-4, b-5, b-6, b-7, b-8}
		   punpcklbw  xmm12, xmm15		//xmm12 = {b7, b6, b5, b4, b3, b2, b1, b0}
		por        xmm1, xmm4			//get abs diff
		 por        xmm2, xmm5
		  por        xmm3, xmm6
		   punpcklbw  xmm13, xmm15		//xmm13 = {b15, b14, b13, b12, b11, b10, b9, b8}
		   movdqa     xmm4, xmm12
		   movdqa     xmm5, xmm12
		movdqa	xmm6, sum_temp
		pcmpgtw    xmm1, xmm14			//compare with thresh
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		paddw      xmm10, xmm1			//sub from counter
		pandn      xmm1, xmm7
		   psrldq     xmm11, 14			//xmm11 = {0, 0, 0, 0, 0, 0, 0, b-1}
		   pslldq     xmm4, 2
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm8
		   pslldq     xmm13, 14			//xmm13 = {b8, 0, 0, 0, 0, 0, 0, 0}
		   psrldq     xmm5, 2
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm9
		paddw      xmm6, xmm1			//add to sum
		   por        xmm11, xmm4		//xmm11 = {b6, b5, b4, b3, b2, b1, b0, b-1}
		   por        xmm13, xmm5		//xmm13 = {b8, b7, b6, b5, b4, b3, b2, b1}
		 paddw      xmm6, xmm2
		  paddw      xmm6, xmm3

		// Do Bottom pixels, and prepare Prev/Next Frame pixels

		movdqa	xmm7, xmm11
		movdqa	xmm8, xmm12
		movdqa	xmm9, xmm13
		movq	xmm12, [r12 + rcx]		//Previous Frame
		movq	xmm13, [r13 + rcx]		//Next Frame
		movdqa	xmm11, xmm6				//xmm11 = sum

		movdqa     xmm4, xmm0
		movdqa     xmm1, xmm7
		 movdqa     xmm5, xmm0
		 movdqa     xmm2, xmm8
		  movdqa     xmm6, xmm0
		  movdqa     xmm3, xmm9
		psubusw    xmm4, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm5, xmm2
		 psubusw    xmm2, xmm0
		  psubusw    xmm6, xmm3
		  psubusw    xmm3, xmm0
		por        xmm1, xmm4			//get abs diff
		 por        xmm2, xmm5
		  por        xmm3, xmm6
		   punpcklbw  xmm12, xmm15
		pcmpgtw    xmm1, xmm14			//compare with thresh
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		   punpcklbw  xmm13, xmm15
		paddw      xmm10, xmm1			//sub from counter
		pandn      xmm1, xmm7
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm8
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm9
		paddw      xmm11, xmm1			//add to sum
		 paddw      xmm11, xmm2
		  paddw      xmm11, xmm3

		// Do Previous Frame and Next Frame pixels
		// Also determine which pixels were fluctuating

		movdqa	xmm9, temp_thresh

		movdqa     xmm3, xmm0
		movdqa     xmm1, xmm12
		 movdqa     xmm4, xmm0
		 movdqa     xmm2, xmm13
		  movdqa     xmm5, xmm0			//begin testing for which pixels are fluctuating
		  movdqa     xmm6, xmm0
		psubusw    xmm3, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm4, xmm2
		 psubusw    xmm2, xmm0
		  movdqa     xmm7, xmm12
		  movdqa     xmm8, xmm13
		por        xmm1, xmm3			//get abs diff
		 por        xmm2, xmm4
		  pcmpgtw    xmm5, xmm12		//curr > prev
		  pcmpgtw    xmm6, xmm13		//curr > next
		  pcmpgtw    xmm7, xmm0 		//prev > curr
		  pcmpgtw    xmm8, xmm0 		//next > curr
		pcmpgtw    xmm1, xmm9			//compare with temporal threshold
		 pcmpgtw    xmm2, xmm9
		  pand       xmm5, xmm6			//(curr > prev) and (curr > next)
		  pand       xmm7, xmm8 		//(prev > curr) and (next > curr)
		paddw      xmm10, xmm1			//sub from counter
		pandn      xmm1, xmm12
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm13
		paddw      xmm11, xmm1			//add to sum
		  por        xmm5, xmm7			//xmm5 = mask set if pixel is fluctuating
		 paddw      xmm11, xmm2

		//xmm5 = mask for fluctuating pixels
		//xmm10 = count
		//xmm11 = sum

		// Average and store results

		//xmm1/xmm2 = low & high data for reciprocal of count
		//xmm3/xmm4 = low & high for count
		//xmm6/xmm7 = temps for N-R calc; later, pre-loaded left/right pixels for next loop iter
		//xmm11/xmm10 = low & high data for sum  (11=low, 10=high)
		//xmm12 = constant 0.5 for rounding back to int

		movdqa		xmm4, xmm10			//copy count for high data
		punpcklwd	xmm10, xmm15
		lea		eax, [rcx + 16]			//generate an address ahead of time for loading next iteration's Middle Right pixels
		add		ecx, 8					//inc loop counter early
		punpckhwd	xmm4, xmm15
		cmp		ecx, edx				//do loop test way early
		cvtdq2ps	xmm3, xmm10			//xmm3 = low data for count
		cmovge	eax, ecx				//if this is last iteration, load from Middle Center pixels instead - this also avoids potentially loading from mem we don't own.
		 movdqa      xmm10, xmm11		//copy sum for high data
		cvtdq2ps	xmm4, xmm4			//xmm4 = high data for count

		//Get reciprocals and do an iteration of Newton-Raphson refinement on them to make them more accurate
		//  result = 2r - (d * r^2)		(d = divisor; r = fast-reciprocal of d)
		//xmm3 = low d;			xmm4 = high d
		//xmm1/xmm6 = low r;	xmm2/xmm7 = high r
		rcpps		xmm1, xmm3
		 punpcklwd   xmm11, xmm15
		 punpckhwd   xmm10, xmm15
		rcpps		xmm2, xmm4
		 cvtdq2ps    xmm11, xmm11 		//xmm11 = low data for sum
		movdqa		xmm6, xmm1
		movdqa		xmm7, xmm2

		mulps		xmm6, xmm6
		addps		xmm1, xmm1
		 movdqa      xmm12, half
		 cvtdq2ps    xmm10, xmm10		//xmm10 = high data for sum
		mulps		xmm7, xmm7
		addps		xmm2, xmm2

		movq	xmm8, [rsi + rcx]		//pre-load main pixels for next iteration
		mulps		xmm3, xmm6
		subps		xmm1, xmm3
		mulps		xmm4, xmm7
		subps		xmm2, xmm4
		movq	xmm7, [rsi + rax]		//pre-load Middle Right pixels for next iteration

		mulps		xmm1, xmm11			//multiply sum by reciprocal of count to get averages
		addps		xmm1, xmm12			//add 0.5 to round
		mulps		xmm2, xmm10
		cvttps2dq	xmm1, xmm1			//convert to int with truncation
		addps		xmm2, xmm12
		 movdqa      xmm3, xmm5
		 movdqa      xmm6, xmm0 		//save for use as Middle Left pixels for next iteration
		cvttps2dq	xmm2, xmm2
		 pandn       xmm3, xmm0			//xmm3 = pixels that will remain unsmoothed
		packssdw	xmm1, xmm2			//covert back to 8 words - the smoothed pixels

		pand		xmm1, xmm5			//xmm1 = pixels that will be smoothed
		 punpcklbw   xmm8, xmm15		//prep next iteration's pixels
		por			xmm1, xmm3			//xmm1 = result pixels
		 punpcklbw   xmm7, xmm15		//prep next iter's Middle Right pixels
		 movdqa      xmm0, xmm8
		packuswb	xmm1, xmm15			//convert pixels back to bytes in low QWORD
		movq	[rdi + rcx - 8], xmm1	// sh0: Streamed store is not faster here!

		//xmm0 = next iter's pixels, unpacked to words
		//xmm6 = next iter's Middle Left pixels, unpacked
		//xmm7 = next iter's Middle Right pixels, unpacked

		jl  X_LOOP





X_POST_ITER:

		//-----------------------------------
		// Last iteration of loop is also unrolled, for the right-edge pixels.
		// xmm0 is prepared with current 8 pixels zero-extended to words
		// xmm6 is prepared with Middle Left pixels and xmm7 has the Middle Center pixels
		// xmm0 = {p7, p6, p5, p4, p3, p2, p1, p0}


		// Do Middle Left and Middle Right, and prepare Top pixels
		movq	xmm11, [r10 + rcx - 8]	//Start loading Top pixels
		movq	xmm12, [r10 + rcx]

		movdqa	xmm1, xmm6				//xmm1 = {p-1, p-2, p-3, p-4, p-5, p-6, p-7, p-8}
		movdqa	xmm5, xmm0				//used to build Middle Left pixels
		movdqa	xmm2, xmm7				//xmm2 = {p7, p6, p5, p4, p3, p2, p1, p0}

		psrldq     xmm1, 14				//xmm1 = {0, 0, 0, 0, 0, 0, 0, p-1}
		pslldq     xmm5, 2				//xmm5 = {p6, p5, p4, p3, p2, p1, p0, 0}
		 psrldq     xmm2, 2				//xmm2 = {0, p7, p6, p5, p4, p3, p2, p1}
		movdqa     xmm10, counter_init
		movdqa     xmm7, xmm0			//xmm7 = sum
		por        xmm1, xmm5			//xmm1 = {p6, p5, p4, p3, p2, p1, p0, p-1}
		  punpcklbw  xmm11, xmm15		//xmm11 = {t-1, t-2, t-3, t-4, t-5, t-6, t-7, t-8}
		  punpcklbw  xmm12, xmm15		//xmm12 = {t7, t6, t5, t4, t3, t2, t1, t0}
		movdqa     xmm3, xmm0
		movdqa     xmm5, xmm1
		 movdqa     xmm4, xmm0
		 movdqa     xmm6, xmm2
		psubusw    xmm3, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm4, xmm2
		 psubusw    xmm2, xmm0
		  movdqa     xmm13, xmm12
		por        xmm1, xmm3
		 por        xmm2, xmm4
		  psrldq     xmm11, 14			//xmm11 = {0, 0, 0, 0, 0, 0, 0, t-1}
		  movdqa     xmm3, xmm12		  
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  psrldq     xmm13, 2			//xmm13 = {0, t7, t6, t5, t4, t3, t2, t1}
		paddw      xmm10, xmm1
		pandn      xmm1, xmm5
		  pslldq     xmm3, 2			//xmm3 = {t6, t5, t4, t3, t2, t1, t0, 0}
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm6
		  por        xmm11, xmm3		//xmm11 = {t6, t5, t4, t3, t2, t1, t0, t-1}
		paddw      xmm7, xmm1
		 paddw      xmm7, xmm2
		movdqa	sum_temp, xmm7

		// Do Top pixels, and prepare Bottom pixels
		movdqa	xmm7, xmm11
		movdqa	xmm8, xmm12
		movdqa	xmm9, xmm13
		movq	xmm11, [r11 + rcx - 8]	//Start loading Bottom pixels
		movq	xmm12, [r11 + rcx]

		movdqa     xmm4, xmm0
		movdqa     xmm1, xmm7
		 movdqa     xmm5, xmm0
		 movdqa     xmm2, xmm8
		  movdqa     xmm6, xmm0
		  movdqa     xmm3, xmm9
		psubusw    xmm4, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm5, xmm2
		 psubusw    xmm2, xmm0
		  psubusw    xmm6, xmm3
		  psubusw    xmm3, xmm0
		   punpcklbw  xmm11, xmm15		//xmm11 = {b-1, b-2, b-3, b-4, b-5, b-6, b-7, b-8}
		   punpcklbw  xmm12, xmm15		//xmm12 = {b7, b6, b5, b4, b3, b2, b1, b0}
		por        xmm1, xmm4
		 por        xmm2, xmm5
		  por        xmm3, xmm6
		   psrldq     xmm11, 14			//xmm11 = {0, 0, 0, 0, 0, 0, 0, b-1}
		   movdqa     xmm4, xmm12
		movdqa	xmm6, sum_temp
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		   movdqa     xmm13, xmm12
		paddw      xmm10, xmm1
		pandn      xmm1, xmm7
		   pslldq     xmm4, 2			//xmm4 = {b6, b5, b4, b3, b2, b1, b0, 0}
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm8
		   por        xmm11, xmm4		//xmm11 = {b6, b5, b4, b3, b2, b1, b0, b-1}
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm9
		paddw      xmm6, xmm1
		   psrldq     xmm13, 2			//xmm13 = {0, b7, b6, b5, b4, b3, b2, b1}
		 paddw      xmm6, xmm2
		  paddw      xmm6, xmm3

		// Do Bottom pixels, and prepare Prev/Next Frame pixels
		movdqa	xmm7, xmm11
		movdqa	xmm8, xmm12
		movdqa	xmm9, xmm13
		movq	xmm12, [r12 + rcx]		//Previous Frame
		movq	xmm13, [r13 + rcx]		//Next Frame
		movdqa	xmm11, xmm6				//xmm11 = sum

		movdqa     xmm4, xmm0
		movdqa     xmm1, xmm7
		 movdqa     xmm5, xmm0
		 movdqa     xmm2, xmm8
		  movdqa     xmm6, xmm0
		  movdqa     xmm3, xmm9
		psubusw    xmm4, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm5, xmm2
		 psubusw    xmm2, xmm0
		  psubusw    xmm6, xmm3
		  psubusw    xmm3, xmm0
		por        xmm1, xmm4
		 por        xmm2, xmm5
		  por        xmm3, xmm6
   		   punpcklbw  xmm12, xmm15
		pcmpgtw    xmm1, xmm14
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm3, xmm14
		   punpcklbw  xmm13, xmm15
		paddw      xmm10, xmm1
		pandn      xmm1, xmm7
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm8
		  paddw      xmm10, xmm3
		  pandn      xmm3, xmm9
		paddw      xmm11, xmm1
		 paddw      xmm11, xmm2
		  paddw      xmm11, xmm3

		// Do Previous and Next Frame pixels, and determine which pixels were fluctuating
		movdqa     xmm9, temp_thresh
		movdqa     xmm3, xmm0
		movdqa     xmm1, xmm12
		 movdqa     xmm4, xmm0
		 movdqa     xmm2, xmm13
		  movdqa     xmm5, xmm0			//begin testing for which pixels are fluctuating
		  movdqa     xmm6, xmm0
		psubusw    xmm3, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm4, xmm2
		 psubusw    xmm2, xmm0
		  movdqa     xmm7, xmm12
		  movdqa     xmm8, xmm13
		por        xmm1, xmm3			//get abs diff
		 por        xmm2, xmm4
		  pcmpgtw    xmm5, xmm12		//curr > prev
		  pcmpgtw    xmm6, xmm13		//curr > next
		  pcmpgtw    xmm7, xmm0 		//prev > curr
		  pcmpgtw    xmm8, xmm0 		//next > curr
		pcmpgtw    xmm1, xmm9			//compare with temporal threshold
		 pcmpgtw    xmm2, xmm9
		  pand       xmm5, xmm6			//(curr > prev) and (curr > next)
		  pand       xmm7, xmm8 		//(prev > curr) and (next > curr)
		paddw      xmm10, xmm1			//sub from counter
		pandn      xmm1, xmm12
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm13
		paddw      xmm11, xmm1			//add to sum
		  por        xmm5, xmm7			//xmm5 = mask set if pixel is fluctuating
		 paddw      xmm11, xmm2
		  pand       xmm5, r_edge_mask	//exclude the right-edge pixel and any pad bytes from being smoothed

		// Average and store results
		movdqa		xmm4, xmm10			//copy count for high data
		punpcklwd	xmm10, xmm15
		punpckhwd	xmm4, xmm15
		cvtdq2ps	xmm3, xmm10			//xmm3 = low data for count
		 movdqa      xmm10, xmm11		//copy sum for high data
		cvtdq2ps	xmm4, xmm4			//xmm4 = high data for count

		rcpps		xmm1, xmm3
		 punpcklwd   xmm11, xmm15
		 punpckhwd   xmm10, xmm15
		rcpps		xmm2, xmm4
		 cvtdq2ps    xmm11, xmm11 		//xmm11 = low data for sum
		movdqa		xmm6, xmm1
		movdqa		xmm7, xmm2

		mulps		xmm6, xmm6
		addps		xmm1, xmm1
		 movdqa      xmm12, half
		mulps		xmm7, xmm7
		addps		xmm2, xmm2
		 cvtdq2ps    xmm10, xmm10		//xmm10 = high data for sum
		mulps		xmm3, xmm6
		subps		xmm1, xmm3
		mulps		xmm4, xmm7
		subps		xmm2, xmm4

		mulps		xmm1, xmm11
		addps		xmm1, xmm12
		mulps		xmm2, xmm10
		cvttps2dq	xmm1, xmm1
		addps		xmm2, xmm12
		 movdqa      xmm3, xmm5
		cvttps2dq	xmm2, xmm2
		 pandn       xmm3, xmm0			//xmm3 = pixels that will remain unsmoothed
		packssdw	xmm1, xmm2			//covert back to 8 words

		pand		xmm1, xmm5			//xmm1 = pixels that will be smoothed
		por			xmm1, xmm3			//xmm1 = result pixels
		packuswb	xmm1, xmm15			//convert back to bytes in low QWORD
		movq	[rdi + rcx], xmm1


NEXT_ROW:

		// Advance to next row

		mov		eax, height
		add		rsi, r9
		add		r10, r9
		add		r11, r9
		add		r12, prv_pitch
		add		r13, nxt_pitch
		add		rdi, dst_pitch
		sub		eax, 1
		mov		height, eax

		jnz  Y_LOOP

		sfence
		jmp  FINISHED




JUST_ONE_QWORD:

		movdqa		xmm1, l_edge_mask
		movdqa		xmm2, r_edge_mask
		pand		xmm1, xmm2
		movdqa		l_edge_mask, xmm1	//now excludes both left and right edge
		jmp  Y_LOOP


FINISHED:

	}
}





#else	//defined DO_SPATIAL



void CLASS_NAME::DoFilter_SSE2( const BYTE * currp, const long long src_pitch,
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
		xor		ecx, ecx
		mov		r10, src_pitch
		movdqa	xmm13, multiplier3
		pxor	xmm15, xmm15			//xmm15 is used for unpacking bytes into zero-extended words
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
		movdqa	   xmm14, temp_thresh
		punpcklbw  xmm1, xmm15
		movdqa     xmm5, xmm0
		punpcklbw  xmm2, xmm15
		movdqa     xmm6, xmm0
		movdqa     xmm7, xmm0			//init sum
		movdqa	   xmm10, counter_init	//counter is only needed to add to sum to complete FluxSmooth's original fixed-point avg calc.

		//xmm0 = {p7, p6, p5, p4, p3, p2, p1, p0}
		//xmm1 = check and add for previous frame
		//xmm2 = check and add for next frame
		//xmm3 = mask set if next and prev pixel both outside thresh (count=1)
		//xmm4 = mask set if one but not both pixels outside thresh (count=2)
		//xmm7 = sum of included pixel values
		//xmm10 = counter of neighbor pixels included in the avg
		//xmm5/6/8/9 = used for checking if pixels are fluctuating; xmm5 will be the combined result
		//xmm11/xmm12 = copy of prev/next frame pixels
		//xmm13 = multiplier3 (const)
		//xmm14 = temp_thresh (const)

		movdqa     xmm3, xmm0
		movdqa     xmm8, xmm1
		 movdqa     xmm4, xmm0
		 movdqa     xmm9, xmm2
		  movdqa     xmm11, xmm1
		  movdqa     xmm12, xmm2
		psubusw    xmm3, xmm1
		psubusw    xmm1, xmm0
		 psubusw    xmm4, xmm2
		 psubusw    xmm2, xmm0
		por        xmm1, xmm3			//get absolute difference
		  pcmpgtw    xmm5, xmm8			//curr > prev
		 por        xmm2, xmm4			//old values of xmm3/xmm4 no longer needed
		  pcmpgtw    xmm6, xmm9			//curr > next
		pcmpgtw    xmm1, xmm14			//compare with temporal threshold (mask set if value is outside threshold)
		  pcmpgtw    xmm8, xmm0			//prev > curr
		 pcmpgtw    xmm2, xmm14
		  pcmpgtw    xmm9, xmm0			//next > curr
		  movdqa     xmm3, xmm1
		  movdqa     xmm4, xmm1
		paddw      xmm10, xmm1			//subtract 1 from counter if not within threshold
		pandn      xmm1, xmm11
		  pand       xmm3, xmm2			//xmm3 = mask set if neither pixel is included
		  pxor       xmm4, xmm2			//xmm4 = mask set if one but not both pixels are included
		 paddw      xmm10, xmm2
		 pandn      xmm2, xmm12
		paddw      xmm7, xmm1			//add included pixels to sum
		  pand       xmm5, xmm6			//(curr > prev) and (curr > next)
		 paddw      xmm7, xmm2
		  pand       xmm8, xmm9			//(prev > curr) and (next > curr)

		add		ecx, 8

		//Average - Since there's only 3 possible divisors, calculate for all of them and choose using masks.
		paddw      xmm7, xmm7
		  movdqa     xmm1, xmm3
		  por        xmm5, xmm8			//xmm5 = mask set if pixel is fluctuating
		paddw      xmm7, xmm10			//xmm7 = sum * 2 + count
		  por        xmm3, xmm4			//xmm3 = mask set for pixels with count=1 or 2; inversed later to get count=3 mask
		  pand       xmm1, xmm0			//xmm1 = pixels with count=1 (no avg needed)
		movdqa     xmm2, xmm7
		  movdqa     xmm6, xmm5
		pmulhw     xmm7, xmm13			//xmm7 = averages for count=3
		psrlw      xmm2, 2				//xmm2 = averages for count=2
		  pandn      xmm6, xmm0			//xmm6 = pixels that will remain unsmoothed
		  pand       xmm4, xmm5			//xmm4 = count=2 mask with non-fluctuating pixels removed
		  pandn      xmm3, xmm5			//xmm3 = count=3 mask with non-fluctuating pixels removed
		  por        xmm1, xmm6			//xmm1 = non-fluctuating pixels and ones with count=1
		pand       xmm4, xmm2			
		pand       xmm3, xmm7			

		//Combine pixels into final result and store
		cmp        ecx, edx
		por        xmm1, xmm4
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


