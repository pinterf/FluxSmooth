# FluxSmooth - pfmod
Avisynth filter for spatio-temporal smoothing of fluctuations

By Ross Thomas <ross@grinfinity.com>

There is no copyright on this code, and there are no conditions
on its distribution or use. Do with it what you will.

- (20190402) v1.3, rewrite by pinterf
  - project moved to github: https://github.com/pinterf/FluxSmooth
  - Built using Visual Studio 2017, additional LLVM 8.0 clang support
  - Changed to AVS 2.6 plugin interface
  - x64 build for Avisynth+
  - Added version resource to DLL
  - Removed MMX support, requires SSE2. (Though pure C is still available in the source)
  - Drop all inline assembly, SIMD intrinsics based on C code, SSE2, SSE4.1 and AVX2 optimizations
  - Single DLL, optimizations for different CPU instruction sets are chosen automatically.
  - Reports MT Modes for Avisynth+: MT_NICE_FILTER
  - Added Y, YV411, YV16 and YV24, 10-16 bits 4:2:0, 4:2:2, 4:4:4, planar RGB(A) 8-16 bits support besides existing YV12
  - (YUY2 support with workaround: internally converted to YV16, process and convert back 
    conversion is lossless, but slower than using native YV16)
  - New parameters: bool "luma", bool "chroma" (default true) to disable processing of luma/chroma planes
- (20101130) x64 inline assembler optimized version by Devin Gardner
- (2002-2004) FluxSmooth v1.1b 
  Original version by Ross Thomas
  http://web.archive.org/web/20070225212908/http://bengal.missouri.edu/~kes25c/FluxSmooth-1.1b.zip
  https://forum.doom9.org/showthread.php?t=38296

Installation note: Previous DLL versions named differently (FluxSmoothSSE2.DLL, FluxSmoothSSSE3) should be deleted from your plugin folder.
From version 1.3 a single DLL exists, which automatically chosen CPU optimization (SSE2, SSE4.1, AVX2)

Links
=====
Project: https://github.com/pinterf/FluxSmooth
Forum: ?
Additional info: http://avisynth.nl/index.php/FluxSmooth