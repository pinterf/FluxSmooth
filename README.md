# FluxSmooth - pfmod
Avisynth filter for spatio-temporal smoothing of fluctuations

By Ross Thomas <ross@grinfinity.com>

There is no copyright on this code, and there are no conditions
on its distribution or use. Do with it what you will.

- (20190329) v1.x 
  - project moved to github: https://github.com/pinterf/FluxSmooth
  - built using Visual Studio 2017
  - x64 build for Avisynth+
  - Added version resource to DLL
  - Changed to AVS 2.6 plugin interface
  - Removed MMX code, now requires SSE2
  - Single DLL, choosing the optimizations for different CPU instruction sets automatically.
  - Drop all inline assembly, refactor source
  - Rewrite to SIMD intrinsics based on pure C code.
  - added SSE4.1 code path
  - Reports MT Modes for Avisynth+: ???
  - (Added Y, YV16 and YV24 support besides existing YV12)
- (20101130) x64 inline assembler optimized version by Devin Gardner
- (2002-2004) FluxSmooth v1.1b 
  Original version by Ross Thomas
  http://web.archive.org/web/20070225212908/http://bengal.missouri.edu/~kes25c/FluxSmooth-1.1b.zip
  https://forum.doom9.org/showthread.php?t=38296

Installation note: Previous DLL versions named differently (FluxSmoothSSE2.DLL, FluxSmoothSSSE3) should be deleted from your plugin folder.
From 1.3 only a single DLL is supported with automatic CPU optimization selection (SSE2, SSSE3, SSE4.1)

Links
=====
Project: https://github.com/pinterf/FluxSmooth
Forum: ?
Additional info: http://avisynth.nl/index.php/FluxSmooth