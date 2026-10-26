# Android ARM64 native engine

The target is a native UCI/USI/UCCI executable. This repository does not currently build an APK, install an input service or provide a board UI.

## Cross-compile

Use Android NDK r27c (`27.2.12479018`) on Linux and an API-26 compiler for an Android 8+ baseline:

```bash
python tools/materialize.py
export PATH="$STOCKLESS_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH"
make -C build/engine/src -j2 build COMP=ndk ARCH=armv8 \
  CXX=aarch64-linux-android26-clang++ largeboards=yes all=yes nnue=no \
  EXE=stockless-variants
```

Set `STOCKLESS_NDK` to the installed NDK directory beforehand. The build uses PIE and statically linked C++ runtime options supplied by the upstream makefile. Check actual dynamic dependencies with NDK `llvm-readelf` for the produced artifact.

`StocklessMobile=true` is the Android default. Desktop Overdrive remains disabled in mobile mode even when a GUI leaves the option enabled. The policy's sub-ply buyback is quantized deterministically rather than discarded.

## Execute or import

For a development device, an example manual protocol check is:

```bash
adb push build/engine/src/stockless-variants /data/local/tmp/stockless-variants
adb shell chmod 755 /data/local/tmp/stockless-variants
adb shell /data/local/tmp/stockless-variants
```

Then issue `uci`, `isready`, `position startpos`, and `go depth 4`.

A GUI must support importing a native engine, the relevant protocol and the selected board/rules. Standard chess UCI support in a GUI does not imply that its board interface supports Shogi, Janggi, Duck or compound moves. Android application sandbox/execution restrictions depend on the host app's import implementation; copying an executable into arbitrary shared storage is not a universal installation method.

## Evidence boundary

CI cross-compilation plus `AArch64` ELF inspection proves artifact format/build success. It does not prove execution on a device. Device smoke tests, long play, suspend/resume, thread/battery limits and specific GUI integration remain pending until explicitly recorded.
