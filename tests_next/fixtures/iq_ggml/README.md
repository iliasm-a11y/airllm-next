# Independent GGML IQ decoder oracle

`ggml-common.h` is the complete, unmodified upstream file and
`ggml-iq-dequant.c.inc` contains the three decoder functions copied byte for
byte from `ggml/src/ggml-quants.c`, with one newline between functions.
Both come from [llama.cpp](https://github.com/ggml-org/llama.cpp) commit
`7df8b876b91208a86a6d792e1140b3deec14535b`, under the included MIT license.
`manifest.json` records original and vendored SHA256 hashes.

`oracle.c` adds only a bounded C ABI and a portable IEEE binary16 conversion.
The conversion is checked independently over all 65,536 binary16 bit patterns.
The tests compile this C source with floating point contraction and fast math
disabled. There are no network downloads or model weights in the test.

The oracle never imports the production Python IQ decoder or its tables. It
compares all 256 output values per block, including signed zeros, under a
predefined bitwise policy for finite input scales. Nonfinite scales are tested
separately by value classification; NaN payload preservation is not claimed.

The systematic cases exercise every valid grid index in all sub-block
positions, every IQ3_XXS sign-table index, explicit sign bytes and all scale
nibbles. Additional seeded blocks, FP16 scale boundaries and selective 3D
expert rows cover combinations and ordering. These tests establish codec
decoding fidelity, not real checkpoint generation or M5 performance.

The tests also decode eight real blocks per codec (2,320 bytes total) from the
pinned AJ checkpoint. Their small binary files, exact shard/tensor offsets and
SHA256 provenance are in `research/ronin_aj_iq3xxs/`; all 6,144 real output
values are compared to the C oracle bit for bit. Only bounded HTTP ranges were
used for those samples, and the complete model was not downloaded.
