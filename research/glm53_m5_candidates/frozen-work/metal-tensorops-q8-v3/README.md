# Metal4 TensorOps v3 — isolated experiment

Source compiled into a Metal4 pipeline in a separate compile-only process.
The first two source versions were refused (temporary slice arguments, then
plain char instead of explicit int8_t). Those failures and exact sources remain
preserved; neither is a hardware numerical failure. This source uses explicit
int8xint8→int32 TensorOps per32-term block and ordered FP32 FMA per scale block.
No weights/quantizer, native guards, buffers or limits are changed.

Host compile, component equality/speed and full GLM qualification remain
separate. Build/runtime OFF by default. No positive result is inherited.
The component suite starts with smaller resident matrices, uses five graph
iterations, and keeps the previous128MiB RSS ceiling. Large M8192/K4096/N64,128
remain excluded after the previous scalar-provider resource refusal.
