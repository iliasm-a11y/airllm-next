#pragma once
#include "ggml-backend.h"

// Internal pinned-CPU extra-provider registration, not a public callback ABI.
ggml_backend_buffer_type_t ggml_backend_cpu_metal_q8_bridge_buffer_type();
