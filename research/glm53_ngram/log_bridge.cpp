// The original n-gram translation unit only uses debug logging. Keep it quiet
// without linking the server's terminal/UI/JSON logging dependencies.
#include "log.h"
int common_log_get_verbosity_thold() { return -1; }
common_log * common_log_main() { return nullptr; }
void common_log_add(common_log *, ggml_log_level, const char *, ...) {}
