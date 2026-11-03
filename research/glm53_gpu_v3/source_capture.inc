static std::string source_weight_json(const ggml_tensor * op) {
    if ((op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) || !op->src[0]) return "null";
    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * base = weight;
    int depth = 0;
    while (base->view_src && depth < 16) { base = base->view_src; ++depth; }
    auto escaped = [](const char * text) {
        std::string result;
        for (const unsigned char c : std::string(text ? text : "")) {
            if (c == '\\' || c == '"') result += '\\';
            if (c >= 32 && c != 127) result += char(c); else result += '?';
        }
        return result;
    };
    auto * buffer=weight->buffer ? weight->buffer : base->buffer;
    std::ostringstream out;
    out << "{\"buffer_usage\":" << (buffer ? int(ggml_backend_buffer_get_usage(buffer)) : -1)
        << ",\"buffer_name\":\"" << escaped(buffer?ggml_backend_buffer_name(buffer):"")
        << "\",\"base_op\":\"" << ggml_op_name(base->op) << "\",\"flags\":" << weight->flags
        << ",\"name\":\"" << escaped(ggml_get_name(weight)) << "\",\"base_name\":\""
        << escaped(ggml_get_name(base)) << "\",\"type\":" << int(weight->type)
        << ",\"bytes\":" << ggml_nbytes(weight) << ",\"alias_depth\":" << depth
        << ",\"alias_resolved\":" << (base->view_src ? "false" : "true") << ",\"dimensions\":[";
    for (int i=0; i<4; ++i) out << (i ? "," : "") << weight->ne[i];
    out << "],\"strides\":[";
    for (int i=0; i<4; ++i) out << (i ? "," : "") << weight->nb[i];
    out << "],\"offset_from_base\":";
    if (weight->data && base->data && uintptr_t(weight->data) >= uintptr_t(base->data))
        out << uintptr_t(weight->data)-uintptr_t(base->data); else out << "null";
    out << "}";
    return out.str();
}
