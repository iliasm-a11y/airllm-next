// Diagnostic only: hash completed cache writes with <=1MiB readback scratch.
static const ggml_tensor * state_target(const ggml_tensor * node) {
    if (node->op != GGML_OP_CPY && node->op != GGML_OP_SET_ROWS) return nullptr;
    const ggml_tensor * candidates[] = {node, node->op == GGML_OP_CPY ? node->src[1] : node->src[0]};
    for (const auto * candidate : candidates) {
        if (!candidate) continue;
        for (int depth=0; candidate && depth<=16; ++depth) {
            if (std::string(ggml_get_name(candidate)).rfind("cache_",0)==0) return candidate;
            candidate=candidate->view_src;
        }
    }
    return nullptr;
}
static std::string state_event(ggml_tensor * node, int step) {
    const ggml_tensor * target=state_target(node);
    if (!target) return "";
    std::ostringstream out;
    const size_t bytes=ggml_nbytes(node);
    out << "{\"step\":" << step << ",\"target\":\"" << ggml_get_name(target)
        << "\",\"op\":\"" << ggml_op_name(node->op) << "\",\"type\":" << int(node->type)
        << ",\"bytes\":" << bytes << ",\"shape\":[";
    for (int i=0;i<4;++i) out << (i?",":"") << node->ne[i];
    out << "]";
    const size_t element=node->type==GGML_TYPE_F32?4:2;
    size_t logical=element;
    bool valid_type=node->type==GGML_TYPE_F32 || node->type==GGML_TYPE_F16 || node->type==GGML_TYPE_BF16;
    bool bounded=true;
    for(int i=0;i<4;++i) {
        if(node->ne[i]<0 || (node->ne[i] && logical>(512ull*1024*1024)/size_t(node->ne[i]))) {bounded=false;break;}
        logical*=size_t(node->ne[i]);
    }
    out << ",\"logical_bytes\":" << logical;
    if (!valid_type || !bounded || bytes>512ull*1024*1024 || (!ggml_is_contiguous(node) && logical/element>2*1024*1024)) {
        out << ",\"status\":\"skipped\",\"reason\":\"unsupported_size_or_type\"}";
        return out.str();
    }
    std::vector<unsigned char> data(std::max<size_t>(element,std::min<size_t>(logical,1024*1024)));
    CC_SHA256_CTX hash; CC_SHA256_Init(&hash); bool finite=true;
    auto consume=[&](size_t count) {
        if(!count) return;
        CC_SHA256_Update(&hash,data.data(),(CC_LONG)count);
        if(node->type==GGML_TYPE_F32) {
            for(size_t i=0;i<count;i+=4){float v;std::memcpy(&v,data.data()+i,4);finite=finite&&std::isfinite(v);}
        } else {
            const uint16_t exponent=node->type==GGML_TYPE_F16?0x7c00:0x7f80;
            for(size_t i=0;i<count;i+=2){uint16_t v;std::memcpy(&v,data.data()+i,2);finite=finite&&((v&exponent)!=exponent);}
        }
    };
    if(ggml_is_contiguous(node)) {
        for(size_t offset=0;offset<logical;offset+=data.size()) {
            const size_t count=std::min(data.size(),logical-offset);
            ggml_backend_tensor_get(node,data.data(),offset,count);consume(count);
        }
    } else {
        size_t filled=0;
        for(int64_t i3=0;i3<node->ne[3];++i3) for(int64_t i2=0;i2<node->ne[2];++i2)
        for(int64_t i1=0;i1<node->ne[1];++i1) for(int64_t i0=0;i0<node->ne[0];++i0) {
            const size_t offset=size_t(i3)*node->nb[3]+size_t(i2)*node->nb[2]+size_t(i1)*node->nb[1]+size_t(i0)*node->nb[0];
            if(offset>bytes || element>bytes-offset) throw std::runtime_error("State stride outside tensor");
            ggml_backend_tensor_get(node,data.data()+filled,offset,element);filled+=element;
            if(filled==data.size()){consume(filled);filled=0;}
        }
        consume(filled);
    }
    unsigned char result[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(result,&hash);
    out << ",\"status\":\"hashed\",\"finite\":" << (finite?"true":"false") << ",\"sha256\":\"";
    for(unsigned char v:result) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(v);
    out << "\",\"scratch_bytes\":" << std::dec << data.size() << "}";
    return out.str();
}
