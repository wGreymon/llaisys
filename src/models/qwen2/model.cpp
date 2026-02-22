#include "model.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../../ops/add/op.hpp"
#include "../../device/runtime_api.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>

namespace llaisys::models::qwen2 {

Model::Model(const ModelMeta& meta, llaisysDeviceType_t device_type, int device_id)
    : meta_(meta), device_type_(device_type), device_id_(device_id), cache_len_(0) {
    
    // 初始化 KV Cache
    k_cache_.resize(meta_.nlayer);
    v_cache_.resize(meta_.nlayer);
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        k_cache_[i] = Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, 
                                     meta_.dtype, device_type_, device_id_);
        v_cache_[i] = Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, 
                                     meta_.dtype, device_type_, device_id_);
    }
    
    // 初始化权重数组
    weights_.attn_norm_w.resize(meta_.nlayer);
    weights_.attn_q_w.resize(meta_.nlayer);
    weights_.attn_q_b.resize(meta_.nlayer);
    weights_.attn_k_w.resize(meta_.nlayer);
    weights_.attn_k_b.resize(meta_.nlayer);
    weights_.attn_v_w.resize(meta_.nlayer);
    weights_.attn_v_b.resize(meta_.nlayer);
    weights_.attn_o_w.resize(meta_.nlayer);
    weights_.mlp_norm_w.resize(meta_.nlayer);
    weights_.mlp_gate_w.resize(meta_.nlayer);
    weights_.mlp_up_w.resize(meta_.nlayer);
    weights_.mlp_down_w.resize(meta_.nlayer);
    
    // 创建 dummy bias tensors（全零，用于没有 bias 的层）
    dummy_bias_hs_ = Tensor::create({meta_.hs}, meta_.dtype, device_type_, device_id_);
    dummy_bias_di_ = Tensor::create({meta_.di}, meta_.dtype, device_type_, device_id_);
    dummy_bias_q_ = Tensor::create({meta_.nh * meta_.dh}, meta_.dtype, device_type_, device_id_);
    dummy_bias_kv_ = Tensor::create({meta_.nkvh * meta_.dh}, meta_.dtype, device_type_, device_id_);
    dummy_bias_voc_ = Tensor::create({meta_.voc}, meta_.dtype, device_type_, device_id_);

    // dummy bias 必须显式清零，否则会把未初始化内存当作 bias 加进去，导致输出完全错误
    auto zero_tensor = [](const tensor_t &t) {
        std::vector<std::byte> zeros(t->numel() * t->elementSize(), std::byte{0});
        t->load(zeros.data());
    };
    zero_tensor(dummy_bias_hs_);
    zero_tensor(dummy_bias_di_);
    zero_tensor(dummy_bias_q_);
    zero_tensor(dummy_bias_kv_);
    zero_tensor(dummy_bias_voc_);
}

Model::~Model() {
    // 智能指针会自动管理内存
}

void Model::reset_cache() {
    cache_len_ = 0;
}

void Model::update_kv_cache(size_t layer_idx, tensor_t k_new, tensor_t v_new, size_t seqlen, size_t old_len) {
    // 将新的 K 和 V 追加到 cache
    // k_new: [seqlen, nkvh, dh]
    // v_new: [seqlen, nkvh, dh]

    // old_len 必须是"本次 forward 开始前"的 cache 长度。
    // 注意：cache_len_ 是全局序列长度，不应在每一层里自增。
    ASSERT(old_len == cache_len_, "update_kv_cache: old_len must equal cache_len_");
    size_t new_len = old_len + seqlen;
    
    // 从 cache 中切片出需要更新的部分
    tensor_t k_slice = k_cache_[layer_idx]->slice(0, old_len, new_len);
    tensor_t v_slice = v_cache_[layer_idx]->slice(0, old_len, new_len);
    
    // 复制新计算的 K 和 V 到 cache
    // 使用运行时 API 的内存拷贝，支持跨设备
    llaisys::core::context().setDevice(device_type_, device_id_);
    const LlaisysRuntimeAPI *api = llaisys::core::context().runtime().api();
    
    // 使用 tensor 的 numel 和 elementSize 计算正确的字节数
    size_t k_size = k_new->numel() * k_new->elementSize();
    size_t v_size = v_new->numel() * v_new->elementSize();
    
    // 确保 k_new 和 v_new 是连续的
    ASSERT(k_new->isContiguous() && v_new->isContiguous(), 
           "update_kv_cache: k_new and v_new must be contiguous");
    ASSERT(k_slice->numel() == k_new->numel() && v_slice->numel() == v_new->numel(),
           "update_kv_cache: slice size must match new tensor size");
    
    // cache/new 都在同一设备上，使用 D2D
    api->memcpy_sync(k_slice->data(), k_new->data(), k_size, LLAISYS_MEMCPY_D2D);
    api->memcpy_sync(v_slice->data(), v_new->data(), v_size, LLAISYS_MEMCPY_D2D);
}

void Model::forward_layer(size_t layer_idx, tensor_t& x, size_t seqlen, size_t total_len, tensor_t pos_ids_q) {
    // 设置设备上下文
    llaisys::core::context().setDevice(device_type_, device_id_);
    
    // 1. Pre-attention norm
    x_norm_ = Tensor::create({seqlen, meta_.hs}, meta_.dtype, device_type_, device_id_);
    ops::rms_norm(x_norm_, x, weights_.attn_norm_w[layer_idx], meta_.epsilon);
    
    // 2. Attention
    // 2.1 计算 Q, K, V
    // x_norm: [seqlen, hs]
    // Q weight: [nh * dh, hs], output: [seqlen, nh * dh]
    // K weight: [nkvh * dh, hs], output: [seqlen, nkvh * dh]
    // V weight: [nkvh * dh, hs], output: [seqlen, nkvh * dh]
    
    tensor_t q_flat = Tensor::create({seqlen, meta_.nh * meta_.dh}, meta_.dtype, device_type_, device_id_);
    tensor_t k_flat = Tensor::create({seqlen, meta_.nkvh * meta_.dh}, meta_.dtype, device_type_, device_id_);
    tensor_t v_flat = Tensor::create({seqlen, meta_.nkvh * meta_.dh}, meta_.dtype, device_type_, device_id_);
    
    // 处理可能为空的 bias：如果不存在，使用 dummy bias
    tensor_t q_bias = (weights_.attn_q_b[layer_idx] && weights_.attn_q_b[layer_idx]->numel() > 0) ? 
                      weights_.attn_q_b[layer_idx] : dummy_bias_q_;
    tensor_t k_bias = (weights_.attn_k_b[layer_idx] && weights_.attn_k_b[layer_idx]->numel() > 0) ? 
                      weights_.attn_k_b[layer_idx] : dummy_bias_kv_;
    tensor_t v_bias = (weights_.attn_v_b[layer_idx] && weights_.attn_v_b[layer_idx]->numel() > 0) ? 
                      weights_.attn_v_b[layer_idx] : dummy_bias_kv_;
    
    ops::linear(q_flat, x_norm_, weights_.attn_q_w[layer_idx], q_bias);
    ops::linear(k_flat, x_norm_, weights_.attn_k_w[layer_idx], k_bias);
    ops::linear(v_flat, x_norm_, weights_.attn_v_w[layer_idx], v_bias);
    
    // Reshape: [seqlen, nh * dh] -> [seqlen, nh, dh]
    q_ = q_flat->view({seqlen, meta_.nh, meta_.dh});
    k_ = k_flat->view({seqlen, meta_.nkvh, meta_.dh});
    v_ = v_flat->view({seqlen, meta_.nkvh, meta_.dh});
    
    // 2.2 RoPE（只处理本轮新增 token）
    tensor_t q_rope = Tensor::create({seqlen, meta_.nh, meta_.dh}, meta_.dtype, device_type_, device_id_);
    tensor_t k_rope_new = Tensor::create({seqlen, meta_.nkvh, meta_.dh}, meta_.dtype, device_type_, device_id_);
    ops::rope(k_rope_new, k_, pos_ids_q, meta_.theta);
    ops::rope(q_rope, q_, pos_ids_q, meta_.theta);

    // 2.3 更新 KV Cache（K 使用 RoPE 后结果，V 保持原值）
    size_t old_len = total_len - seqlen;
    update_kv_cache(layer_idx, k_rope_new, v_, seqlen, old_len);

    // 2.4 准备完整的 K 和 V（包含 cache）
    k_full_ = k_cache_[layer_idx]->slice(0, 0, total_len);
    v_full_ = v_cache_[layer_idx]->slice(0, 0, total_len);
    
    // 2.5 Self-attention
    attn_out_ = Tensor::create({seqlen, meta_.nh, meta_.dh}, meta_.dtype, device_type_, device_id_);
    float scale = 1.0f / std::sqrt(static_cast<float>(meta_.dh));
    ops::self_attention(attn_out_, q_rope, k_full_, v_full_, scale);
    
    // 2.6 Attention output projection
    // attn_out: [seqlen, nh, dh] -> [seqlen, nh * dh]
    tensor_t attn_out_flat = attn_out_->view({seqlen, meta_.nh * meta_.dh});
    attn_proj_out_ = Tensor::create({seqlen, meta_.hs}, meta_.dtype, device_type_, device_id_);
    ops::linear(attn_proj_out_, attn_out_flat, weights_.attn_o_w[layer_idx], dummy_bias_hs_);
    
    // 2.7 残差连接
    tensor_t x_attn = Tensor::create({seqlen, meta_.hs}, meta_.dtype, device_type_, device_id_);
    ops::add(x_attn, x, attn_proj_out_);
    x = x_attn;
    
    // 3. Post-attention norm
    x_norm_ = Tensor::create({seqlen, meta_.hs}, meta_.dtype, device_type_, device_id_);
    ops::rms_norm(x_norm_, x, weights_.mlp_norm_w[layer_idx], meta_.epsilon);
    
    // 4. MLP
    // x_norm: [seqlen, hs]
    gate_ = Tensor::create({seqlen, meta_.di}, meta_.dtype, device_type_, device_id_);
    up_ = Tensor::create({seqlen, meta_.di}, meta_.dtype, device_type_, device_id_);
    
    ops::linear(gate_, x_norm_, weights_.mlp_gate_w[layer_idx], dummy_bias_di_);
    ops::linear(up_, x_norm_, weights_.mlp_up_w[layer_idx], dummy_bias_di_);
    
    tensor_t swiglu_out = Tensor::create({seqlen, meta_.di}, meta_.dtype, device_type_, device_id_);
    ops::swiglu(swiglu_out, gate_, up_);
    
    mlp_out_ = Tensor::create({seqlen, meta_.hs}, meta_.dtype, device_type_, device_id_);
    ops::linear(mlp_out_, swiglu_out, weights_.mlp_down_w[layer_idx], dummy_bias_hs_);
    
    // 5. 残差连接
    tensor_t x_mlp = Tensor::create({seqlen, meta_.hs}, meta_.dtype, device_type_, device_id_);
    ops::add(x_mlp, x, mlp_out_);
    x = x_mlp;
}

tensor_t Model::forward(tensor_t input_ids, size_t seqlen, size_t total_len) {
    // 设置设备上下文
    llaisys::core::context().setDevice(device_type_, device_id_);
    
    // 1. Embedding
    x_ = Tensor::create({seqlen, meta_.hs}, meta_.dtype, device_type_, device_id_);
    ops::embedding(x_, input_ids, weights_.in_embed);
    
    // 2. 本轮所有层复用同一份 pos_ids（避免每层重复构造与拷贝）
    tensor_t pos_ids_q = Tensor::create({seqlen}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    std::vector<int64_t> pos_ids_q_host(seqlen);
    size_t start_pos = total_len - seqlen;
    for (size_t i = 0; i < seqlen; ++i) {
        pos_ids_q_host[i] = static_cast<int64_t>(start_pos + i);
    }
    pos_ids_q->load(pos_ids_q_host.data());

    // 3. Transformer layers
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        forward_layer(i, x_, seqlen, total_len, pos_ids_q);
    }

    // 4. Output norm
    x_norm_ = Tensor::create({seqlen, meta_.hs}, meta_.dtype, device_type_, device_id_);
    ops::rms_norm(x_norm_, x_, weights_.out_norm_w, meta_.epsilon);

    // 5. Output projection (logits)
    logits_ = Tensor::create({seqlen, meta_.voc}, meta_.dtype, device_type_, device_id_);
    // out_embed 应该是 [voc, hs]，linear 计算 Y = X W^T，所以 Y = [seqlen, voc]
    ops::linear(logits_, x_norm_, weights_.out_embed, dummy_bias_voc_);
    
    return logits_;
}

int64_t Model::infer(int64_t* token_ids, size_t ntoken) {
    // 设置设备上下文
    llaisys::core::context().setDevice(device_type_, device_id_);
    
    // 创建输入张量
    tensor_t input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    
    // 使用 load 方法加载数据（支持跨设备）
    // 先将数据复制到临时缓冲区
    std::vector<int64_t> host_data(token_ids, token_ids + ntoken);
    input_ids->load(host_data.data());
    
    // 确定序列长度
    size_t seqlen = ntoken;
    size_t total_len = cache_len_ + seqlen;
    
    // 前向传播
    tensor_t logits = forward(input_ids, seqlen, total_len);

    // 本轮 forward 已把每层 K/V 写入 cache 的 [cache_len_, total_len) 区间
    cache_len_ = total_len;
    
    // 获取最后一个 token 的 logits
    tensor_t last_logits = logits->slice(0, seqlen - 1, seqlen);
    last_logits = last_logits->view({meta_.voc});
    
    // Argmax
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    tensor_t max_val = Tensor::create({1}, meta_.dtype, device_type_, device_id_);
    ops::argmax(max_idx, max_val, last_logits);
    
    // 将结果从设备拷贝回主机
    std::vector<int64_t> host_result(1);
    llaisys::core::context().runtime().api()->memcpy_sync(
        host_result.data(), max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    
    return host_result[0];
}

} // namespace llaisys::models::qwen2
