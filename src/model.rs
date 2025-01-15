use std::fs::File;
use std::vec;
extern crate bytemuck;

use bytemuck::cast_slice;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors};
use std::path::{Path, PathBuf};
use rand::Rng;
use tokenizers::Tokenizer;
use crate::operators::{dot, gather, masked_softmax, matmul_transb, rms_norm, swiglu, random_sample};

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        // 通过读取config来获取llama的架构
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        // 读取参数model的safetensor
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        // 主要是完善读取参数到llm
        let names = safetensor.names();
        for name in names {
            let tensor_view = safetensor.tensor("model.layers.1.mlp.up_proj.weight").unwrap();
            println!("Tensor name: {}", name);
            println!("  dtype: {:?}", tensor_view.dtype());
            println!("  shape: {:?}", tensor_view.shape());
            // 如果类型是 F32，可以将其加载到一个 Vec<f32> 中
            if let Dtype::F32 = tensor_view.dtype() {
                // 获取底层字节切片
                let bytes = tensor_view.data();
                // 转换为 f32 slice（注意字节对齐和安全性）
                let float_data = bytemuck::cast_slice::<u8, f32>(bytes);

                // 打印部分数据
                println!("  First 5 elements: {:?}", &float_data[..5.min(float_data.len())]);


            }
        }
        let params = LLamaParams::from_safetensors(&safetensor, &config);

            Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv
            );
            let a1 = hidden_states.data();
            // println!("{:?}",a1);


            // todo!("self_attention(...)");
            // todo!("down_proj matmul and add residual");

            // todo!("mlp(...)");
            OP::matmul_transb(&mut residual, 1., &hidden_states, &self.params.wo[layer], 1.);
            let b =residual.data();
            // println!("{:?}",b);
            // println!("{:?}",a);
            //todo!("mlp(...)");
            let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps
            );
            let a =residual.data();
            // let a = self.params.w_up[1].data();
            // println!("{:?}",a);
            // let a1 = self.params.w_down[1].data();
            // println!("{:?}",a1);
            // let a2 = self.params.w_gate[1].data();
            // println!("{:?}",a2);
            // let a3 = self.params.rms_ffn_w[1].data();
            // println!("{:?}",a3);

        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        // 返回一个词表的概率，hidden是（seq_len,d)的矩阵，最后一行
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);
        // print!("{:?}",logits.data());
        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result: Vec<u32>  = Vec::new();
        let mut kv_cache = Llama::new_cache(self);
        // todo!("实现文本生成");
        // 根据top_p,top_k来进行采样生成
        // 1.是否在max_len之前停止
        // 2.根据top_p和top_k来进行采样
        let mut input = Tensor::<u32>::new(Vec::from(token_ids), &vec![1, token_ids.len()]);
        while result.len() < max_len{
            // 1.根据token_ids生成下一个token
            let output = self.forward(&input, &mut kv_cache);
            let next_token =random_sample(&output,  top_p, top_k, temperature);
            if next_token == self.eos_token_id{
                break;
            }
            input=Tensor::<u32>::new(vec![next_token], &vec![1,1]);
            result.push(next_token);

        }
        result

    }
    pub fn chat( &self,
                 token_ids: &[u32],
                 max_len: usize,
                 top_p: f32,
                 top_k: u32,
                 temperature: f32,
                 mut cache: KVCache<f32>
    ) -> (Vec<u32>,KVCache<f32>){
       // todo!()
    //     基于kvcache的加速，通过服用上一轮的无须使用
        let n =token_ids.len();
        let mut result: Vec<u32>  = Vec::new();
        // todo!("实现文本生成");
        // 根据top_p,top_k来进行采样生成
        // 1.是否在max_len之前停止
        // 2.根据top_p和top_k来进行采样
        let mut input = Tensor::<u32>::new(Vec::from(token_ids), &vec![1, token_ids.len()]);
        while result.len() < max_len{
            // 1.根据token_ids生成下一个token
            let output = self.forward(&input, &mut cache);
            let next_token =random_sample(&output,  top_p, top_k, temperature);
            if next_token == self.eos_token_id{
                break;
            }
            input=Tensor::<u32>::new(vec![next_token], &vec![1,1]);
            result.push(next_token);

        }

        (result,cache)


    }
}

fn self_attention1(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // step 1 ,socre = Q @ K.T / sqrt(dim)
    let sqrt_dim = (dqkv as f32).sqrt();
    let scores = unsafe{att_scores.data_mut()};
    for i in 0..seq_len{
        for j in 0..total_seq_len{
            for m in 0..n_kv_h{
                for n in 0..n_groups{
                    let q_start = (m * n_groups + n) * dqkv + i * n_groups * n_kv_h * dqkv;
                    let q_= q.slice(q_start, &vec![dqkv, 1]);
                    let k_start = m * dqkv + j *  n_kv_h * dqkv;
                    let k_: Tensor<f32> = k.slice(k_start, &vec![dqkv, 1]);
                    let value = OP::dot(&q_, &k_) / sqrt_dim;
                    scores[m * n_groups * seq_len * total_seq_len
                        + n * seq_len * total_seq_len
                        + i * total_seq_len
                        + j]
                        = value;
                }
            }
        }
    }
    // step 2, attn = softmax(score)
    OP::masked_softmax(att_scores);
    // step 3, x = attn @ V
    // attn (n_kv_head, n_group, seq_len, total_seq_len) --> n_kv_head * n_group * (seq_len, total_seq_len)
    // attn_slice = (seq_len, total_seq_len)
    // v (total_seq_len, n_kv_head * head_size) --> v.T (n_kv_head * head_size, total_seq_len)
    // v.T (n_kv_head * head_size, total_seq_len) --> n_kv_head * (head_size, total_seq_len)
    // v.T_slice = (head_size, total_seq_len)
    // matmul_transb (attn_slice , v.T_slice) = (seq_len, head_size)
    // hidden_state = attn @ V = n_kv_head * n_group * (seq_len, head_size) = (seq_len, n_kv_head * n_group * head_size)
    let v_data = v.data();
    let hidden_len = n_kv_h * n_groups * dqkv;
    let hidden = unsafe{hidden_states.data_mut()};
    for i in 0..n_kv_h{
        for j in 0..n_groups{
            let attn_start = (i * n_groups + j) * seq_len * total_seq_len;
            let attn_slice = &att_scores.slice(attn_start, &vec![seq_len, total_seq_len]);
            // reverse v
            let mut v_rev = vec![0.; dqkv * total_seq_len];
            for m in 0..dqkv{
                for n in 0..total_seq_len{
                    v_rev[m * total_seq_len + n] = v_data[n * dqkv * n_kv_h + i * dqkv + m];
                }
            }
            let v_rev_tensor = Tensor::new(v_rev, &vec![dqkv, total_seq_len]);
            // matmul_transb result
            let mut mat_result = Tensor::default(&vec![seq_len, dqkv]);
            OP::matmul_transb(&mut mat_result, 0., &attn_slice, &v_rev_tensor, 1.);
            // hidden_state
            let mat_data = mat_result.data();
            for row in 0..seq_len{
                for col in 0..dqkv{
                    hidden[hidden_len * row + (i * n_groups + j) * dqkv + col] = mat_data[row * dqkv + col];
                }
            }
        }
    }
    // print!("{:?}", hidden_states.data());


}
fn reshape_tensor(
    q: & [f32],              // 原来的 2D 数组: shape=(seq, n_kv_h*n_groups*dqkv)
    seq: usize,
    n_kv_h: usize,
    n_groups: usize,
    dqkv: usize,
) ->  Vec<f32> {
    // 新数组大小 = n_kv_h * n_groups * seq * dqkv
    let mut q_new = vec![0f32; n_kv_h * n_groups * seq * dqkv];

    for i in 0..n_kv_h {
        for j in 0..n_groups {
            for s in 0..seq {
                for d in 0..dqkv {
                    // ------- 计算原数组里对应的 old_idx -------
                    // x = i*(n_groups*dqkv) + j*dqkv + d
                    let x = i * (n_groups * dqkv) + j * dqkv + d;
                    // old_idx = s*(n_kv_h*n_groups*dqkv) + x
                    let old_idx = s * (n_kv_h * n_groups * dqkv) + x;

                    // ------- 计算新数组里的 new_idx -------
                    // new_idx = i*(n_groups*seq*dqkv) + j*(seq*dqkv) + s*dqkv + d
                    let new_idx = i * (n_groups * seq * dqkv)
                        + j * (seq * dqkv)
                        + s * dqkv
                        + d;

                    q_new[new_idx] = q[old_idx];
                }
            }
        }
    }
    q_new

}

fn reshape_tensor_2d(
    q: & [f32],              // 原来的 4D 数组: shape=( n_kv_h,n_groups,seq,dqkv)

    n_kv_h: usize,
    n_groups: usize,
    seq: usize,
    dqkv: usize,
) ->  Vec<f32> {
    // 新数组大小 = n_kv_h * n_groups * seq * dqkv
    let mut q_new = vec![0f32; n_kv_h * n_groups * seq * dqkv];

    for seq_idx in 0..seq {
        for n_idx in 0..n_kv_h {
            for g_idx in 0..n_groups {
                for dqkv_idx in 0..dqkv {
                    // 计算原始数据的索引
                    let original_index = n_idx * (n_groups * seq * dqkv)
                        + g_idx * (seq * dqkv)
                        + seq_idx * dqkv
                        + dqkv_idx;

                    // 计算目标数据的索引
                    let target_index = seq_idx * (n_kv_h * n_groups * dqkv)
                        + n_idx * (n_groups * dqkv)
                        + g_idx * dqkv
                        + dqkv_idx;

                    // 复制数据
                    q_new[target_index] = q[original_index];
                }
            }
        }
    }


    q_new

}
fn reshape_vtensor(
    seq: usize,
    n: usize,
    dqkv: usize,
    original_data: &[f32], // 原始数据，形状为 (seq, n * dqkv)
) -> Vec<f32> {
    let mut target_data = vec![0.0; n * 1 * dqkv * seq]; // 目标数据，形状为 (n, 1, dqkv, seq)

    for n_idx in 0..n {
        for dqkv_idx in 0..dqkv {
            for seq_idx in 0..seq {
                // 计算原始数据的索引
                let original_index = seq_idx * (n * dqkv) + n_idx * dqkv + dqkv_idx;

                // 计算目标数据的索引
                let target_index = n_idx * (1 * dqkv * seq) + 0 * (dqkv * seq) + dqkv_idx * seq + seq_idx;

                // 复制数据
                target_data[target_index] = original_data[original_index];
            }
        }
    }

    target_data
}
fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // todo!("Implement self_attention");
//     实现self-attention
    let _q =q.data();
    let q_new =reshape_tensor(_q, seq_len, n_kv_h, n_groups, dqkv);

    let _k = k.data();

    let k_new =reshape_tensor(_k, total_seq_len, n_kv_h, 1, dqkv);
    let _v = v.data();

    let v_new =reshape_vtensor( total_seq_len, n_kv_h,  dqkv,_v);

    // 第一步qk
    {
    let _attn = unsafe{att_scores.data_mut()};
    matmul_with_batch1(n_kv_h, n_groups, seq_len, total_seq_len, &dqkv, &q_new, &k_new, _attn);


        _attn.iter_mut().for_each(|val| {
            *val /= (dqkv as f32).sqrt();
        });




    }// 第二步进行掩码

    masked_softmax(att_scores);
    let _attn = unsafe{att_scores.data_mut()};
    let attn_new = _attn.to_vec();
    // 第三步进行qv
    let mut att_v =
        Tensor::<f32>::default(&vec![n_kv_h, n_groups, seq_len, dqkv]);
   let _att_v = unsafe { att_v.data_mut()};
    matmul_with_batch1(n_kv_h,n_groups,seq_len,dqkv,&total_seq_len,&attn_new,&v_new,_att_v);

    let out = reshape_tensor_2d(_att_v,n_kv_h,n_groups,seq_len,dqkv);

    // 这里需要把（n,g,seq,dqkv回复成为（seq,n*g*dqvk)
    let n =hidden_states.size();

    let _hidden = unsafe{hidden_states.data_mut()};
    for i in (0..n) {
        _hidden[i]=out[i];
    }

}

fn matmul_with_batch1(n_kv_h: usize, n_groups: usize, seq_len: usize, total_seq_len: usize, dqkv: &usize, _q: & Vec<f32>, _k: & Vec<f32>, _attn: &mut [f32]) {
//     计算有多少个batch
    let m = _q.len()/(n_kv_h * seq_len*dqkv);
    let n = _k.len()/(n_kv_h * total_seq_len*dqkv);
    for i in 0..n_kv_h{
        for j in 0..n{
            for k in n_groups*(j)..(n_groups*(j+1)){
                let k_offset = (i * n+j)*(total_seq_len*dqkv);
                let q_offset = (i*m+k)*(seq_len*dqkv);
                let attn_offset =(i*m+k)*(total_seq_len*seq_len);
                for p in 0..seq_len{
                    for q in 0..total_seq_len{
                        let qx = Tensor::new(
                            _q[(q_offset + p * dqkv)..(q_offset + (p + 1) * dqkv)].to_vec(),
                            &vec![1, *dqkv]
                        );
                        // 这里应该还是不变的，扩大4被来计算
                        let ky = Tensor::new(
                            _k[(k_offset + q * dqkv)..(k_offset + (q + 1) * dqkv)].to_vec(),
                            &vec![1, *dqkv]
                        );
                        let plus = dot(&qx, &ky);
                        _attn[p *  total_seq_len + q + attn_offset] = plus;
                    }
                }
            }
        }
    }
}


fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // todo!("Implement mlp");

    rms_norm(hidden_states,residual, rms_w, eps);
  

    matmul_transb(gate,0.0,hidden_states, w_gate, 1.0);
   

    matmul_transb(up,0.0,hidden_states, w_up, 1.0);
    swiglu(up,gate);
    

    let mut act =Tensor::default(residual.shape());
    matmul_transb(&mut act,0.0,up,w_down,1.0);
    let n =act.size();
    let _res = unsafe{residual.data_mut()};
    

    for i in (0..n){
        _res[i]+=act.data()[i];
    }


}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );
    println!("{:?}", residual);
    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
#[test]
pub fn test_forward() {
//     直接加载load进行对比参数
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    let input = Tensor::<u32>::new(vec![0, 1, 2, 3, 4, 5, 6, 7], &vec![8]);
    let mut cache = model.new_cache();
    let output = model.forward(&input, &mut cache).print();
}
