use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 读取参数加llama传入的config
        // todo!("实现从safetensors文件的模型参数加载");

        let get_tensor: Box<dyn Fn(&str) -> Tensor<f32>> = Box::new(|name: &str| {
            //直接就是参数名称
            let tensor_view = safetensor.tensor(name).unwrap();
            let bytes = tensor_view.data();
            // 转换为 f32 slice（注意字节对齐和安全性）
            let float_data = bytemuck::cast_slice::<u8, f32>(bytes);
            // 创建一个新的 Tensor<f32>
            let shape = tensor_view.shape();
            Tensor::new(Vec::from(float_data), &Vec::from(shape))

        });
        let get_layer_tensor : Box<dyn Fn(&str, u32) -> Vec<Tensor<f32>>> = Box::new(|name:&str,layers:u32|{
            (0..layers).map(|layer|{
                // 使用 format! 宏进行字符串拼接，并将 i 转换为字符串
                let index = format!("model.layers.{}.{}", layer, name);
                get_tensor(&index)
            }
               ).collect::<Vec<Tensor<f32>>>()
        });
        let embedding_table = if config.tie_word_embeddings {
            "lm_head.weight"
        } else {
            "model.embed_tokens.weight"
        };
        LLamaParams {
            // 注意embed与lm可能不一样

            embedding_table: get_tensor(embedding_table),
            rms_att_w:get_layer_tensor("input_layernorm.weight", config.num_hidden_layers as u32),
            wq:get_layer_tensor("self_attn.q_proj.weight", config.num_hidden_layers as u32),
            wk:get_layer_tensor("self_attn.k_proj.weight", config.num_hidden_layers as u32),
            wv:get_layer_tensor("self_attn.v_proj.weight", config.num_hidden_layers as u32),
            wo:get_layer_tensor("self_attn.o_proj.weight", config.num_hidden_layers as u32),
            rms_ffn_w:get_layer_tensor("post_attention_layernorm.weight", config.num_hidden_layers as u32),
            w_up:get_layer_tensor("mlp.up_proj.weight", config.num_hidden_layers as u32),
            w_gate:get_layer_tensor("mlp.gate_proj.weight", config.num_hidden_layers as u32),
            w_down:get_layer_tensor("mlp.down_proj.weight", config.num_hidden_layers as u32),
            rms_out_w:get_tensor("model.norm.weight"),
            lm_head:get_tensor("lm_head.weight"),
        }

    }



}
