mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::io;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use crate::model::Llama;

fn main() {
    chat()
    // generate();
    // chats(100,
    //       5.,
    //       10,
    //       1.);
}
fn chats(
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32
){
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut cache = llama.new_cache();
    let mut user;
    let mut input: String;
    let mut output_ids;
    println!("chat start!\n-----------------------------------------------------------------------------");
    loop {
        user = String::new();
        println!("user:");
        std::io::stdin().read_line(&mut user).expect("you need to type some words into console!");
        if user.trim().eq("/exit") {
            println!("chat over!");
            break;
        }
        input = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user
        );
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        (output_ids, cache) = llama.chat(input_ids, max_len, top_p, top_k, temperature, cache);
        println!("assistant:");
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    }
}
fn chat(){
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut kv_cache = llama.new_cache();
    let mut output ;

    while 1==1 {
        println!("请输入需要查询的内容，输入exit退出");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("读取用户输入时出错");

        // 去除字符串首尾的空白字符（换行、空格等）
        let trimmed = input.trim();

        // 判断是否是exit
        if trimmed.eq_ignore_ascii_case("exit") {
            println!("检测到 exit，程序将退出...");
            break;
        }

        // 正常处理输入内容使用chat函数
    //     模板设置
        input = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            trimmed
        );

    //     encoder
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        print!("assistant: ");
        (output,kv_cache) =llama.chat(
            input_ids,
            500,
            0.8,
            30,
            1.,
            kv_cache,
        );
        println!("{}", tokenizer.decode(&output, true).unwrap());

    }
}

fn generate() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(
        input_ids,
        500,
        0.8,
        30,
        1.,
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}
