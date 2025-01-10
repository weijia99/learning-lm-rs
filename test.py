import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_directory = "models/story"
model_name = "story"

tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

# # 如果已经是 token id 列表，可以直接构造成张量
# # 例如: [0, 1, 2, 3, 4, 5, 6, 7]
inputs = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])  # shape: (batch=1, seq_len=8)

# 让模型返回 hidden_states
# AutoModelForCausalLM -> CausalLMOutputWithCrossAttentions
with torch.no_grad():
    outputs = model(
        input_ids=inputs,
        output_hidden_states=True  # 关键点
    )
print(outputs)
# # outputs 是一个 CausalLMOutputWithCrossAttentions
#   - outputs.logits:   shape (batch, seq_len, vocab_size)
#   - outputs.hidden_states: tuple 长度 = #layers + 1
#       hidden_states[0]: embedding 结果
#       hidden_states[1]: 第1层输出
#       hidden_states[2]: 第2层输出
#       ...
# 最后一个 hidden_states[...] 通常是最后一层输出

hidden_states = outputs.hidden_states

# hidden_states[1] 就是“第 0 层”（即第一个 transformer 层）计算后的张量
# shape 一般是 (batch, seq_len, hidden_dim)
layer_0_output = hidden_states[1]

# print(f"layer_0_output shape = {layer_0_output.shape}")
# print("layer_0_output[0]:", layer_0_output[0][0][10])
# print(outputs)

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# model_directory = "models/story"
#
# model_name = "story"
# tokenizer = AutoTokenizer.from_pretrained(model_directory)
# model = AutoModelForCausalLM.from_pretrained(model_directory)
# outputs_dict = {}
# # inputs = tokenizer("hello, world", return_tensors="pt")
# # print(inputs)
# inputs = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])  # shape: (batch=1, seq_len=8)
#
# def hook_fn(layer_name):
#     def hook(module, input, output):
#         outputs_dict[layer_name] = {
#             "input": input,
#             "output": output
#         }
#     return hook
#
# for name, layer in model.named_modules():
#     print(f"layer name: {name}")
#
#
# for name, layer in model.named_modules():
#     layer_name = f"transformer_layer_{name}"
#     layer.register_forward_hook(hook_fn(layer_name))
#
# # with torch.no_grad():
# #     model(**inputs)
# with torch.no_grad():
#     outputs = model(
#         input_ids=inputs,
#         output_hidden_states=True  # 关键点
#     )
# x = outputs_dict['transformer_layer_model.layers.0']['output'][0]
# print(outputs)
# print(x.shape)
#



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_directory = "models/story"

model_name = "story"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)
outputs_dict = {}
inputs = tokenizer("hello, world", return_tensors="pt")

def hook_fn(layer_name):
    def hook(module, input, output):
        outputs_dict[layer_name] = {
            "input": input,
            "output": output
        }
    return hook

for name, layer in model.named_modules():
    print(f"layer name: {name}")


for name, layer in model.named_modules():
    layer_name = f"transformer_layer_{name}"
    layer.register_forward_hook(hook_fn(layer_name))

with torch.no_grad():
    model(**inputs)

x = outputs_dict['transformer_layer_model.layers.0']['output'][0]

print(outputs)

