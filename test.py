from transformers import AutoModelForCausalLM
import torch

def download():
    for model in ['']:
        AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2'
        )

    for model in ['']:
        AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2'
        )
