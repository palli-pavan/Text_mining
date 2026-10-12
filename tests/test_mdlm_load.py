import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def main():
    print("Testing MDLM checkpoint loading...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print(f"GPT-2 tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    model_name = 'kuleshov-group/mdlm-owt'
    print(f"Loading model {model_name} (trust_remote_code=True)...")
    
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model.eval()
    model.to(device)
    print("Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if device == "cuda":
        print(f"VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        print(f"VRAM Reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")

    # Quick forward pass test
    # MDLM vocab size is 50258, mask index is 50257
    mask_token_id = 50257
    input_ids = torch.tensor([[tokenizer.bos_token_id or 50256, 123, 456, mask_token_id, mask_token_id]], device=device)
    timesteps = torch.tensor([[0.5]], device=device) # or scalar/1d
    print(f"Running test forward pass with shape {input_ids.shape}...")
    with torch.no_grad():
        out = model(input_ids=input_ids, timesteps=timesteps)
        logits = out.logits if hasattr(out, 'logits') else (out[0] if isinstance(out, tuple) else out)
        print(f"Output logits shape: {logits.shape}")
        pred_tokens = logits.argmax(dim=-1)
        print(f"Argmax tokens: {pred_tokens}")
        print("Test forward pass succeeded!")

if __name__ == "__main__":
    main()
