import torch
from minigpt.model import MiniGPT
from minigpt.tokenizer import Tokenizer, CharTokenizer, BPETokenizer


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> tuple[MiniGPT, Tokenizer]:
    """
    Load a trained MiniGPT model and reconstructed Tokenizer from a saved checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    tok_type = checkpoint.get("tokenizer_type", "char")

    if tok_type == "bpe":
        tokenizer = BPETokenizer(merges=checkpoint["merges"])
    else:
        chars = checkpoint["chars"]
        reconstructed_corpus = "".join(chars)
        tokenizer = CharTokenizer(reconstructed_corpus)

    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str = "\n",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    device: str = "cpu",
) -> str:
    """
    Generate text starting from a prompt using configured sampling.
    """
    model.eval()
    if not prompt:
        prompt = "\n"

    encoded_prompt = tokenizer.encode(prompt)
    idx = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_idx = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    output_tokens = generated_idx[0].tolist()
    return tokenizer.decode(output_tokens)
