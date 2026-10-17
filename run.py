import argparse
import os
import torch

from minigpt.config import GPTConfig, TrainConfig
from minigpt.dataset import prepare_data
from minigpt.generate import generate_text, load_model_from_checkpoint
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharTokenizer, BPETokenizer
from minigpt.train import train


def train_command(args):
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found at: {args.data}")

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded corpus from {args.data} ({len(text):,} characters).")
    if args.tokenizer == "bpe":
        print(f"Training BPETokenizer with target vocabulary size {args.bpe_vocab_size}...")
        tokenizer = BPETokenizer.train_from_text(text, target_vocab_size=args.bpe_vocab_size)
    else:
        tokenizer = CharTokenizer(text)

    print(f"Constructed {args.tokenizer.upper()} vocabulary of size {tokenizer.vocab_size}.")

    train_data, val_data = prepare_data(text, tokenizer, train_split=args.train_split)

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        use_sdpa=args.use_sdpa,
    )

    device = (
        "mps"
        if torch.backends.mps.is_available() and not args.cpu
        else ("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    )

    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        device=device,
    )

    model = MiniGPT(model_config)

    train(
        model=model,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        model_config=model_config,
        train_config=train_config,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Generate a brief sample right after training
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    if os.path.exists(checkpoint_path):
        best_model, _ = load_model_from_checkpoint(checkpoint_path, device=device)
        print("\n--- Sample Generation ---")
        prompt = "First Citizen:\n"
        sample = generate_text(
            best_model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.8,
            device=device,
        )
        print(sample)
        print("-------------------------\n")


def generate_command(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    device = (
        "mps"
        if torch.backends.mps.is_available() and not args.cpu
        else ("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    )

    model, tokenizer = load_model_from_checkpoint(args.checkpoint, device=device)
    print(f"Loaded checkpoint from {args.checkpoint} on device: {device}\n")

    # Allow CLI users to use literal \n and \t escape sequences
    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print(output)


def main():
    parser = argparse.ArgumentParser(
        description="MiniGPT: Train and generate text using custom Transformer LM."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Train subparser
    train_parser = subparsers.add_parser("train", help="Train MiniGPT on text corpus.")
    train_parser.add_argument("--data", type=str, default="data/input.txt", help="Path to text corpus.")
    train_parser.add_argument("--block_size", type=int, default=64, help="Context length / block size.")
    train_parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers.")
    train_parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads.")
    train_parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension.")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    train_parser.add_argument("--bias", action="store_true", help="Enable bias in layers.")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    train_parser.add_argument("--max_iters", type=int, default=500, help="Maximum training iterations.")
    train_parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval.")
    train_parser.add_argument("--eval_iters", type=int, default=20, help="Number of evaluation batches.")
    train_parser.add_argument("--train_split", type=float, default=0.9, help="Train split fraction.")
    train_parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"], help="Tokenizer type ('char' or 'bpe').")
    train_parser.add_argument("--bpe_vocab_size", type=int, default=380, help="Target vocabulary size for BPE.")
    train_parser.add_argument("--use_sdpa", action="store_true", help="Enable fused SDPA attention.")
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints.")
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU instead of MPS/CUDA.")

    # Generate subparser
    gen_parser = subparsers.add_parser("generate", help="Generate text from checkpoint.")
    gen_parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Checkpoint path.")
    gen_parser.add_argument("--prompt", type=str, default="First Citizen:\n", help="Initial prompt string.")
    gen_parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    gen_parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling filter.")
    gen_parser.add_argument("--cpu", action="store_true", help="Force CPU instead of MPS/CUDA.")

    args = parser.parse_args()
    if args.subcommand == "train":
        train_command(args)
    elif args.subcommand == "generate":
        generate_command(args)


if __name__ == "__main__":
    main()
