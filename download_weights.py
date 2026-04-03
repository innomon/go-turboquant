import os
import argparse
from huggingface_hub import snapshot_download

def download_gemma3(model_id="google/gemma-3-4b-it", local_dir="models/gemma-3-4b-it"):
    """
    Downloads Gemma 3 weights from Hugging Face.
    """
    print(f"🚀 Starting download for {model_id}...")
    print(f"📂 Target directory: {os.path.abspath(local_dir)}")

    # Ensure the directory exists
    os.makedirs(local_dir, exist_ok=True)

    try:
        # We download safetensors, config, and tokenizer files
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.safetensors", 
                "*.json", 
                "*.txt",
                "tokenizer.model"
            ],
            # Note: Gemma 3 requires accepting a license on Hugging Face.
            # Ensure you have run 'huggingface-cli login' or set the HF_TOKEN env var.
        )
        print(f"\n✅ Download complete!")
        print(f"📍 Files are located at: {path}")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n💡 Tip: Ensure you have accepted the license for Gemma 3 on Hugging Face")
        print("💡 Tip: Run 'pip install huggingface_hub' and 'huggingface-cli login' first.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gemma 3 weights from Hugging Face.")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it", help="Hugging Face model ID")
    parser.add_argument("--dir", type=str, default="models/gemma-3-4b-it", help="Local directory to save weights")
    
    args = parser.parse_args()
    download_gemma3(args.model, args.dir)
