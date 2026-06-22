"""
Download alibaba-pai/VideoCLIP-XL model from HuggingFace for testing.

This script downloads the VideoCLIP-XL model which supports longer text sequences
compared to standard CLIP models.
"""

import os
from pathlib import Path

try:
    from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠ Warning: transformers library not available")

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠ Warning: huggingface_hub library not available")


def download_videoclip_xl(model_name="alibaba-pai/VideoCLIP-XL", output_dir=None):
    """
    Download VideoCLIP-XL model from HuggingFace.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Optional local directory to save model (if None, uses HF cache)
    """
    print("=" * 60)
    print(f"Downloading VideoCLIP-XL: {model_name}")
    print("=" * 60)
    
    # Method 1: Try AutoModel with trust_remote_code
    if TRANSFORMERS_AVAILABLE:
        print("\n[Method 1] Trying AutoModel.from_pretrained with trust_remote_code=True...")
        try:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=False
            )
            print(f"   ✓ Model loaded: {type(model).__name__}")
            
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"\n2. Saving model to {output_path}...")
                model.save_pretrained(output_path)
                print(f"   ✓ Model saved to {output_path}")
            
            print("\n3. Loading tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                print(f"   ✓ Tokenizer loaded")
                if output_dir:
                    tokenizer.save_pretrained(output_path)
                    print(f"   ✓ Tokenizer saved to {output_path}")
            except Exception as e:
                print(f"   ⚠ Warning: Could not load tokenizer: {e}")
            
            print("\n4. Loading processor (if available)...")
            try:
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                print(f"   ✓ Processor loaded")
                if output_dir:
                    processor.save_pretrained(output_path)
                    print(f"   ✓ Processor saved to {output_path}")
            except Exception as e:
                print(f"   ⚠ Warning: Could not load processor: {e}")
            
            print("\n" + "=" * 60)
            print("✓ Download complete!")
            print("=" * 60)
            print(f"\nModel info:")
            print(f"  - Model type: {type(model).__name__}")
            print(f"  - Cache location: {os.environ.get('HF_HOME', 'default')}")
            if output_dir:
                print(f"  - Saved to: {output_path}")
            return
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            print("   Trying alternative method...")
    
    # Method 2: Use huggingface_hub to download files directly
    if HF_HUB_AVAILABLE:
        print("\n[Method 2] Using huggingface_hub to download files directly...")
        try:
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"   Downloading to: {output_path}")
                
                # Download all files including the large .bin file
                print("   Downloading all repository files (this may take a while for large files)...")
                snapshot_download(
                    repo_id=model_name,
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False,
                    resume_download=True  # Resume if interrupted
                )
                
                # Check if bin file exists
                bin_file = output_path / "VideoCLIP-XL.bin"
                if bin_file.exists():
                    size_mb = bin_file.stat().st_size / (1024 * 1024)
                    print(f"   ✓ Model weights file found: {bin_file.name} ({size_mb:.2f} MB)")
                else:
                    print(f"   ⚠ Warning: {bin_file.name} not found, trying to download directly...")
                    # Try to download bin file directly
                    try:
                        hf_hub_download(
                            repo_id=model_name,
                            filename="VideoCLIP-XL.bin",
                            local_dir=str(output_path),
                            local_dir_use_symlinks=False,
                            resume_download=True
                        )
                        if bin_file.exists():
                            size_mb = bin_file.stat().st_size / (1024 * 1024)
                            print(f"   ✓ Model weights file downloaded: {bin_file.name} ({size_mb:.2f} MB)")
                    except Exception as e2:
                        print(f"   ⚠ Could not download bin file directly: {e2}")
                
                # List all downloaded files
                print(f"\n   Downloaded files:")
                for file in sorted(output_path.rglob("*")):
                    if file.is_file():
                        size_mb = file.stat().st_size / (1024 * 1024)
                        print(f"     - {file.name}: {size_mb:.2f} MB")
                
                print(f"\n   ✓ All files downloaded to {output_path}")
            else:
                # Download to cache
                print("   Downloading to HuggingFace cache...")
                cache_dir = snapshot_download(
                    repo_id=model_name,
                    local_files_only=False,
                    resume_download=True
                )
                print(f"   ✓ Files downloaded to cache: {cache_dir}")
            
            print("\n" + "=" * 60)
            print("✓ Download complete!")
            print("=" * 60)
            print("\nNote: Model files have been downloaded.")
            print("You may need to load the model using custom code or check the model's documentation.")
            if output_dir:
                print(f"Files location: {output_path}")
            return
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # If all methods failed
    print("\n" + "=" * 60)
    print("❌ All download methods failed!")
    print("=" * 60)
    print("\nPossible solutions:")
    print("1. Check if the model exists on HuggingFace:")
    print(f"   https://huggingface.co/{model_name}")
    print("2. Update transformers library:")
    print("   pip install --upgrade transformers")
    print("3. Install huggingface_hub:")
    print("   pip install huggingface_hub")
    print("4. Check if the model requires authentication (private model)")
    print("5. Try downloading manually from HuggingFace website")
    raise RuntimeError("Failed to download model using all available methods")


def main():
    """Main function."""
    # File is in PhysicalDB/utils/, so we need to go up 3 levels to reach project root
    # Save model to PhysicalDB/VideoCLIP-XL
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "PhysicalDB" / "VideoCLIP-XL"
    
    download_videoclip_xl(
        model_name="alibaba-pai/VideoCLIP-XL",
        output_dir=str(output_dir)
    )


if __name__ == "__main__":
    main()
