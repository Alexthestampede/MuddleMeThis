#!/usr/bin/env python3
"""
List available models and LoRAs from the Draw Things gRPC server.
"""

import argparse
import imageService_pb2
from drawthings_client import DrawThingsClient

def list_models(server='192.168.2.150:7859', show_loras=False):
    """
    Query the server for available models and optionally LoRAs.
    
    Args:
        server: Server address
        show_loras: If True, also list LoRAs
    """
    client = DrawThingsClient(server, insecure=False, verify_ssl=False)
    
    try:
        # Use Echo to get file list
        echo_request = imageService_pb2.EchoRequest(name='list_files')
        response = client.stub.Echo(echo_request)
        
        if response.files:
            # Categorize files
            models = []
            loras = []
            other = []
            
            for filename in response.files:
                lower = filename.lower()
                if '.ckpt' in lower or '.safetensors' in lower:
                    # Determine if it's a LoRA or model
                    # LoRAs are typically in loras/ directory or have specific naming
                    if 'lora' in lower or filename.startswith('loras/'):
                        loras.append(filename)
                    else:
                        models.append(filename)
                else:
                    other.append(filename)
            
            # Display models
            print(f"\n📦 Available Models ({len(models)}):")
            print("=" * 80)
            
            # Group by type
            sd15_models = [m for m in models if 'xl' not in m.lower() and 'sdxl' not in m.lower()]
            sdxl_models = [m for m in models if 'xl' in m.lower() or 'sdxl' in m.lower()]
            other_models = [m for m in models if m not in sd15_models and m not in sdxl_models]
            
            if sd15_models:
                print("\n🔷 SD 1.5 / Similar Architecture:")
                for model in sorted(sd15_models):
                    print(f"  • {model}")
            
            if sdxl_models:
                print("\n🔶 SDXL Architecture:")
                for model in sorted(sdxl_models):
                    print(f"  • {model}")
            
            if other_models:
                print("\n❓ Other / Unknown Architecture:")
                for model in sorted(other_models):
                    print(f"  • {model}")
                print("\n  ⚠️  For these models, you may need to manually specify the latent size.")
            
            # Display LoRAs if requested
            if show_loras and loras:
                print(f"\n🎨 Available LoRAs ({len(loras)}):")
                print("=" * 80)
                for lora in sorted(loras):
                    print(f"  • {lora}")
            
            # Usage hint
            print("\n💡 Usage:")
            print("  python generate_image.py \"your prompt\" --model <model_name>")
            print("  python generate_image.py \"your prompt\" --model <model_name> --latent-size <64|128>")
            
        else:
            print("No files returned from server. Make sure --model-browser flag is set.")
            
    except Exception as e:
        print(f"Error listing models: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='List available models and LoRAs from Draw Things server'
    )
    parser.add_argument('--server', default='192.168.2.150:7859', help='Server address')
    parser.add_argument('--loras', action='store_true', help='Also show LoRAs')
    
    args = parser.parse_args()
    
    list_models(args.server, args.loras)
