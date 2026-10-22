#!/usr/bin/env python3
"""Real optional Metal trace capture, for inspection with Xcode GPU tools."""
import argparse,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'air_llm'))
from airllm.nextgen import AirLLMNextModel
from airllm.nextgen.tuning import auto_config


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--capture',default='evidence-next/decode.gputrace');p.add_argument('--prompt',default='Bonjour');p.add_argument('--quantized-dir');a=p.parse_args()
    if os.environ.get('MTL_CAPTURE_ENABLED')!='1':raise SystemExit('Relancer avec MTL_CAPTURE_ENABLED=1 pour autoriser la capture Metal')
    import mlx.core as mx
    if not mx.metal.is_available():raise SystemExit('Capture indisponible sans GPU Metal réel')
    Path(a.capture).parent.mkdir(parents=True,exist_ok=True)
    with AirLLMNextModel(a.model,next_config=auto_config(device='gpu',quantized_dir=a.quantized_dir)) as model:
        ids=[model.tokenizer.encode(a.prompt)];model.generate(ids,max_new_tokens=1)
        mx.synchronize()
        mx.metal.start_capture(a.capture)
        try:model.generate(ids,max_new_tokens=4);mx.synchronize()
        finally:mx.metal.stop_capture()
    print(a.capture)


if __name__=='__main__':main()
