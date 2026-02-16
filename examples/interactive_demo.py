"""Launch the interactive neuron steering REPL.

Commands available in the REPL:
    prompt <text>           - Run a prompt, show output
    discover [target]       - Find circuit for current prompt
    ablate [spec]           - Ablate neurons (L23/N8079, top10, all)
    amplify [spec] [mult]   - Amplify neurons (default 2.0x)
    sweep [m1 m2 ...]       - Multiplier sweep
    edges [top_k]           - Compute edge attributions
    top [k]                 - Top-k next-token predictions
    save <name>             - Save circuit to file
    load [name]             - Load circuit (no arg = list available)
    info                    - Show current state
    quit / exit             - Exit REPL
"""
import torch
from neuron_steer import NeuronSteerer

steerer = NeuronSteerer("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
steerer.interactive()
