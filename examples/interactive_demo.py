"""Interactive neuron steering REPL.

Commands: prompt, discover, ablate, amplify, sweep, edges, top, save, load, info, quit
"""
from neuron_steer.core import NeuronSteerer

steerer = NeuronSteerer("meta-llama/Llama-3.1-8B-Instruct")
steerer.interactive()
