"""Minimal example: find and steer capital city neurons."""
import torch
from neuron_steer import NeuronSteerer

steerer = NeuronSteerer("meta-llama/Llama-3.1-8B-Instruct")

# Discover the circuit for "say a capital city"
circuit = steerer.find_feature(
    prompt="What is the capital of the state containing Dallas?",
    target=" Austin",
    name="capitals",
    seed_response="Answer:",
    top_k=200,
)
print(circuit.summary())

# Steer it
normal = steerer.generate("What is the capital of Ohio?")
ablated = steerer.steer("What is the capital of Ohio?", feature="capitals", multiplier=0.0)
amplified = steerer.steer("What is the capital of Ohio?", feature="capitals", multiplier=3.0)

print(f"\nNormal:    {normal}")
print(f"Ablated:   {ablated}")
print(f"Amplified: {amplified}")

# Inspect the circuit
print(f"\nTop 10 neurons:")
for nidx, attr in circuit.top(10):
    print(f"  L{nidx.layer}/N{nidx.neuron}: {attr:+.4f}")
