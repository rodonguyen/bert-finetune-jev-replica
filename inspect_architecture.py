"""Print the main components of the untouched GLiNER 2.5 base checkpoint."""

from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1", map_location="cpu")
encoder = model.encoder.config

print(f"Model: {type(model).__name__}")
print(f"Encoder: {encoder.model_type}")
print(f"Encoder layers: {encoder.num_hidden_layers}")
print(f"Hidden width: {encoder.hidden_size}")
print(f"Attention heads per layer: {encoder.num_attention_heads}")
print(f"Feed-forward width: {encoder.intermediate_size}")
print(f"Tokenizer: {type(model.processor.tokenizer).__name__}")
print("\nTop-level components:")
for name, module in model.named_children():
    parameters = sum(parameter.numel() for parameter in module.parameters())
    print(f"  {name:18} {type(module).__name__:24} {parameters:>12,} parameters")
print(f"Total: {sum(parameter.numel() for parameter in model.parameters()):,} parameters")
print("\nBANKING77 classification head:")
print(model.classifier)
