"""GPU smoke: load backbone, extract h_last, verify shapes + semantic sanity."""
import torch
from kapteeni.backbone import load_backbone, extract_h_last
from kapteeni.serialize import noul_pass

model, tok = load_backbone()
print("model loaded:", model.config.model_type, "hidden:", model.config.hidden_size)
print("gpu mem allocated:", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")

crit = {"true": "supported by the evidence", "false": "contradicted by the evidence"}
passes = [
    noul_pass({"claim": "The sky is blue", "evidence": "Observers report the sky is blue."},
              "Is `claim` supported by `evidence`?", crit),
    noul_pass({"claim": "The sky is green", "evidence": "Observers report the sky is blue."},
              "Is `claim` supported by `evidence`?", crit),
    noul_pass({"claim": "Paris is the capital of France", "evidence": "France's government sits in Paris."},
              "Is `claim` supported by `evidence`?", crit),
]
h = extract_h_last(model, tok, passes, batch_tokens=4096)
print("h shape:", tuple(h.shape), h.dtype)
print("norms:", [round(float(x.norm()), 2) for x in h])
cos = torch.nn.functional.cosine_similarity
print("cos(blue|blue vs green|blue):", round(float(cos(h[0], h[1], dim=0)), 4))
print("cos(blue|blue vs paris|paris):", round(float(cos(h[0], h[2], dim=0)), 4))
print("OK")