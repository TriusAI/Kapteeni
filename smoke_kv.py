"""Smoke: shared-prefix KV path must match full-forward (bf16 tolerance) and
be faster on long-state multi-option decisions."""
import time

import torch

from kapteeni.backbone import extract_h_last_verb, extract_shared_prefix, load_backbone
from kapteeni.serialize import choice_option_pass, state_text

STATE = {
    "policy": (
        "Coverage applies only when: (1) the leak began within 30 days of the "
        "claim, (2) the damaged surface was not already listed on the "
        "maintenance schedule, and (3) the customer paid the deductible. "
        "Water damage from repeated seepage is excluded. Flood damage is "
        "excluded. If a listed exclusion applies, the claim is denied in full."
        + " Additional terms: the adjuster report must estimate repair costs "
        "before approval; approvals above $10,000 require a second review; "
        "vacancy beyond 60 days voids coverage; the deductible is $1,000; "
        "emergency mitigation costs are reimbursable up to $500 with receipts."
        + " The customer reported a leak that ran 6 weeks, damaging a wall "
        "already on the maintenance schedule, with an estimate of $9,200."
    ),
    "claim": "Water leak damage claim, submitted 2026-09-24.",
}
SUFFIXES = [
    choice_option_pass(STATE, "Which outcome applies to this claim?",
                       f"outcome_{i}", f"Outcome number {i} of the review")[
        len(state_text(STATE)) + 2:]
    for i in range(5)
]
FULL = [state_text(STATE) + "\n\n" + s for s in SUFFIXES]

model, tok = load_backbone()
print("model loaded; state tokens:", len(tok(state_text(STATE))["input_ids"]))

# warmup
_ = extract_h_last_verb(model, tok, FULL[:1], batch_tokens=8192)
torch.cuda.synchronize()

t0 = time.time()
h_full, v_full = extract_h_last_verb(model, tok, FULL, batch_tokens=16384)
t_full = time.time() - t0

t0 = time.time()
h_shared, v_shared = extract_shared_prefix(
    model, tok, state_text(STATE), SUFFIXES)
t_shared = time.time() - t0

cos = torch.nn.functional.cosine_similarity(h_full.float(), h_shared.float(), dim=1)
print(f"\nfull-forward: {t_full:.2f}s   shared-prefix: {t_shared:.2f}s "
      f"({t_full/max(t_shared,1e-9):.1f}x)")
print("h cosine per pass:", [round(float(c), 5) for c in cos])
print("s_verb full:", [round(x, 2) for x in v_full.tolist()])
print("s_verb shared:", [round(x, 2) for x in v_shared.tolist()])
print("s_verb max |diff|:", round(float((v_full - v_shared).abs().max()), 3))
print("argmax agrees:", v_full.argmax().item() == v_shared.argmax().item())
print("peak GPU mem:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")