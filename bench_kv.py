"""Latency benchmark: shared-prefix vs full-forward on a 4k-token state."""
import time

import torch

from kapteeni.backbone import extract_h_last_verb, extract_shared_prefix, load_backbone
from kapteeni.serialize import choice_option_pass, state_text

policy = (
    "Section 1. Coverage scope. This policy covers sudden and accidental "
    "discharge of water from plumbing within the insured premises. Section 2. "
    "Exclusions. Losses from flood, surface water, sewer backup, or repeated "
    "or continuous seepage occurring over a period of 14 or more days before "
    "the loss are excluded. Section 3. Conditions. The insured must have "
    "completed all maintenance listed on the schedule within the preceding "
    "12 months. Vacancy beyond 60 consecutive days voids coverage. Section 4. "
    "Limits. Water damage is subject to a special limit of $10,000 unless an "
    "endorsement raises it. Emergency mitigation is reimbursable up to $500 "
    "with receipts. The standard deductible is $1,000. Section 5. Claims "
    "procedure. A licensed adjuster must inspect and estimate repairs before "
    "approval; approvals above $10,000 require a second review by the "
    "regional office; partial approvals are permitted at the adjuster's "
    "discretion. Section 6. Definitions. 'Seepage' means water intrusion "
    "occurring slowly over time; 'sudden' means occurring within 24 hours of "
    "the causal event; 'maintenance schedule' means the document attached as "
    "Endorsement A; 'vacancy' means no personal property or occupancy for the "
    "stated period. Section 7. Subrogation and cooperation. The insured must "
    "cooperate with the investigation, preserve damaged property for "
    "inspection, and submit receipts within 30 days of the loss. Section 8. "
    "Endorsements. Endorsement A lists the maintenance schedule. Endorsement B "
    "raises the water damage limit to $15,000 for an additional premium. "
    "Endorsement C adds sump pump overflow coverage. Section 9. History. "
)
state = {"policy": policy * 14, "claim": "Leak ran 6 weeks; wall on schedule; estimate $9,200."}
SUFFIXES = [
    choice_option_pass(state, "Which outcome applies to this claim?", f"outcome_{i}",
                       f"Outcome number {i} of the review")[len(state_text(state)) + 2:]
    for i in range(5)
]
FULL = [state_text(state) + "\n\n" + s for s in SUFFIXES]

model, tok = load_backbone()
ntok = len(tok(state_text(state))["input_ids"])
print("state tokens:", ntok)

_ = extract_h_last_verb(model, tok, FULL[:1], batch_tokens=8192)  # warmup

for name, fn in (
    ("full-forward", lambda: extract_h_last_verb(model, tok, FULL, batch_tokens=16384)),
    ("shared-prefix", lambda: extract_shared_prefix(model, tok, state_text(state), SUFFIXES)),
):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    print(f"{name}: {(time.time()-t0)/3:.2f}s per 5-option decision")
print("peak GPU mem:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")