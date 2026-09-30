# Response evaluation

The benchmark asks a judge to identify the authored game continuation from an original/generated
pair with shared context. The judge supplies a choice, confidence, recognition flag and free-text
reason. Origin detection measures distinguishability, not storytelling quality.

Preparation verifies saved contexts and response provenance before exporting trials. Public trials
contain opaque IDs, context and candidates A and B. Private mappings contain origin labels and
model identities. Candidate serialization is always A then B; original positions are randomized
independently. Each judgment is an isolated request. The current benchmark uses one order per pair
and reports controls separately.

Reports include correct, incorrect, abstained, failed and missing counts. Accuracy uses decided
trials as its denominator; coverage includes all scheduled trials. Paired comparisons use shared
scenes. Conversation groups, rather than repeated turns, define the uncertainty units. Recognition
and free-text explanations remain inspectable.

Generation and judge runners preserve settings, requests, responses and usage. Failed or uncertain
requests are retained rather than silently retried. Hosted execution requires explicit provider
configuration and a budget. Evaluation does not initiate training.

Historical evidence predating the A-then-B serialization fix contains an original-first leak and
must not be pooled with corrected results. Validation has been used for development. A model judging
its own generations can introduce self-preference. Neither a low detection rate nor failure to find
a difference establishes equivalence or human approval.
