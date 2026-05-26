# rec_fine_ranking — RecFlow CTR benchmark across 7 ranking models. See experiments/plans/2026-05-25-rec-fine-ranking-design.md for design.

A fair, compute-aligned benchmark of 7 ranking architectures (DCNv2, DCNv3, MaskNet, GDCN, OneTrans, RankMixer, UniMixer, HyFormer) on the RecFlow dataset under shared embeddings and matched FLOPs.

Each model is trained and evaluated with the same data pipeline, embedding tables, optimizer settings, and held-out evaluation set so that differences in AUC / GAUC / LogLoss reflect architectural choices rather than confounds.
