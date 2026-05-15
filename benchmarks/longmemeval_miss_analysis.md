# ContextFit LongMemEval miss analysis
Artifact: `benchmarks/longmemeval_coverage_companion_full.json`
Scored: 470. Any@5 misses: 23. Any@10 misses: 14. Rank 6-10: 9.
## Misses by type
- temporal-reasoning: 8
- multi-session: 6
- single-session-preference: 5
- single-session-user: 3
- knowledge-update: 1

## Current miss@5 details

### 3ba21379 — knowledge-update — best_rank=10
Q: What type of vehicle model am I currently working on?

Gold: answer_cd345582_1, answer_cd345582_2

Retrieved top10: d557e57a, 490bb46d_3, a946dbf5_3, 6d3ac017_1, ultrachat_74850, ddcd3c0c, sharegpt_xYURy32_0, ef8cc173_1, a50b0d73_2, answer_cd345582_1

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:10; longmemeval_contextfit_openai_fusion_full_top5:3; longmemeval_conversation_openai_fusion_full:1

### 88432d0a — multi-session — best_rank=7
Q: How many times did I bake something in the past two weeks?

Gold: answer_733e443a_1, answer_733e443a_2, answer_733e443a_3, answer_733e443a_4

Retrieved top10: bb2180e9_1, f777f641, 1c8832b4_2, e0956e0a_2, d1e7d11c_2, sharegpt_HpkiAOb_9, answer_733e443a_1, sharegpt_tESxx1y_0, answer_733e443a_2, 4b9ed528_1

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:4; longmemeval_conversation_openai_fusion_full:absent

### 87f22b4a — multi-session — best_rank=8
Q: How much have I made from selling eggs this month?

Gold: answer_f56e6152_1, answer_f56e6152_2

Retrieved top10: ultrachat_182225, 7c2ec9e9_2, 02b99ec3_1, 0320f558_2, 889076d6, ultrachat_350147, 4e671700, answer_f56e6152_1, 85846900_1, 28741edc_1

Compare ranks: longmemeval_ab_baseline_nolabel_full:7; longmemeval_parent_child_r100_full:5; longmemeval_contextfit_openai_fusion_full_top5:1; longmemeval_conversation_openai_fusion_full:1

### gpt4_f2262a51 — multi-session — best_rank=absent
Q: How many different doctors did I visit?

Gold: answer_55a6940c_1, answer_55a6940c_2, answer_55a6940c_3

Retrieved top10: sharegpt_BWMyoNr_0, 2aeb1268, ultrachat_139167, ultrachat_268434, c34b6a1c_2, sharegpt_hChsWOp_128, sharegpt_eoEbthf_0, 765339aa, c0d6fa6f_2, f2ffaf25

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### 92a0aa75 — multi-session — best_rank=absent
Q: How long have I been working in my current role?

Gold: answer_6cb8f792_1, answer_6cb8f792_2

Retrieved top10: ultrachat_178564, 2bdd78cb_1, 1c0921ee_3, bd00a24d_2, 85eb2b94_2, a5854a8d_2, 5a0d28f8_3, 35456e0c_2, 173e2e47, fdf7e3e7

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:9; longmemeval_contextfit_openai_fusion_full_top5:3; longmemeval_conversation_openai_fusion_full:2

### c18a7dc8 — multi-session — best_rank=absent
Q: How many years older am I than when I graduated from college?

Gold: answer_2e2085fa_1, answer_2e2085fa_2

Retrieved top10: sharegpt_ErOTMZ3_149, ultrachat_473615, 5022812c, ultrachat_463016, 9d5a389d, ultrachat_231462, ultrachat_353377, ultrachat_577269, 4ea4f02f_1, 4f52229e

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### 8e91e7d9 — multi-session — best_rank=absent
Q: What is the total number of siblings I have?

Gold: answer_477ae455_1, answer_477ae455_2

Retrieved top10: c96fac82_3, 4f8caea3_2, 4e3da326_1, 2f980ae0_2, 04bf1261, dfa4025c_3, ultrachat_147574, ultrachat_339024, sharegpt_PZrBCF2_0, ultrachat_257571

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### 32260d93 — single-session-preference — best_rank=absent
Q: Can you recommend a show or movie for me to watch tonight?

Gold: answer_0250ae1c

Retrieved top10: sharegpt_WYQSZ3n_0, sharegpt_H4jw5s7_39, 1b26bdd5_5, 04bef53b, c4d370d3_2, sharegpt_PfiDxfU_189, c9a763e9, 551c9f06, 6d3f5c68_4, ultrachat_171173

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### 06f04340 — single-session-preference — best_rank=absent
Q: What should I serve for dinner this weekend with my homegrown ingredients?

Gold: answer_92d5f7cd

Retrieved top10: 728deb4d_4, 0844dea6, 91223fd5_1, 6e6fbb6b, 66bfa1db, 42924d15, de1f4aec_2, sharegpt_MkLNumZ_0, 14f9ee3c, fea299b4

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### 09d032c9 — single-session-preference — best_rank=absent
Q: I've been having trouble with the battery life on my phone lately. Any tips?

Gold: answer_b10dce5e

Retrieved top10: 3fc2244f, b21bd3e2, 16ebc8f8_4, 21ef2d05_1, 612e23f1, 87e8ec02, af631aa3_2, sharegpt_e9sAtcZ_63, d3971322_1, e8bfacec_2

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### d24813b1 — single-session-preference — best_rank=absent
Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?

Gold: answer_7c0ade93

Retrieved top10: 3124bb28_1, 084802f9_3, 61a46ff7_3, 128f4e4d_3, sharegpt_JeaDWry_0, c2a34674_1, d49b1a24_3, 2108281a_1, d8e33f5c_abs_1, e08bf81f_4

Compare ranks: longmemeval_ab_baseline_nolabel_full:9; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:4; longmemeval_conversation_openai_fusion_full:absent

### d6233ab6 — single-session-preference — best_rank=absent
Q: I've been feeling nostalgic lately. Do you think it would be a good idea to attend my high school reunion?

Gold: answer_b0fac439

Retrieved top10: ecfd2047_1, 87cfeb28_2, e419b7c3_4, e6c3a50a, 94bc18df_3, 7af38385_2, ultrachat_457634, d850eba6_2, sharegpt_ZWqMvoL_607, 0e726047

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### 5d3d2817 — single-session-user — best_rank=6
Q: What was my previous occupation?

Gold: answer_235eb6fb

Retrieved top10: sharegpt_ipLglky_48, e93efd9a_1, 85d16d01, 7dc6f276_1, e7b0637e_3, answer_235eb6fb, 670ca40e, ultrachat_198348, 4ae45145, cef33b28_2

Compare ranks: longmemeval_ab_baseline_nolabel_full:6; longmemeval_parent_child_r100_full:6; longmemeval_contextfit_openai_fusion_full_top5:4; longmemeval_conversation_openai_fusion_full:3

### e01b8e2f — single-session-user — best_rank=6
Q: Where did I go on a week-long trip with my family?

Gold: answer_5ca6cd28

Retrieved top10: e74be4c6_4, 1aaf8057_3, 2b4104e3_2, 963e896f_2, b2543a58_3, answer_5ca6cd28, 8be2c3f1, sharegpt_oX0xInz_0, 97338ddd_1, ultrachat_76289

Compare ranks: longmemeval_ab_baseline_nolabel_full:3; longmemeval_parent_child_r100_full:6; longmemeval_contextfit_openai_fusion_full_top5:1; longmemeval_conversation_openai_fusion_full:1

### ccb36322 — single-session-user — best_rank=absent
Q: What is the name of the music streaming service have I been using lately?

Gold: answer_f1fbb330

Retrieved top10: bdcee74e_3, 04a0b385, c38b33ae, 522dd987_4, 56555c52_2, facfdfe2, ultrachat_395168, 8b1019b8_1, 6d52ee93_1, ultrachat_203880

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:4; longmemeval_conversation_openai_fusion_full:3

### 4dfccbf8 — temporal-reasoning — best_rank=6
Q: What did I do with Rachel on the Wednesday two months ago?

Gold: answer_4bebc783_1, answer_4bebc783_2

Retrieved top10: sharegpt_3D3oQC0_213, a63ad8e3_3, 7db28e68_1, ultrachat_444490, 6c16c3ec_2, answer_4bebc783_1, 1bc87711_1, 0943dee1, b4f63a70_3, edd89480_1

Compare ranks: longmemeval_ab_baseline_nolabel_full:4; longmemeval_parent_child_r100_full:6; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### gpt4_fa19884d — temporal-reasoning — best_rank=7
Q: What is the artist that I started to listen to last Friday?

Gold: answer_ff201787_1, answer_ff201787_2

Retrieved top10: ccc87da9_1, sharegpt_vonEwUo_17, ultrachat_525919, sharegpt_CIPQtDW_0, 5f9dd782, sharegpt_ROjo48X_0, answer_ff201787_2, f910dc31_4, ebd2a973, ultrachat_376026

Compare ranks: longmemeval_ab_baseline_nolabel_full:7; longmemeval_parent_child_r100_full:7; longmemeval_contextfit_openai_fusion_full_top5:2; longmemeval_conversation_openai_fusion_full:3

### gpt4_e061b84g — temporal-reasoning — best_rank=8
Q: I mentioned participating in a sports event two weeks ago. What was the event?

Gold: answer_8c64ce26_1, answer_8c64ce26_2, answer_8c64ce26_3

Retrieved top10: 9dac9e37_1, ae15e8b6_1, ce584ba0, 68705467_1, 0e3d8491_1, sharegpt_sDwO1bp_35, 7083fcb9_1, answer_8c64ce26_3, 3929e6cc_1, 09d021e7_2

Compare ranks: longmemeval_ab_baseline_nolabel_full:10; longmemeval_parent_child_r100_full:8; longmemeval_contextfit_openai_fusion_full_top5:2; longmemeval_conversation_openai_fusion_full:2

### 9a707b82 — temporal-reasoning — best_rank=9
Q: I mentioned cooking something for my friend a couple of days ago. What was it?

Gold: answer_dba89488_1, answer_dba89488_2

Retrieved top10: 8e78fa70_1, 990f3ef9_2, 7a4d00b3_2, 12b6f0f1_3, fab41c07, 79d122f0, e72739cd_1, f62c04c6, answer_dba89488_1, f8e9d4fa

Compare ranks: longmemeval_ab_baseline_nolabel_full:8; longmemeval_parent_child_r100_full:9; longmemeval_contextfit_openai_fusion_full_top5:4; longmemeval_conversation_openai_fusion_full:5

### gpt4_468eb063 — temporal-reasoning — best_rank=absent
Q: How many days ago did I meet Emma?

Gold: answer_9b09d95b_1

Retrieved top10: sharegpt_ipLglky_42, d97db962_2, e60a93ff_2, sharegpt_1rIk8tS_0, ce584ba0, c6a2e4cc, 320482e0_2, 47183c60_5, sharegpt_Sf51OjE_8, 09599d45_3

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### gpt4_468eb064 — temporal-reasoning — best_rank=absent
Q: Who did I meet with during the lunch last Tuesday?

Gold: answer_9b09d95b_1

Retrieved top10: a2f0054f, ultrachat_337251, 5bd9f1e6_3, 4e77dbbe_6, 1e5bd28d_2, sharegpt_0PLBHdX_15, 4d4df0e0_1, ultrachat_109577, db1aefb6_3, 970a6e5c

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### eac54add — temporal-reasoning — best_rank=absent
Q: What was the significant buisiness milestone I mentioned four weeks ago?

Gold: answer_0d4d0348_1, answer_0d4d0348_2

Retrieved top10: 2ea6fda4_1, 28269633_2, 7966888b_2, 369695b4_2, ultrachat_208612, e552e1f9_2, 7bb01253, ultrachat_133409, a1166ecc_3, 773aebbd_3

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent

### gpt4_8279ba03 — temporal-reasoning — best_rank=absent
Q: What kitchen appliance did I buy 10 days ago?

Gold: answer_56521e66_1

Retrieved top10: 518d26d3_4, 652c0717_3, b357fb8b_2, ba9f938b_1, 49b78a55_1, 50d66391_4, 66d3fdb7_1, ultrachat_507878, 2ef55f49_3, 570fe405

Compare ranks: longmemeval_ab_baseline_nolabel_full:absent; longmemeval_parent_child_r100_full:absent; longmemeval_contextfit_openai_fusion_full_top5:absent; longmemeval_conversation_openai_fusion_full:absent
