## validation and test set masking: expanding‑window evaluation, pre‑quential evaluation, rolling forecast origin
| t | date | epss | use\_for\_loss‑TRAIN | use\_for\_loss‑VAL | use\_for\_loss‑TEST |
| - | ---- | ---- | -------------------- | ------------------ | ------------------- |
| 0 | d0   | .02  | **1**                | 0                  | 0                   |
| 1 | d1   | .03  | **1**                | 0                  | 0                   |
| 2 | d2   | .05  | **1**                | 0                  | 0                   |
| 3 | d3   | .08  | 0                    | **1**              | 0                   |
| 4 | d4   | .12  | 0                    | **1**              | **1**               |

### terminology for this:
| what we did                               | usual term in literature                                                                  |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| keep history, score tail only             | **expanding‑window evaluation**, **pre‑quential evaluation**, **rolling forecast origin** |
| Boolean row mask that disables loss       | **hold‑out mask**, **evaluation mask**                                                    |
| prefix used only for hidden‑state warm‑up | **context window**, **warm‑up window**, **history length**                                |


## latest: exp_window for lstm writes out (to numpy): 
predictions_fullseq.npz 
   ├─ pred    float32 [N_cve, L_max, H]
   ├─ true    float32 [N_cve, L_max, H]
   ├─ mask    float32 [N_cve, L_max, H]  ← horizon‐valid mask 
   └─ eval_mask float32 [N_cve, L_max]   ← which rows count for test 
