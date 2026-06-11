# Contamination Diagnostics Report

> **Language note:** labels refer to folded-ToF bins, not particle species.
> Species interpretation requires SME input (see Open Questions).

## 1. Baseline-Under-Peak Mislabel Estimate

Background density: 6.9602e-05 events/ns
Baseline window: 0.0–100.0

- **Class 1**: estimated mislabel fraction = 0.0076  (n=2601)
- **Class 2**: estimated mislabel fraction = 0.1236  (n=285)

## 2. Within-Bin Morphology Bimodality

### class_1
- n_pixels: **BIMODAL**, minority fraction ≈ 0.308
- tot_sum: **BIMODAL**, minority fraction ≈ 0.423
- tot_max: **BIMODAL**, minority fraction ≈ 0.486
- eccentricity: **BIMODAL**, minority fraction ≈ 0.308

![class_1 morphology](plots\morphology_class_1.png)

### class_2
- n_pixels: **BIMODAL**, minority fraction ≈ 0.091
- tot_sum: **BIMODAL**, minority fraction ≈ 0.091
- tot_max: **BIMODAL**, minority fraction ≈ 0.098
- eccentricity: **BIMODAL**, minority fraction ≈ 0.091

![class_2 morphology](plots\morphology_class_2.png)


## 3. Confident-Disagreement Map

Cross-tab of classifier predictions vs ToF labels (confident events only):

```
predicted_label  1    2
tof_label              
1                0  109
2                8    0
```

## Open Questions for SMEs

- **Flight path and trigger offset**: unknown — prevents ToF-to-species conversion.
- **Source pulse structure and expected species**: not provided.
- **Energy calibration**: no per-pixel calibration available; raw ToT used.
- **Time-walk correction**: no correction applied; see `dt_min_vs_argmax` diagnostic.
- **Expected maximum ToF**: determines whether a 1× period fold is sufficient.
