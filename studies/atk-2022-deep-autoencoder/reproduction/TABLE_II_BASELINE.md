# Table II breadth baseline

## Question

Can every Table II model be run once on the exact SGCC source using the
smallest architecture-preserving completion, and are its reported metrics even
in the same numerical region?

## Frozen ladder

1. **Printed instruction:** 1,034 SGCC daily readings are supplied to models
   whose printed input width is 48.
2. **Status:** non-executable; the paper supplies no 1,034-to-48 operation or
   missing-value rule. This failure is itself a result.
3. **First executable assumption (`I-SGCC-LAST48`):** chronological dates,
   within-customer interpolation plus benign-B1 edge medians, one most-recent
   48-day row per customer.
4. **Printed order preserved:** joint benign/malicious feature scaling;
   anomaly `B1` training and ADASYN on `B2+M`; supervised ADASYN before a 2:1
   row split.
5. **Breadth run:** seed 11, one run of all six benchmarks and all five
   proposed models, exact 48-wide architectures from Table I, printed anomaly
   thresholds transferred from ISET, and every attempt preserved.

This baseline can falsify a claim that the frozen completion reproduces the
table. It cannot prove that no conceivable unreported preprocessing choice
could match it. That stronger question is addressed only after the breadth map
by predeclared one-factor branches, repeated seeds, uncertainty intervals, and
the already-frozen metric-identity checks.

After the complete `last_48` model-family map, two one-factor representation
contrasts are run without changing anything else: `first_48`, and
`binned_mean_48` (48 contiguous chronological means spanning all 1,034 days).
Together they test early, late, and whole-history 48-wide readings. Windowed
multi-sample interpretations change the sample unit and split semantics and
remain a later distinct branch rather than being mixed into this contrast.
