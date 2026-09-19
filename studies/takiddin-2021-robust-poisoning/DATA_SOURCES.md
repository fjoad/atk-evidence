# Data sources and present access state

## Current verification, 20 September 2026

The named data are available through the
[official ISSDA record](https://doi.org/10.7929/ISSDA/BX59EU), whose consumption
files require approved access, and a
[ScienceDB deposit](https://doi.org/10.57760/sciencedb.17619). The six
consumption ZIPs already present locally and on Panther are the ScienceDB
copies. A new verification against the live official API matched all six
filenames, byte sizes, and MD5 checksums. Their ZIP CRC checks passed too.
See [the new verification record](results/data_identity_20260920.json).

The allocation CSV is a format conversion with SHA-256
96298be047f34ba91fe281c899b440d2b28747b4f102af6f239dbbd93dd354d4.
It contains 6,445 unique IDs: 4,225 residential, 485 SME, and 1,735 other.
A fresh read-only comparison with the
[independent public workbook](https://github.com/wwzjustin/CER-Smart-Meter-Project-by-Irish-Social-Science-Data-Archive/blob/db791cfeb5d725b28bfbd4b2b1bf33a8cadaa15c/SME%20and%20Residential%20allocations.xlsx)
matched all five allocation columns after blank/zero normalization. This
checks the mapping's contents, not the official TAB serialization.

The [official file manifest](https://issda.ucd.ie/api/access/datafile/793)
documents day 1 as 1 January 2009 and half-hour slots 1-48. Raw source files
and the independent workbook remain outside Git; only identities and counts
are published. The mirror's license label does not override ISSDA terms.

## Named source

The paper names the Irish Smart Energy Trial and cites its Irish Social Science
Data Archive record as reference [19]. It describes 3,000 residential meters,
30-minute readings over 18 months, and roughly 25,000 reports per customer.

The exact 3,000 customer IDs remain unspecified. The newly authorized
[preparation contract](PREPARATION.md) makes a seeded selection before any
outcome is observed. It is an explicit reconstruction choice using the named
source, not a claim to have recovered the authors' exact sample.

## Admission requirements

The new contract records:

- the authoritative record and access terms;
- exact archive filenames, sizes, and checksums;
- which 3,000 customers the paper used;
- date coverage and daily-profile admission rules;
- missing, duplicate, and daylight-saving handling;
- whether the allocation metadata used by the earlier paper is relevant; and
- whether any reused local bytes are identical to this paper's named source.

Restricted files, credentials, and copyrighted PDFs remain outside Git.
Exact-sample reproduction remains unavailable without the original IDs.
The September 20 direction permits a disclosed, outcome-independent selection
for an interpreted reproduction; no target-guided subset may replace it.
