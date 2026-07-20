# Data sources and integrity

## SGCC electricity-theft dataset

- Publisher-linked repository: https://github.com/henryRDlab/ElectricityTheftDetection/
- Corresponding-author dataset page: https://www.henrylab.net/pubs/wide-deep-convolutional-neural-networks-for-electricity-theft-detection-to-secure-smart-grids/
- Repository commit downloaded: `8db682e65422d24689a61bd044eab7235121c5df`
- Raw multipart archive location: `../data/raw/sgcc-source/`
- Verified extraction: `../data/raw/sgcc-verified/data.csv`
- Extracted CSV SHA-256: `99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7`
- Shape: 42,372 customers x 1,036 columns.
- Contents: customer identifier, binary `FLAG`, and 1,034 daily readings from 2014-01-01 through 2016-10-31.
- Labels: 38,757 `FLAG=0`; 3,615 `FLAG=1`.
- Missing consumption cells: 11,233,528 (25.64%).
- Fully missing customers: 5.
- Observed zero fraction: 17.77%.
- Duplicate customer identifiers: 0.

Important: columns in the source CSV are lexicographically ordered, not chronologically ordered. Sequential models must parse and sort the dates explicitly.

Multipart archive SHA-256 values:

- `data.z01`: `c324df53c88358a50aa23fd843b1e15af06e7a20b72d901a98d957c304a52b67`
- `data.z02`: `34a30c8eea0fdfa77d58e15cd01c5593ea354ead4cc408a1b87364f6c46d4ed7`
- `data.zip`: `1e06ad5f5e13f56f2a72bea304864d259e060d6ad95b3b030a4ad050d8df82d4`

The archive was tested successfully with 7-Zip 26.02 before extraction. The built-in macOS `unzip` utility cannot correctly process this multipart archive.

## Irish CER Smart Metering Project (ISET in the paper)

- Official record: https://doi.org/10.7929/ISSDA/BX59EU
- Publisher: Commission for Energy Regulation (CER), via ISSDA.
- Dataset title: `CER Smart Metering Project - Electricity Customer Behaviour Trial, 2009-2010`.
- Version: V1.
- Access: restricted for research/educational use; an ISSDA account and approved request are required.
- Official manifest and unrestricted documentation: `../data/raw/cer-issda-docs/`.

Restricted consumption archives listed by the official API:

| File | Bytes | Official MD5 |
|---|---:|---|
| File1.txt.zip | 101,978,611 | `00203f66f3f5e5201b20ed160b787684` |
| File2.txt.zip | 102,197,028 | `5e3af1474d3c8976e2e1e0f8c1969507` |
| File3.txt.zip | 101,624,145 | `b537785f8b37cb3e89103600d39da8ff` |
| File4.txt.zip | 102,401,577 | `53ec9e70c1610b74ae72417cc010a0c3` |
| File5.txt.zip | 102,257,883 | `6f8c7c9dfba3bbfbff0e5f1703e122fc` |
| File6.txt.zip | 147,826,765 | `c0a435d0359974f23ce434b5e838e251` |

Manifest format:

- Three columns: meter ID, five-digit day/time code, and half-hour kWh consumption.
- Day code occupies digits 1-3; day 1 is 2009-01-01.
- Time code occupies digits 4-5; values 1-48 represent half-hour intervals.

No unofficial mirror will be silently substituted for the restricted official files. If an authorized copy is supplied, every archive will be checked against the official MD5 before use.

