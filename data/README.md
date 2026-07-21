# Local Data Directory

Raw and derived datasets live here but are intentionally ignored by Git.
Recreate them from a fresh clone using [`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md).

Expected Study 1 layout after acquisition:

```text
data/
  raw/
    sgcc-source/          # author-linked repository at the pinned commit
    sgcc-verified/
      data.csv            # checksum-verified extracted SGCC table
    cer-authorized/
      File1.txt.zip       # restricted, authorized ISSDA downloads
      ...
      File6.txt.zip
```

Never modify raw inputs in place. Derived data must record the source hashes and
transform configuration that produced it.

