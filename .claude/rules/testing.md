# Testing Discipline

- Test deterministic parsing, attack construction, metric calculations,
  preprocessing, split invariants, and experiment-matrix validation.
- Write tests before new deterministic implementation logic.
- Do not mock external services merely to create a passing test; verify the
  boundary logic and document manual checks.
- Every confirmatory experiment must validate exact data hashes, frozen branch
  IDs, seeds, and configuration before execution.
- A result is incomplete without raw scores/predictions, configuration, status,
  environment, and failure records.
- Never claim a step complete without running the full relevant test suite.

Current deterministic suite:

```bash
bash scripts/test.sh
```
