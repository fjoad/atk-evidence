# Full-sequence Sigmoid-bridge GPU preflight

Job407255 completed **0:0 in8:01** on one V100-16GB (crimv3mgpu026) on
Panther, with4 CPUs/16GiB and a20-minute maximum allocation. Frozen code:
3705bcca04279bbfa8523a573b9b573f123463b7. All14 sequential software checks
passed before timing three constructed48-step epochs (4,464 rows, batch100
with64-row remainder). No CER inputs or research fit were included.

The source-supported Sigmoid interpretation passed both fixed local software
learning cases; the separate linear control also passed but is not selected
for numerical continuation. Their complete record is
[the interface comparison](../sequential_interface_20261001/README.md).

The three epoch times were58.0757,36.7454 and36.9169 seconds (135 updates).
The gate uses50 times the slower warmed epoch: **1845.8433 seconds**, below
the3,600-second ceiling. Device, finite-value, shape/count and exact model/
optimizer reload checks all pass. TensorFlow's training allocator peak was
1,570,985,216 bytes; this excludes memory outside that allocator and is not
total process/driver VRAM. The preflight program took152.55s; the8:01 job
also includes software checks, imports and startup. Allocation ceilings are
not measured execution costs.

[preflight.json](preflight.json) SHA256 is
`721256059d34a104fa2eda04ccb0bb8775603d01206117e4e82e1a9e54782c07`,
identical on Panther and locally, and accepted by the direct runner's gate
verifier. [The job log](slurm-407255.txt) is preserved. Passing this timing
check does not establish useful learning on CER. The separately authorized
fixed real-data pair was subsequently submitted once as407294; see its
[record](../sequential_pilot_20261001/README.md).

Do not repeat the completed preflight. Outputs use
`sequential-preflight-sigmoid-20261001-attempt1` under
the original Panther study data directory. The isolated code checkout is
`atk-evidence-paper3-seq-20261001`; the transfer log is
`seq-transfer-20261001/preflight-407255.out`. Website remains untouched.
