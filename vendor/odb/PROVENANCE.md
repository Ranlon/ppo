# ODB source

Copied from `Project200/odb200/src/odb` (package version 0.1.2) for the
`obd_sft` integration. Apache-2.0 license is included in this directory.
Runtime source hashes are included in each prepared run's RUNTIME_BUILD.json.

Local correction: use a two-element wake-up item for PyTorch's pin-memory
thread at shutdown. The original archive sends None, although PyTorch expects
an (index, data) pair.
