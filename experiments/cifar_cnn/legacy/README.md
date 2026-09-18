# Legacy subset-selection helper

`data_subset.py` is preserved from an earlier experiment but is not imported by
the current CIFAR training scripts. Its beta-sampling paths require the external
`dynm_utils` module, which was not present in the original `MC` directory.
