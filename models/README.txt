Put the model files here (they are not part of the code folder):

  large_vox_iter5.pt          3,905,065,977 bytes   AV-HuBERT Large checkpoint (Meta) — never modified
  lipsync_detector_v4.pkl            7,001 bytes   trained classifiers + thresholds (scikit-learn 1.6.1)

Created by ./setup.sh (tools/make_slim_ckpt.py):
  avhubert_slim.pt            ~1.30 GB             only the weights the model uses, verified identical
                                                   (the app loads this one: ~1 s instead of ~40 s, ~3 GB less RAM)
Created by tools/compare_devices.py (optional):
  device_check.json                                whether this Mac may use MPS (Apple GPU)
