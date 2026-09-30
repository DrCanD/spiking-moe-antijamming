# Classifier tail on KV260

This is the measured feature-finalisation and Random Forest kernel. It includes both frozen forests, the HLS source, **44 native test frames**, the board driver and the paced power protocol.

From the repository root:

```bash
python hardware/kv260/classifier_tail/build_classifier.py --native-only
python hardware/kv260/classifier_tail/extract_vectors.py
python analysis/energy/classifier_tail.py
```

Native verification checks all nine float64 features and both forests over all twenty subframe rotations. The recorded energy is **1.660 ± 0.209 µJ/inference** at 7,045 inferences/s and **1.815 ± 0.434 µJ/inference** at 3,153 inferences/s.

To build the FPGA design in an AMD/Vitis 2025.2 environment:

```bash
python hardware/kv260/classifier_tail/build_classifier.py --jobs 4
```

Install the generated classifier firmware on the board before these commands. The gate and classifier kernels use the same AXI address and are measured in separate firmware campaigns.

```bash
sudo python3 hardware/kv260/classifier_tail/board/verify_classifier.py hardware/kv260/classifier_tail/vectors --spec hardware/kv260/classifier_tail/spec.json
sudo python3 hardware/kv260/classifier_tail/board/measure_classifier.py hardware/kv260/classifier_tail/vectors --block 60 --repeats 3 --out runs/classifier_power.json
```

The generator can rebuild the exported forest header and native test file:

```bash
python hardware/kv260/classifier_tail/golden/generate_forest.py hardware/kv260/classifier_tail/spec.json hardware/kv260/classifier_tail/vectors
```
