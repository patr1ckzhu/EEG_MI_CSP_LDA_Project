# Hardware Deployment Guide

Generated: 2025-10-14 01:40:04

---

## Performance Summary

| Configuration | Channels | Accuracy | Kappa | Hardware |
|---------------|----------|----------|-------|----------|
| 3-channel | 3 | 60.00% ± 15.07% | 0.198 | Minimum viable BCI (Graz-BCI standard) |
| 8-channel-motor | 8 | 82.22% ± 15.07% | 0.644 | ADS1299 (8-channel) - recommended layout |
| 8-channel-extended | 8 | 73.33% ± 15.07% | 0.465 | ADS1299 (8-channel) - alternative layout |
| 16-channel-full | 16 | 75.56% ± 10.89% | 0.511 | Dual ADS1299 or OpenBCI 16-channel |
| 16-channel-compact | 16 | 75.56% ± 14.74% | 0.510 | Dual ADS1299 or OpenBCI 16-channel |

---

## 3-CHANNEL

**Description**: Minimal configuration - core motor cortex

**Hardware**: Minimum viable BCI (Graz-BCI standard)

**Performance**:
- Accuracy: 60.00% ± 15.07%
- Cohen's Kappa: 0.198
- ROC AUC: 0.530
- Number of channels: 3

**Channel Mapping** (for ADS1299):
```
Channel 1: C3..
Channel 2: Cz..
Channel 3: C4..
```

**CSP Parameters**:
- Number of components: 2
- Regularization: Ledoit-Wolf
- Log transform: Yes

**Electrode Layout**: See `electrode_layout_3-channel.png`

---

## 8-CHANNEL-MOTOR

**Description**: ADS1299 optimal - motor imagery focus

**Hardware**: ADS1299 (8-channel) - recommended layout

**Performance**:
- Accuracy: 82.22% ± 15.07%
- Cohen's Kappa: 0.644
- ROC AUC: 0.911
- Number of channels: 8

**Channel Mapping** (for ADS1299):
```
Channel 1: Fc3.
Channel 2: Fc4.
Channel 3: C3..
Channel 4: Cz..
Channel 5: C4..
Channel 6: Cp3.
Channel 7: Cpz.
Channel 8: Cp4.
```

**CSP Parameters**:
- Number of components: 4
- Regularization: Ledoit-Wolf
- Log transform: Yes

**Electrode Layout**: See `electrode_layout_8-channel-motor.png`

---

## 8-CHANNEL-EXTENDED

**Description**: Extended lateral coverage

**Hardware**: ADS1299 (8-channel) - alternative layout

**Performance**:
- Accuracy: 73.33% ± 15.07%
- Cohen's Kappa: 0.465
- ROC AUC: 0.826
- Number of channels: 8

**Channel Mapping** (for ADS1299):
```
Channel 1: C5..
Channel 2: C3..
Channel 3: C1..
Channel 4: Cz..
Channel 5: C2..
Channel 6: C4..
Channel 7: C6..
Channel 8: Cpz.
```

**CSP Parameters**:
- Number of components: 4
- Regularization: Ledoit-Wolf
- Log transform: Yes

**Electrode Layout**: See `electrode_layout_8-channel-extended.png`

---

## 16-CHANNEL-FULL

**Description**: Full sensorimotor coverage

**Hardware**: Dual ADS1299 or OpenBCI 16-channel

**Performance**:
- Accuracy: 75.56% ± 10.89%
- Cohen's Kappa: 0.511
- ROC AUC: 0.872
- Number of channels: 16

**Channel Mapping** (for ADS1299):
```
Channel 1: Fc5.
Channel 2: Fc3.
Channel 3: Fc1.
Channel 4: Fcz.
Channel 5: Fc2.
Channel 6: Fc4.
Channel 7: Fc6.
Channel 8: C5..
Channel 9: C3..
Channel 10: C1..
Channel 11: Cz..
Channel 12: C2..
Channel 13: C4..
Channel 14: C6..
Channel 15: Cp3.
Channel 16: Cpz.
```

**CSP Parameters**:
- Number of components: 6
- Regularization: Ledoit-Wolf
- Log transform: Yes

**Electrode Layout**: See `electrode_layout_16-channel-full.png`

---

## 16-CHANNEL-COMPACT

**Description**: Compact sensorimotor array

**Hardware**: Dual ADS1299 or OpenBCI 16-channel

**Performance**:
- Accuracy: 75.56% ± 14.74%
- Cohen's Kappa: 0.510
- ROC AUC: 0.830
- Number of channels: 16

**Channel Mapping** (for ADS1299):
```
Channel 1: Fc3.
Channel 2: Fc1.
Channel 3: Fcz.
Channel 4: Fc2.
Channel 5: Fc4.
Channel 6: C5..
Channel 7: C3..
Channel 8: C1..
Channel 9: Cz..
Channel 10: C2..
Channel 11: C4..
Channel 12: C6..
Channel 13: Cp3.
Channel 14: Cp1.
Channel 15: Cpz.
Channel 16: Cp2.
```

**CSP Parameters**:
- Number of components: 6
- Regularization: Ledoit-Wolf
- Log transform: Yes

**Electrode Layout**: See `electrode_layout_16-channel-compact.png`

---

## Recommendations

### For ADS1299 (8-channel)

**Recommended**: `8-channel-motor`
- Accuracy: 82.22%
- Configuration: ['Fc3.', 'Fc4.', 'C3..', 'Cz..', 'C4..', 'Cp3.', 'Cpz.', 'Cp4.']

### For Dual ADS1299 (16-channel)

**Recommended**: `16-channel-full`
- Accuracy: 75.56%
- Configuration: See electrode layout diagram

---

## Next Steps

1. Choose configuration based on your hardware
2. Follow electrode placement diagram
3. Ensure proper skin preparation (impedance <10kΩ)
4. Verify signal quality before recording
5. Collect calibration data (50-100 trials per class)
6. Train model using this exact channel configuration
7. Test online classification performance

