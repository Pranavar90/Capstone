# Model Limitations & Future Improvements

## Limitations
The model’s main limitations arise from:
- **Inaccurate Atmospheric Light Estimation**: usage leads to color shifts.
- **Aggressive Transmission Recovery**: Causes over-contrast in dense haze scenarios.
- **Artifacts**: 
    - Grid artifacts indicate issues with patch-based inference or upsampling.
    - Noise amplification suggests the absence of edge-aware smoothing.

## Proposed Solutions
1.  **Dehazing Strength**: Address by clamping or adaptively learning dehazing strength.
2.  **Color Constancy**: Introduce color constancy losses (e.g., gray-world or channel balance).
3.  **Scene Awareness**: Add scene-aware gating for indoor vs. outdoor conditions.
4.  **Upsampling**: Replace transposed convolutions with artifact-free upsampling methods.

## Evaluation Strategy
Future evaluation should rely on **no-reference image quality metrics** and **downstream task performance** rather than visual inspection alone.
