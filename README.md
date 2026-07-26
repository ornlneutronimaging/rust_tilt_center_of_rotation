# tilt_center_of_rotation

Standalone estimation of the **rotation-axis tilt** and **center of
rotation** of a CT projection stack, for the cases where the in-pipeline
correction struggles. Works on the `rust_ct_reconstruction` checkpoint HDF5
format (`/projections` as an f32 `(n, h, w)` stack, `/angles_deg`,
`/center_of_rotation`).

## What it does

The estimation is built on parallel-beam opposite-pair geometry: with the
rotation axis at column `c(y)`, a projection and its 180° opposite mirror
into each other, `p_θ(y, x) = p_{θ+180}(y, 2·c(y) − x)`. Flipping the
opposite projection horizontally turns the mirror into a per-row shift, so:

1. **Selected pair** — for each row band (averaged rows, mean/std-normalized
   profiles, overlap-only error), find the sub-pixel shift best aligning the
   0° and flipped 180° projections; convert each shift to an axis column;
   robust (trimmed) line fit through the columns. Slope → tilt, value at
   mid-height → center of rotation.
2. **All pairs (consensus)** — the same, with the match error summed over
   *every* opposite pair in the scan before taking the minimum; a single
   noisy pair cannot pull the axis.
3. **Whole-image single shift** — one global shift of the selected pair,
   shown as a cross-check of the fitted values.

The difference view (|0° − 180° mirrored about the adopted axis|) goes flat
when the axis is right; the per-row axis positions and the fit are plotted
below it.

**Test with gridrec (FBP)** reconstructs one slice at the adopted center —
and a sweep of centers around it — through algotom's
`gridrec_reconstruction` (the same pixi Python environment
`rust_ct_reconstruction` reconstructs with). The sinogram row is extracted
through the to-be-applied tilt rotation, so flipping the "use the adopted
tilt" toggle compares the reconstruction with and without the correction,
and flipping through the center sweep shows which center is sharpest.

**Apply the tilt & save** rotates every projection so the fitted axis is
vertical (bilinear, edge-clamped), then writes `/projections`,
`/center_of_rotation` and a JSON record of the correction
(`metadata/tilt_center_of_rotation`, one line per applied correction) back
into the file — which `rust_ct_reconstruction` picks up when it reloads the
checkpoint.

## Usage

```
tilt_center_of_rotation [checkpoint.h5] [--called-from-app]
```

Without a path, load a file from the 📂 button. Launched from the
reconstruction screen of `rust_ct_reconstruction`, the checkpoint is passed
automatically and the main app reloads it when this tool closes.

## Build

```
cargo build --release   # target/release/tilt_center_of_rotation
cargo test              # synthetic-stack tests lock the sign conventions
