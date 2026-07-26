//! The gridrec (FBP) test slice: reconstruct one slice at the adopted
//! center — and a small sweep of centers around it — so the tilt and the
//! center of rotation can be judged on an actual reconstruction, not just
//! on the 0/180 overlap.
//!
//! The reconstruction itself runs in the same pixi Python environment the
//! main application uses, through algotom's `gridrec_reconstruction`.

use crate::npy;
use crate::stack::Stack;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};

/// The interpreter of the pixi environment that has algotom installed
/// (same one rust_ct_reconstruction reconstructs with).
pub const RECON_PYTHON: &str =
    "/SNS/VENUS/shared/software/git/all_ct_reconstruction_development/.pixi/envs/default/bin/python";

const SCRIPT: &str = r#"
import json
import sys

import numpy as np
import algotom.rec.reconstruction as rec_mod

sino_file, spec_file, out_file = sys.argv[1:4]
with open(spec_file) as f:
    spec = json.load(f)
sinos = np.load(sino_file)  # (n_tilts * n_rows, n_angles, width)
sinos = np.nan_to_num(sinos, nan=0.0, posinf=0.0, neginf=0.0)
angles = np.array(spec["angles_rad"], dtype=np.float32)
if spec["apply_log"]:
    sinos = (-np.log(np.clip(sinos, 1e-6, None))).astype(np.float32)
outs = []
for sino in sinos:
    per_center = []
    for c in spec["centers"]:
        r = rec_mod.gridrec_reconstruction(
            sino,
            float(c),
            angles=angles,
            ratio=1.0,
            filter_name="shepp",
            apply_log=False,
            pad=100,
            filter_par=0.9,
            ncore=None,
        )
        per_center.append(np.asarray(r, dtype=np.float32))
    outs.append(np.stack(per_center))
np.save(out_file, np.stack(outs))  # (n_tilts, n_centers, size, size)
"#;

/// One test reconstruction: a slice per (tilt, row, center) combination,
/// each `size×size`. Two rows — one near the top, one near the bottom —
/// are the real tilt test: a wrong tilt makes the best center differ
/// between the two heights.
pub struct ReconTest {
    pub rows: Vec<usize>,
    pub tilts: Vec<f64>,
    pub centers: Vec<f64>,
    pub size: usize,
    /// `tilts.len() * rows.len() * centers.len() * size * size` f32 values.
    pub slices: Vec<f32>,
}

impl ReconTest {
    pub fn slice(&self, tilt: usize, row: usize, center: usize) -> &[f32] {
        let n = self.size * self.size;
        let k = (tilt * self.rows.len() + row) * self.centers.len() + center;
        &self.slices[k * n..(k + 1) * n]
    }
}

/// Row `row` of every projection with a finite angle, sampled through the
/// (to-be-applied) tilt rotation — bilinear, like `apply_tilt` — so the
/// test reconstructs what the corrected stack would look like without
/// rotating the whole stack. Returns the flat `(n_used, width)` sinogram
/// and the matching angles in radians.
pub fn extract_sinogram(stack: &Stack, row: usize, tilt_deg: f64) -> (Vec<f32>, Vec<f64>) {
    let (h, w) = (stack.height, stack.width);
    let row = row.min(h - 1);
    let (sin, cos) = tilt_deg.to_radians().sin_cos();
    let (cx, cy) = ((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0);
    let mut sino = Vec::new();
    let mut angles = Vec::new();
    for i in 0..stack.n {
        let a = stack.angles_deg[i];
        if !a.is_finite() {
            continue;
        }
        let plane = stack.plane(i);
        let dy = row as f64 - cy;
        for x in 0..w {
            let dx = x as f64 - cx;
            let sx = (cos * dx + sin * dy + cx).clamp(0.0, w as f64 - 1.0);
            let sy = (-sin * dx + cos * dy + cy).clamp(0.0, h as f64 - 1.0);
            let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            let (fx, fy) = (sx - x0 as f64, sy - y0 as f64);
            let v = f64::from(plane[y0 * w + x0]) * (1.0 - fx) * (1.0 - fy)
                + f64::from(plane[y0 * w + x1]) * fx * (1.0 - fy)
                + f64::from(plane[y1 * w + x0]) * (1.0 - fx) * fy
                + f64::from(plane[y1 * w + x1]) * fx * fy;
            // A single NaN pixel (0/0 during normalization) would spread
            // through gridrec's FFT to the whole slice — zero it instead.
            sino.push(if v.is_finite() { v as f32 } else { 0.0 });
        }
        angles.push(a.to_radians());
    }
    (sino, angles)
}

pub struct ReconTestJob {
    rx: Receiver<Result<ReconTest, String>>,
}

impl ReconTestJob {
    pub fn start(
        stack: std::sync::Arc<Stack>,
        rows: Vec<usize>,
        tilts: Vec<f64>,
        centers: Vec<f64>,
        apply_log: bool,
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(run(&stack, rows, tilts, centers, apply_log));
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<ReconTest, String>> {
        self.rx.try_recv().ok()
    }
}

fn run(
    stack: &Stack,
    rows: Vec<usize>,
    tilts: Vec<f64>,
    centers: Vec<f64>,
    apply_log: bool,
) -> Result<ReconTest, String> {
    if rows.is_empty() || tilts.is_empty() || centers.is_empty() {
        return Err("no row, tilt or center values to reconstruct".to_owned());
    }
    // One sinogram per (tilt, row) combination, stacked in that order.
    let mut sinos = Vec::new();
    let mut n_angles = 0;
    for &tilt in &tilts {
        for &row in &rows {
            let (sino, angles) = extract_sinogram(stack, row, tilt);
            if angles.len() < 3 {
                return Err("fewer than 3 projections carry an angle".to_owned());
            }
            n_angles = angles.len();
            sinos.extend(sino);
        }
    }
    let angles: Vec<f64> = stack
        .angles_deg
        .iter()
        .filter(|a| a.is_finite())
        .map(|a| a.to_radians())
        .collect();
    let scratch: PathBuf =
        std::env::temp_dir().join(format!("tilt_cor_gridrec_{}", std::process::id()));
    std::fs::create_dir_all(&scratch)
        .map_err(|e| format!("cannot create {}: {e}", scratch.display()))?;
    let sino_file = scratch.join("sino.npy");
    let spec_file = scratch.join("spec.json");
    let script_file = scratch.join("gridrec_test.py");
    let out_file = scratch.join("out.npy");
    let result = (|| {
        npy::write_f32(
            &sino_file,
            &[tilts.len() * rows.len(), n_angles, stack.width],
            &sinos,
        )?;
        let spec = serde_json::json!({
            "angles_rad": angles,
            "centers": centers,
            "apply_log": apply_log,
        });
        std::fs::write(&spec_file, spec.to_string())
            .map_err(|e| format!("write {}: {e}", spec_file.display()))?;
        std::fs::write(&script_file, SCRIPT)
            .map_err(|e| format!("write {}: {e}", script_file.display()))?;
        let output = std::process::Command::new(RECON_PYTHON)
            .arg(&script_file)
            .arg(&sino_file)
            .arg(&spec_file)
            .arg(&out_file)
            .output()
            .map_err(|e| format!("cannot launch {RECON_PYTHON}: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: Vec<&str> = stderr.trim().lines().rev().take(4).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(format!(
                "gridrec failed ({}): {}",
                output.status,
                tail.join(" | ")
            ));
        }
        let (shape, slices) = npy::read_f32(&out_file)?;
        let [tr, c, s1, s2] = shape[..] else {
            return Err(format!("unexpected reconstruction shape {shape:?}"));
        };
        if tr != tilts.len() * rows.len() || c != centers.len() || s1 != s2 {
            return Err(format!(
                "unexpected reconstruction shape {shape:?} for {} tilts × {} rows × {} centers",
                tilts.len(),
                rows.len(),
                centers.len()
            ));
        }
        Ok(ReconTest {
            rows,
            tilts,
            centers,
            size: s1,
            slices,
        })
    })();
    let _ = std::fs::remove_dir_all(&scratch);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack(h: usize, w: usize, n: usize) -> Stack {
        Stack {
            path: PathBuf::new(),
            // Plane i: f(x, y) = x + 10·y + 1000·i — linear, so the tilted
            // extraction is analytic.
            data: (0..n)
                .flat_map(|i| {
                    (0..h * w).map(move |p| ((p % w) + 10 * (p / w) + 1000 * i) as f32)
                })
                .collect(),
            n,
            height: h,
            width: w,
            angles_deg: (0..n).map(|i| i as f64).collect(),
            center_of_rotation: None,
            previous_corrections: Vec::new(),
        }
    }

    #[test]
    fn untilted_extraction_is_the_raw_row() {
        let s = stack(20, 30, 3);
        let (sino, angles) = extract_sinogram(&s, 7, 0.0);
        assert_eq!(angles.len(), 3);
        assert_eq!(sino.len(), 3 * 30);
        assert_eq!(&sino[..30], &s.plane(0)[7 * 30..8 * 30]);
    }

    #[test]
    fn tilted_extraction_matches_the_analytic_rotation() {
        let (h, w) = (21, 31);
        let s = stack(h, w, 1);
        let row = 10; // the center row: dy = 0, so sy = cy − sin·dx
        let tilt: f64 = 2.0;
        let (sino, _) = extract_sinogram(&s, row, tilt);
        let (sin, cos) = tilt.to_radians().sin_cos();
        let (cx, cy) = ((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0);
        for x in 5..w - 5 {
            let dx = x as f64 - cx;
            let expected = (cos * dx + cx) + 10.0 * (-sin * dx + cy);
            assert!(
                (f64::from(sino[x]) - expected).abs() < 1e-3,
                "x={x}: {} vs {expected}",
                sino[x]
            );
        }
    }

    /// The full pipeline against the real pixi python — run manually with
    /// `cargo test end_to_end_gridrec -- --ignored`.
    #[test]
    #[ignore]
    fn end_to_end_gridrec() {
        let (h, w, n) = (64, 96, 24);
        // A disk phantom: every projection of a centered disk is the same
        // chord-length profile, so any angle set reconstructs it.
        let cx = (w as f64 - 1.0) / 2.0 + 3.0;
        let profile: Vec<f32> = (0..w)
            .map(|x| {
                let d = x as f64 - cx;
                (2.0 * (20.0f64.powi(2) - d * d).max(0.0).sqrt()) as f32
            })
            .collect();
        let stack = Stack {
            path: PathBuf::new(),
            data: (0..n)
                .flat_map(|_| {
                    (0..h).flat_map(|_| profile.clone()).collect::<Vec<f32>>()
                })
                .collect(),
            n,
            height: h,
            width: w,
            angles_deg: (0..n).map(|i| i as f64 * 180.0 / n as f64).collect(),
            center_of_rotation: None,
            previous_corrections: Vec::new(),
        };
        let test = run(
            &stack,
            vec![5, 58],
            vec![0.0, 0.2],
            vec![cx - 1.0, cx, cx + 1.0],
            false,
        )
        .unwrap();
        assert_eq!(test.rows, vec![5, 58]);
        assert_eq!(test.tilts.len(), 2);
        assert_eq!(test.centers.len(), 3);
        assert_eq!(test.size, w);
        assert_eq!(test.slices.len(), 2 * 2 * 3 * w * w);
        let center_px = test.slice(0, 0, 1)[(w / 2) * w + w / 2];
        assert!(
            center_px.is_finite() && center_px > 0.1,
            "disk center reconstructed to {center_px}"
        );
    }

    #[test]
    fn nan_pixels_become_zero_in_the_sinogram() {
        let mut s = stack(8, 9, 1);
        s.data[3 * 9 + 4] = f32::NAN;
        let (sino, _) = extract_sinogram(&s, 3, 0.0);
        assert!(sino.iter().all(|v| v.is_finite()));
        assert_eq!(sino[4], 0.0);
    }

    #[test]
    fn nan_angles_are_skipped() {
        let mut s = stack(8, 9, 3);
        s.angles_deg[1] = f64::NAN;
        let (sino, angles) = extract_sinogram(&s, 2, 0.0);
        assert_eq!(angles.len(), 2);
        assert_eq!(sino.len(), 2 * 9);
    }
}
