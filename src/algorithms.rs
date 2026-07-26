//! Tilt and center-of-rotation estimation from opposite projection pairs.
//!
//! Geometry: with the rotation axis at (possibly tilted) column `c(y)`, a
//! parallel-beam projection and its 180° opposite mirror into each other,
//! `p_θ(y, x) = p_{θ+180}(y, 2·c(y) − x)`. Flipping the opposite projection
//! horizontally turns that mirror relation into a plain shift per row,
//! `s(y) = w − 1 − 2·c(y)`, so estimating the axis is: find the sub-pixel
//! shift best aligning each row band, convert to an axis column, and fit a
//! line through the per-row columns — slope → tilt, value at mid-height →
//! center of rotation.
//!
//! Row profiles are mean/std normalized before matching (beam intensity
//! differs between the two projections) and the shift error only counts the
//! overlapping columns (no wrap-around artifacts).

use crate::pairs::Pair;
use crate::stack::Stack;
use rayon::prelude::*;

/// How the row bands are sampled and matched.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Inclusive row range used (clamped to the stack).
    pub y_top: usize,
    pub y_bottom: usize,
    /// Every `ystep`-th row starts a band.
    pub ystep: usize,
    /// Rows averaged into each band's profile (SNR boost).
    pub band: usize,
    /// Largest |shift| searched, px (axis assumed within `max_shift/2` of
    /// the detector center).
    pub max_shift: usize,
}

impl Params {
    pub fn defaults_for(height: usize, width: usize) -> Self {
        Self {
            y_top: height / 20,
            y_bottom: height - 1 - height / 20,
            ystep: (height / 60).max(1),
            band: 5,
            max_shift: (width / 3).max(10),
        }
    }
}

/// Axis column estimated at one row band.
#[derive(Clone, Copy, Debug)]
pub struct RowEstimate {
    pub row: f64,
    pub cor: f64,
    /// `false` when the trimmed fit dropped this band as an outlier.
    pub used: bool,
}

/// One algorithm's answer.
#[derive(Clone, Debug)]
pub struct Estimate {
    pub method: String,
    /// Tilt of the rotation axis from the vertical, degrees (positive =
    /// the axis top leans towards larger columns).
    pub tilt_deg: f64,
    /// Axis column at the stack's vertical mid-height, px.
    pub cor: f64,
    /// The fitted line `column = slope · row + intercept` and its quality.
    pub slope: f64,
    pub intercept: f64,
    pub r2: f64,
    pub rows: Vec<RowEstimate>,
}

/// Mean/std-normalized average of `band` rows starting at `y0`.
fn band_profile(plane: &[f32], width: usize, y0: usize, band: usize) -> Vec<f64> {
    let mut profile = vec![0.0f64; width];
    for row in y0..y0 + band {
        let line = &plane[row * width..(row + 1) * width];
        for (acc, v) in profile.iter_mut().zip(line) {
            *acc += f64::from(*v);
        }
    }
    let n = width as f64;
    let mean = profile.iter().sum::<f64>() / n;
    let var = profile.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(1e-12);
    for v in &mut profile {
        *v = (*v - mean) / std;
    }
    profile
}

/// Mean squared difference between `a` and `f` shifted by `s`, over the
/// overlapping columns only.
fn shift_error(a: &[f64], f: &[f64], s: isize) -> f64 {
    let w = a.len() as isize;
    let (from, to) = (0.max(-s), w.min(w - s));
    let count = (to - from).max(1) as f64;
    let mut sum = 0.0;
    for x in from..to {
        let d = a[x as usize] - f[(x + s) as usize];
        sum += d * d;
    }
    sum / count
}

/// The sub-pixel shift minimizing the error of `a` against `f`, searched in
/// `[-max_shift, +max_shift]`, refined with a parabola through the minimum
/// and its neighbors. `None` when the search range is degenerate.
fn best_shift(a: &[f64], f: &[f64], max_shift: usize) -> Option<f64> {
    let w = a.len() as isize;
    let m = (max_shift as isize).min(w - 2).max(1);
    let errors: Vec<f64> = (-m..=m).map(|s| shift_error(a, f, s)).collect();
    let (k, _) = errors
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))?;
    let s = k as isize - m;
    if k == 0 || k == errors.len() - 1 {
        return Some(s as f64);
    }
    let (e0, e1, e2) = (errors[k - 1], errors[k], errors[k + 1]);
    let denom = e0 - 2.0 * e1 + e2;
    let delta = if denom.abs() > 1e-12 {
        (0.5 * (e0 - e2) / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    Some(s as f64 + delta)
}

/// Least-squares line through `(x, y)` points → `(slope, intercept, r2)`.
fn linear_fit(points: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    let n = points.len() as f64;
    if points.len() < 2 {
        return None;
    }
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x, sy + y));
    let (mx, my) = (sx / n, sy / n);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in points {
        sxx += (x - mx) * (x - mx);
        sxy += (x - mx) * (y - my);
        syy += (y - my) * (y - my);
    }
    if sxx.abs() < 1e-12 {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let r2 = if syy.abs() < 1e-12 {
        1.0
    } else {
        (sxy * sxy / (sxx * syy)).min(1.0)
    };
    Some((slope, intercept, r2))
}

/// Trimmed fit: fit all points, drop the worst-residual quarter (keeping at
/// least half), refit. Returns the fit plus the kept flags.
fn trimmed_fit(points: &[(f64, f64)]) -> Option<((f64, f64, f64), Vec<bool>)> {
    let (m, q, _) = linear_fit(points)?;
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&i, &j| {
        let ri = (points[i].1 - (m * points[i].0 + q)).abs();
        let rj = (points[j].1 - (m * points[j].0 + q)).abs();
        ri.total_cmp(&rj)
    });
    let keep_n = (points.len() * 3 / 4).max(points.len().min(2)).max(points.len() / 2);
    let mut used = vec![false; points.len()];
    for &i in &order[..keep_n] {
        used[i] = true;
    }
    let kept: Vec<(f64, f64)> = points
        .iter()
        .zip(&used)
        .filter(|(_, u)| **u)
        .map(|(p, _)| *p)
        .collect();
    Some((linear_fit(&kept)?, used))
}

/// Row bands of `params`, clamped to the stack: `(band_start, center_row)`.
/// Public so the UI can show how many sample points the settings produce.
pub fn bands(height: usize, params: &Params) -> Vec<(usize, f64)> {
    let band = params.band.max(1);
    let top = params.y_top.min(height.saturating_sub(band));
    let bottom = params.y_bottom.min(height - 1);
    let mut out = Vec::new();
    let mut y = top;
    while y + band <= bottom + 1 {
        out.push((y, y as f64 + (band as f64 - 1.0) / 2.0));
        y += params.ystep.max(1);
    }
    out
}

/// Turn per-band axis columns into the final estimate.
fn fit_estimate(
    method: String,
    height: usize,
    columns: Vec<(f64, f64)>,
) -> Result<Estimate, String> {
    if columns.len() < 2 {
        return Err("fewer than 2 row bands gave a usable shift".to_owned());
    }
    let ((slope, intercept, r2), used) =
        trimmed_fit(&columns).ok_or("degenerate row range for the axis fit")?;
    let mid = (height as f64 - 1.0) / 2.0;
    Ok(Estimate {
        method,
        tilt_deg: slope.atan().to_degrees(),
        cor: slope * mid + intercept,
        slope,
        intercept,
        r2,
        rows: columns
            .iter()
            .zip(&used)
            .map(|(&(row, cor), &used)| RowEstimate { row, cor, used })
            .collect(),
    })
}

/// Per-row shift of one flipped opposite pair → axis column per band →
/// line fit. The classic (neutompy/MuhRec-style) estimator, with band
/// averaging, normalized profiles and sub-pixel shifts.
pub fn single_pair(stack: &Stack, pair: &Pair, params: &Params) -> Result<Estimate, String> {
    let (w, h) = (stack.width, stack.height);
    let p0 = stack.plane(pair.i);
    let p180 = stack.plane(pair.j);
    let columns: Vec<(f64, f64)> = bands(h, params)
        .par_iter()
        .filter_map(|&(y0, row)| {
            let a = band_profile(p0, w, y0, params.band.max(1));
            let mut f = band_profile(p180, w, y0, params.band.max(1));
            f.reverse();
            let s = best_shift(&a, &f, params.max_shift)?;
            Some((row, (w as f64 - 1.0 - s) / 2.0))
        })
        .collect();
    fit_estimate(
        format!("0/180 pair {}", pair.label()),
        h,
        columns,
    )
}

/// Consensus over every opposite pair: for each band the shift error is
/// summed across all pairs before the minimum is taken, so a single noisy
/// pair cannot pull the axis. The strongest estimator when the scan has
/// many opposite pairs.
pub fn multi_pair(stack: &Stack, pairs: &[Pair], params: &Params) -> Result<Estimate, String> {
    if pairs.is_empty() {
        return Err("no opposite (0/180) pairs in the angle list".to_owned());
    }
    let (w, h) = (stack.width, stack.height);
    // Cap the work: up to 32 pairs, spread evenly over the list.
    let take: Vec<&Pair> = if pairs.len() <= 32 {
        pairs.iter().collect()
    } else {
        (0..32)
            .map(|k| &pairs[k * pairs.len() / 32])
            .collect()
    };
    let columns: Vec<(f64, f64)> = bands(h, params)
        .par_iter()
        .filter_map(|&(y0, row)| {
            let band = params.band.max(1);
            let profiles: Vec<(Vec<f64>, Vec<f64>)> = take
                .iter()
                .map(|p| {
                    let a = band_profile(stack.plane(p.i), w, y0, band);
                    let mut f = band_profile(stack.plane(p.j), w, y0, band);
                    f.reverse();
                    (a, f)
                })
                .collect();
            let m = (params.max_shift as isize).min(w as isize - 2).max(1);
            let errors: Vec<f64> = (-m..=m)
                .map(|s| {
                    profiles
                        .iter()
                        .map(|(a, f)| shift_error(a, f, s))
                        .sum::<f64>()
                })
                .collect();
            let (k, _) = errors
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))?;
            let s0 = k as isize - m;
            let s = if k == 0 || k == errors.len() - 1 {
                s0 as f64
            } else {
                let (e0, e1, e2) = (errors[k - 1], errors[k], errors[k + 1]);
                let denom = e0 - 2.0 * e1 + e2;
                let delta = if denom.abs() > 1e-12 {
                    (0.5 * (e0 - e2) / denom).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                s0 as f64 + delta
            };
            Some((row, (w as f64 - 1.0 - s) / 2.0))
        })
        .collect();
    fit_estimate(
        format!("consensus of {} opposite pairs", take.len()),
        h,
        columns,
    )
}

/// Whole-range single shift of one pair: no tilt, one global center — a
/// quick cross-check of the fitted estimates.
pub fn global_cor(stack: &Stack, pair: &Pair, params: &Params) -> Result<f64, String> {
    let (w, h) = (stack.width, stack.height);
    let top = params.y_top.min(h - 1);
    let bottom = params.y_bottom.min(h - 1).max(top + 1);
    let a = band_profile(stack.plane(pair.i), w, top, bottom - top + 1);
    let mut f = band_profile(stack.plane(pair.j), w, top, bottom - top + 1);
    f.reverse();
    let s = best_shift(&a, &f, params.max_shift)
        .ok_or("degenerate search range for the global shift")?;
    Ok((w as f64 - 1.0 - s) / 2.0)
}

/// Straighten a measured axis tilt: rotate every projection so the fitted
/// axis becomes vertical. The sign convention (`+tilt_deg` with this
/// rotation matrix) is locked by the measure-rotate-measure test.
pub fn apply_tilt(stack: &mut Stack, tilt_deg: f64) {
    rotate_stack(stack, tilt_deg);
}

/// Rotate every projection by `angle_deg` about the image center (bilinear,
/// edge-clamped).
pub fn rotate_stack(stack: &mut Stack, angle_deg: f64) {
    let (n, h, w) = (stack.n, stack.height, stack.width);
    let (sin, cos) = angle_deg.to_radians().sin_cos();
    let (cx, cy) = ((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0);
    let size = h * w;
    stack
        .data
        .par_chunks_mut(size)
        .take(n)
        .for_each(|plane| {
            let src = plane.to_vec();
            for y in 0..h {
                for x in 0..w {
                    // Inverse-rotate the target pixel into the source.
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    let sx = (cos * dx + sin * dy + cx).clamp(0.0, w as f64 - 1.0);
                    let sy = (-sin * dx + cos * dy + cy).clamp(0.0, h as f64 - 1.0);
                    let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
                    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                    let (fx, fy) = (sx - x0 as f64, sy - y0 as f64);
                    let v00 = f64::from(src[y0 * w + x0]);
                    let v01 = f64::from(src[y0 * w + x1]);
                    let v10 = f64::from(src[y1 * w + x0]);
                    let v11 = f64::from(src[y1 * w + x1]);
                    let v = v00 * (1.0 - fx) * (1.0 - fy)
                        + v01 * fx * (1.0 - fy)
                        + v10 * (1.0 - fx) * fy
                        + v11 * fx * fy;
                    plane[y * w + x] = v as f32;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairs::opposite_pairs;
    use std::path::PathBuf;

    /// Tiny deterministic pseudo-random stream (no external crates).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as f64 / (1u64 << 31) as f64
        }
    }

    /// Linear interpolation of a row at fractional column `x`.
    fn sample(row: &[f32], x: f64) -> f32 {
        let w = row.len();
        let xc = x.clamp(0.0, w as f64 - 1.0);
        let x0 = xc.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let f = xc - x0 as f64;
        (f64::from(row[x0]) * (1.0 - f) + f64::from(row[x1]) * f) as f32
    }

    /// A synthetic stack whose 0° projection is smooth noise and whose 180°
    /// projection mirrors it about a tilted axis `c(y) = c0 + m·(y - mid)`.
    fn synthetic(c0: f64, m: f64, n_pairs: usize) -> Stack {
        let (h, w) = (120, 200);
        let mut rng = Lcg(7);
        let mut planes: Vec<Vec<f32>> = Vec::new();
        let mut angles = Vec::new();
        for k in 0..n_pairs {
            // Smooth pseudo-random image: sum of a few sinusoids with
            // random phases (different per pair).
            let (a1, a2, a3) = (rng.next() * 6.0, rng.next() * 6.0, rng.next() * 6.0);
            let mut p0 = vec![0.0f32; h * w];
            for y in 0..h {
                for x in 0..w {
                    let xf = x as f64 / w as f64;
                    let yf = y as f64 / h as f64;
                    p0[y * w + x] = ((xf * 19.0 + a1).sin()
                        + (xf * 7.0 + yf * 3.0 + a2).sin()
                        + (xf * 31.0 + a3).cos() * 0.5) as f32;
                }
            }
            let mid = (h as f64 - 1.0) / 2.0;
            let mut p180 = vec![0.0f32; h * w];
            for y in 0..h {
                let c = c0 + m * (y as f64 - mid);
                let row0 = &p0[y * w..(y + 1) * w];
                for x in 0..w {
                    p180[y * w + x] = sample(row0, 2.0 * c - x as f64);
                }
            }
            planes.push(p0);
            planes.push(p180);
            angles.push(k as f64 * 2.0);
            angles.push(k as f64 * 2.0 + 180.0);
        }
        Stack {
            path: PathBuf::new(),
            data: planes.concat(),
            n: 2 * n_pairs,
            height: h,
            width: w,
            angles_deg: angles,
            center_of_rotation: None,
            previous_corrections: Vec::new(),
        }
    }

    #[test]
    fn single_pair_recovers_center_and_tilt() {
        let (c0, m) = (91.3, 0.02); // tilt ≈ 1.15°
        let stack = synthetic(c0, m, 1);
        let pairs = opposite_pairs(&stack.angles_deg, 0.1);
        let params = Params::defaults_for(stack.height, stack.width);
        let e = single_pair(&stack, &pairs[0], &params).unwrap();
        assert!((e.cor - c0).abs() < 0.5, "cor {} vs {c0}", e.cor);
        let true_tilt = m.atan().to_degrees();
        assert!(
            (e.tilt_deg - true_tilt).abs() < 0.1,
            "tilt {} vs {true_tilt}",
            e.tilt_deg
        );
        assert!(e.r2 > 0.95, "r2 {}", e.r2);
    }

    #[test]
    fn multi_pair_recovers_center_and_tilt() {
        let (c0, m) = (104.6, -0.015);
        let stack = synthetic(c0, m, 4);
        let pairs = opposite_pairs(&stack.angles_deg, 0.1);
        assert!(pairs.len() >= 4);
        let params = Params::defaults_for(stack.height, stack.width);
        let e = multi_pair(&stack, &pairs, &params).unwrap();
        assert!((e.cor - c0).abs() < 0.5, "cor {} vs {c0}", e.cor);
        let true_tilt = m.atan().to_degrees();
        assert!(
            (e.tilt_deg - true_tilt).abs() < 0.1,
            "tilt {} vs {true_tilt}",
            e.tilt_deg
        );
    }

    #[test]
    fn global_cor_matches_on_untilted_stack() {
        let c0 = 88.25;
        let stack = synthetic(c0, 0.0, 1);
        let pairs = opposite_pairs(&stack.angles_deg, 0.1);
        let params = Params::defaults_for(stack.height, stack.width);
        let c = global_cor(&stack, &pairs[0], &params).unwrap();
        assert!((c - c0).abs() < 0.5, "global cor {c} vs {c0}");
    }

    #[test]
    fn rotation_is_exact_on_a_linear_ramp() {
        // Bilinear interpolation reproduces a linear function exactly, so
        // rotating f(x,y) = 2x + 3y must give f(R⁻¹(x,y)) at every interior
        // pixel.
        let (h, w) = (41, 53);
        let mut stack = Stack {
            path: PathBuf::new(),
            data: (0..h * w)
                .map(|i| (2 * (i % w) + 3 * (i / w)) as f32)
                .collect(),
            n: 1,
            height: h,
            width: w,
            angles_deg: vec![0.0],
            center_of_rotation: None,
            previous_corrections: Vec::new(),
        };
        let angle: f64 = 1.7;
        rotate_stack(&mut stack, angle);
        let (sin, cos) = angle.to_radians().sin_cos();
        let (cx, cy) = ((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0);
        for y in 10..h - 10 {
            for x in 10..w - 10 {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let sx = cos * dx + sin * dy + cx;
                let sy = -sin * dx + cos * dy + cy;
                let expected = 2.0 * sx + 3.0 * sy;
                let got = f64::from(stack.data[y * w + x]);
                assert!(
                    (got - expected).abs() < 1e-3,
                    "({x},{y}): {got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn measure_rotate_measure_converges() {
        // Apply the measured correction and re-measure: the residual tilt
        // must be an order of magnitude smaller.
        let (c0, m) = (97.0, 0.03);
        let mut stack = synthetic(c0, m, 1);
        let pairs = opposite_pairs(&stack.angles_deg, 0.1);
        let params = Params::defaults_for(stack.height, stack.width);
        let before = single_pair(&stack, &pairs[0], &params).unwrap();
        apply_tilt(&mut stack, before.tilt_deg);
        let after = single_pair(&stack, &pairs[0], &params).unwrap();
        assert!(
            after.tilt_deg.abs() < before.tilt_deg.abs() / 5.0,
            "tilt {} -> {}",
            before.tilt_deg,
            after.tilt_deg
        );
    }
}
