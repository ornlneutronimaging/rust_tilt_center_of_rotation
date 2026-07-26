//! Finding the opposite (180 degrees apart) projection pairs the tilt and
//! center-of-rotation estimation is built on.

/// One opposite pair: projection indices and their angles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    pub i: usize,
    pub j: usize,
    pub angle_i: f64,
    pub angle_j: f64,
    /// `|angle_j - angle_i - 180|`, degrees.
    pub error_deg: f64,
}

impl Pair {
    pub fn label(&self) -> String {
        format!(
            "{:.2}° / {:.2}°  (Δ {:.2}°)",
            self.angle_i, self.angle_j, self.error_deg
        )
    }
}

/// Every pair of projections 180° apart within `tol_deg`, sorted by how
/// exactly opposite they are (then by angle). Angles are compared modulo
/// 360 so a 0–360 scan pairs 350° with 170° too; NaN angles never pair.
pub fn opposite_pairs(angles_deg: &[f64], tol_deg: f64) -> Vec<Pair> {
    let mut pairs = Vec::new();
    for i in 0..angles_deg.len() {
        let a = angles_deg[i];
        if !a.is_finite() {
            continue;
        }
        for j in (i + 1)..angles_deg.len() {
            let b = angles_deg[j];
            if !b.is_finite() {
                continue;
            }
            let d = (b - a).rem_euclid(360.0);
            let error = (d - 180.0).abs();
            if error <= tol_deg {
                pairs.push(Pair {
                    i,
                    j,
                    angle_i: a,
                    angle_j: b,
                    error_deg: error,
                });
            }
        }
    }
    pairs.sort_by(|p, q| {
        p.error_deg
            .total_cmp(&q.error_deg)
            .then(p.angle_i.total_cmp(&q.angle_i))
    });
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_exact_and_near_pairs() {
        let angles = [0.0, 90.0, 180.0, 270.1];
        let pairs = opposite_pairs(&angles, 0.5);
        assert_eq!(pairs.len(), 2);
        assert_eq!((pairs[0].i, pairs[0].j), (0, 2)); // exact first
        assert_eq!((pairs[1].i, pairs[1].j), (1, 3));
        assert!((pairs[1].error_deg - 0.1).abs() < 1e-9);
    }

    #[test]
    fn wraps_modulo_360() {
        let angles = [350.0, 170.0];
        let pairs = opposite_pairs(&angles, 0.1);
        assert_eq!(pairs.len(), 1);
        assert_eq!((pairs[0].i, pairs[0].j), (0, 1));
    }

    #[test]
    fn ignores_nan_and_unpaired() {
        let angles = [0.0, f64::NAN, 90.0];
        assert!(opposite_pairs(&angles, 0.5).is_empty());
    }
}
