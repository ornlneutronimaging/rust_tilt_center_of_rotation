//! Reading and writing the ct_reconstruction checkpoint HDF5: the sample
//! projections at the root (`/projections`, one f32 plane per angle), the
//! angle list, and the center-of-rotation scalar the reconstruction uses.
//!
//! Only the datasets this tool needs are touched; everything else in the
//! file (ob/dc stacks, provenance) is left as-is.

use std::path::{Path, PathBuf};

/// The projection stack as loaded from a checkpoint.
pub struct Stack {
    pub path: PathBuf,
    /// `n * height * width` f32 planes, angle-ordered like the file.
    pub data: Vec<f32>,
    pub n: usize,
    pub height: usize,
    pub width: usize,
    /// One angle (degrees) per projection; NaN when the file has none.
    pub angles_deg: Vec<f64>,
    /// `/center_of_rotation` (px), when the file carries one.
    pub center_of_rotation: Option<f64>,
    /// Previous runs of this tool recorded in the metadata (JSON strings),
    /// newest last — shown so an already-applied tilt is not applied twice.
    pub previous_corrections: Vec<String>,
}

impl Stack {
    pub fn plane(&self, i: usize) -> &[f32] {
        let size = self.height * self.width;
        &self.data[i * size..(i + 1) * size]
    }
}

pub fn load(path: &Path) -> Result<Stack, String> {
    let file = hdf5_metno::File::open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let ds = file
        .dataset("projections")
        .map_err(|e| format!("{} has no /projections dataset: {e}", path.display()))?;
    let shape = ds.shape();
    let [n, height, width] = shape[..] else {
        return Err(format!(
            "/projections is not a 3D stack (shape {shape:?})"
        ));
    };
    if n == 0 || height == 0 || width == 0 {
        return Err("the projection stack is empty".to_owned());
    }
    let data: Vec<f32> = ds
        .read_raw()
        .map_err(|e| format!("read /projections: {e}"))?;
    let angles_deg: Vec<f64> = match file.dataset("angles_deg") {
        Ok(ds) => ds
            .read_raw()
            .map_err(|e| format!("read /angles_deg: {e}"))?,
        Err(_) => vec![f64::NAN; n],
    };
    if angles_deg.len() != n {
        return Err(format!(
            "{} angles for {n} projections",
            angles_deg.len()
        ));
    }
    let center_of_rotation = file
        .dataset("center_of_rotation")
        .and_then(|ds| ds.read_scalar::<f64>())
        .ok();
    let previous_corrections = match file.group("metadata") {
        Ok(meta) => meta
            .dataset(RESULT_METADATA)
            .and_then(|ds| ds.read_scalar::<hdf5_metno::types::VarLenUnicode>())
            .map(|v| {
                v.as_str()
                    .lines()
                    .map(str::to_owned)
                    .filter(|l| !l.trim().is_empty())
                    .collect()
            })
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    Ok(Stack {
        path: path.to_path_buf(),
        data,
        n,
        height,
        width,
        angles_deg,
        center_of_rotation,
        previous_corrections,
    })
}

/// Metadata dataset the applied corrections are appended to (one JSON
/// object per line, newest last).
pub const RESULT_METADATA: &str = "tilt_center_of_rotation";

/// Write the corrected stack back: replace `/projections`, set
/// `/center_of_rotation`, and append `record` (a JSON object) to the
/// tool's metadata dataset.
pub fn save(stack: &Stack, cor: f64, record: &str) -> Result<(), String> {
    use hdf5_metno::types::VarLenUnicode;
    let file = hdf5_metno::File::open_rw(&stack.path)
        .map_err(|e| format!("cannot open {} for writing: {e}", stack.path.display()))?;
    if file.dataset("projections").is_ok() {
        file.unlink("projections")
            .map_err(|e| format!("replace /projections: {e}"))?;
    }
    file.new_dataset::<f32>()
        .shape((stack.n, stack.height, stack.width))
        .create("projections")
        .and_then(|ds| ds.write_raw(&stack.data))
        .map_err(|e| format!("write /projections: {e}"))?;
    if file.dataset("center_of_rotation").is_ok() {
        file.unlink("center_of_rotation")
            .map_err(|e| format!("replace /center_of_rotation: {e}"))?;
    }
    file.new_dataset::<f64>()
        .create("center_of_rotation")
        .and_then(|ds| ds.write_scalar(&cor))
        .map_err(|e| format!("write /center_of_rotation: {e}"))?;

    let metadata = match file.group("metadata") {
        Ok(group) => group,
        Err(_) => file
            .create_group("metadata")
            .map_err(|e| format!("create metadata group: {e}"))?,
    };
    let mut history: Vec<String> = stack.previous_corrections.clone();
    history.push(record.to_owned());
    let text = history.join("\n");
    if metadata.dataset(RESULT_METADATA).is_ok() {
        metadata
            .unlink(RESULT_METADATA)
            .map_err(|e| format!("replace metadata/{RESULT_METADATA}: {e}"))?;
    }
    let value: VarLenUnicode = text.parse().unwrap_or_default();
    metadata
        .new_dataset::<VarLenUnicode>()
        .create(RESULT_METADATA)
        .and_then(|ds| ds.write_scalar(&value))
        .map_err(|e| format!("write metadata/{RESULT_METADATA}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tilt_cor_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn write_checkpoint(path: &Path, n: usize, h: usize, w: usize) {
        let file = hdf5_metno::File::create(path).unwrap();
        let data: Vec<f32> = (0..n * h * w).map(|i| i as f32).collect();
        file.new_dataset::<f32>()
            .shape((n, h, w))
            .create("projections")
            .and_then(|ds| ds.write_raw(&data))
            .unwrap();
        let angles: Vec<f64> = (0..n).map(|i| i as f64 * 180.0).collect();
        file.new_dataset::<f64>()
            .shape(n)
            .create("angles_deg")
            .and_then(|ds| ds.write_raw(&angles))
            .unwrap();
    }

    #[test]
    fn load_save_round_trip() {
        let path = scratch("roundtrip.h5");
        write_checkpoint(&path, 2, 4, 5);
        let mut stack = load(&path).unwrap();
        assert_eq!((stack.n, stack.height, stack.width), (2, 4, 5));
        assert_eq!(stack.angles_deg, vec![0.0, 180.0]);
        assert_eq!(stack.center_of_rotation, None);
        assert!(stack.previous_corrections.is_empty());

        stack.data[0] = 42.0;
        save(&stack, 2.25, "{\"tilt_deg\":0.5}").unwrap();
        let reloaded = load(&path).unwrap();
        assert_eq!(reloaded.data[0], 42.0);
        assert_eq!(reloaded.center_of_rotation, Some(2.25));
        assert_eq!(reloaded.previous_corrections, vec!["{\"tilt_deg\":0.5}"]);

        // A second save appends to the history instead of replacing it.
        save(&reloaded, 2.5, "{\"tilt_deg\":0.1}").unwrap();
        let again = load(&path).unwrap();
        assert_eq!(again.previous_corrections.len(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_wrong_shapes() {
        let path = scratch("flat.h5");
        let file = hdf5_metno::File::create(&path).unwrap();
        file.new_dataset::<f32>()
            .shape((4, 5))
            .create("projections")
            .and_then(|ds| ds.write_raw(&vec![0.0f32; 20]))
            .unwrap();
        drop(file);
        assert!(load(&path).is_err());
        std::fs::remove_file(&path).ok();
    }
}
