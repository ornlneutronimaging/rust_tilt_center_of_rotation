//! Minimal NumPy .npy v1.0 I/O for f32 arrays — just what the gridrec test
//! slice hand-over needs (write a 2-D sinogram, read back the stack of
//! reconstructed slices).

use std::io::{Read, Write};
use std::path::Path;

/// Write a C-order f32 array of the given shape.
pub fn write_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<(), String> {
    if shape.iter().product::<usize>() != data.len() {
        return Err(format!(
            "npy shape {shape:?} does not match {} values",
            data.len()
        ));
    }
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_str = match shape.len() {
        1 => format!("({},)", dims[0]),
        _ => format!("({})", dims.join(", ")),
    };
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}");
    let unpadded = 8 + 2 + header.len() + 1;
    header.push_str(&" ".repeat(unpadded.div_ceil(64) * 64 - unpadded));
    header.push('\n');
    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    file.write_all(&out)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Read a C-order little-endian f32 array of any rank → (shape, data).
pub fn read_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a .npy file", path.display()));
    }
    let (header_len, data_start) = match bytes[6] {
        1 => (
            u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
            10usize,
        ),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        ),
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let header = String::from_utf8_lossy(&bytes[data_start..data_start + header_len]);
    if !header.contains("'descr': '<f4'") {
        return Err(format!("expected little-endian f32 data, header: {header}"));
    }
    if header.contains("'fortran_order': True") {
        return Err("Fortran-ordered .npy is not supported".to_owned());
    }
    let shape_part = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| format!("cannot parse the shape from: {header}"))?;
    let shape: Vec<usize> = shape_part
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let n: usize = shape.iter().product();
    let data_bytes = &bytes[data_start + header_len..];
    if data_bytes.len() < n * 4 {
        return Err(format!(
            "{}: {} data bytes for shape {shape:?}",
            path.display(),
            data_bytes.len()
        ));
    }
    let data: Vec<f32> = data_bytes[..n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((shape, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_2d_and_3d() {
        let dir = std::env::temp_dir().join(format!("npy_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.npy");
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
        write_f32(&path, &[4, 6], &data).unwrap();
        assert_eq!(read_f32(&path).unwrap(), (vec![4, 6], data.clone()));
        write_f32(&path, &[2, 3, 4], &data).unwrap();
        assert_eq!(read_f32(&path).unwrap(), (vec![2, 3, 4], data));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn header_is_64_byte_aligned() {
        let dir = std::env::temp_dir().join(format!("npy_align_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.npy");
        write_f32(&path, &[1], &[1.0]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
