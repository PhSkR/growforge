//! Binary STL export and a reader used to verify what was written.
//!
//! Layout: an 80 byte header, a little endian `u32` triangle count, then one
//! 50 byte record per triangle holding the facet normal, the three vertices as
//! little endian `f32` triples, and a 16 bit attribute field.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::constants;
use crate::geometry::{cross, difference};
use crate::mesh::Mesh;

/// The 80 byte header growforge 3D stamps onto its STL files.
///
/// The display name, not the machine one: a slicer shows this header beside the
/// part, so it is a surface a user reads.
pub fn header() -> [u8; constants::STL_HEADER_BYTES] {
    let mut header = [0u8; constants::STL_HEADER_BYTES];
    let bytes = constants::DISPLAY_NAME_AND_VERSION.as_bytes();
    let n = bytes.len().min(constants::STL_HEADER_BYTES);
    header[..n].copy_from_slice(&bytes[..n]);
    header
}

/// Unit normal of a triangle, or the zero vector when it is degenerate.
fn facet_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f32; 3] {
    let n = cross(difference(b, a), difference(c, a));
    let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if length == 0.0 {
        return [0.0; 3];
    }
    [
        (n[0] / length) as f32,
        (n[1] / length) as f32,
        (n[2] / length) as f32,
    ]
}

/// Write `mesh` to `path` as a binary STL.
pub fn write(path: &Path, mesh: &Mesh) -> Result<()> {
    if mesh.triangles.len() > constants::MAX_STL_TRIANGLES {
        bail!(
            "the mesh has {} triangles, more than a binary STL can index ({})",
            mesh.triangles.len(),
            constants::MAX_STL_TRIANGLES
        );
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    let file =
        File::create(path).with_context(|| format!("creating STL file {}", path.display()))?;
    let mut out = BufWriter::new(file);
    out.write_all(&header())?;
    out.write_all(&(mesh.triangles.len() as u32).to_le_bytes())?;

    let mut record = [0u8; constants::STL_TRIANGLE_BYTES];
    for t in &mesh.triangles {
        let a = mesh.vertices[t[0] as usize];
        let b = mesh.vertices[t[1] as usize];
        let c = mesh.vertices[t[2] as usize];
        let normal = facet_normal(a, b, c);
        let mut offset = 0;
        for value in normal {
            record[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            offset += 4;
        }
        for vertex in [a, b, c] {
            for coordinate in vertex {
                record[offset..offset + 4].copy_from_slice(&(coordinate as f32).to_le_bytes());
                offset += 4;
            }
        }
        record[offset..offset + 2]
            .copy_from_slice(&constants::STL_ATTRIBUTE_BYTE_COUNT.to_le_bytes());
        out.write_all(&record)?;
    }
    out.flush()
        .with_context(|| format!("writing STL file {}", path.display()))?;
    Ok(())
}

/// A triangle as stored in a binary STL file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StlTriangle {
    /// Facet normal.
    pub normal: [f32; 3],
    /// The three corner positions.
    pub vertices: [[f32; 3]; 3],
}

/// Parse a binary STL file.
pub fn read(path: &Path) -> Result<(Vec<u8>, Vec<StlTriangle>)> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("opening STL file {}", path.display()))?
        .read_to_end(&mut bytes)?;
    let prefix = constants::STL_HEADER_BYTES + 4;
    if bytes.len() < prefix {
        bail!("{} is too short to be a binary STL", path.display());
    }
    let header = bytes[..constants::STL_HEADER_BYTES].to_vec();
    let count = u32::from_le_bytes(
        bytes[constants::STL_HEADER_BYTES..prefix]
            .try_into()
            .expect("four bytes"),
    ) as usize;
    let expected = prefix + count * constants::STL_TRIANGLE_BYTES;
    if bytes.len() != expected {
        bail!(
            "{} declares {count} triangles, which needs {expected} bytes, but the file holds {}",
            path.display(),
            bytes.len()
        );
    }

    let read_f32 = |at: usize| -> f32 {
        f32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
    };
    let mut triangles = Vec::with_capacity(count);
    for t in 0..count {
        let base = prefix + t * constants::STL_TRIANGLE_BYTES;
        let normal = [read_f32(base), read_f32(base + 4), read_f32(base + 8)];
        let mut vertices = [[0.0f32; 3]; 3];
        for (v, vertex) in vertices.iter_mut().enumerate() {
            for (d, slot) in vertex.iter_mut().enumerate() {
                *slot = read_f32(base + 12 + v * 12 + d * 4);
            }
        }
        triangles.push(StlTriangle { normal, vertices });
    }
    Ok((header, triangles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::validate;

    fn tetrahedron() -> Mesh {
        Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            triangles: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
        }
    }

    #[test]
    fn round_trip_preserves_count_and_geometry() {
        let mesh = tetrahedron();
        validate::validate(&mesh).expect("the fixture must be a closed solid");
        let path = std::env::temp_dir().join("growforge_stl_round_trip.stl");
        write(&path, &mesh).expect("write");

        let (header, triangles) = read(&path).expect("read");
        // Spelled out rather than read back from the constant that wrote it: a
        // slicer shows this line, so what is asserted is the string on a user's
        // screen and not a second reading of the same value.
        let stamped = String::from_utf8_lossy(&header);
        let expected = format!("growforge 3D {}", env!("CARGO_PKG_VERSION"));
        assert!(
            stamped.starts_with(&expected),
            "the header does not open with {expected:?}: {stamped:?}"
        );
        assert_eq!(triangles.len(), mesh.triangles.len());

        for (t, stored) in mesh.triangles.iter().zip(triangles.iter()) {
            for (m, &index) in t.iter().enumerate() {
                let expected = mesh.vertices[index as usize];
                for (d, &coordinate) in expected.iter().enumerate() {
                    assert!(
                        (stored.vertices[m][d] as f64 - coordinate).abs() < 1e-6,
                        "vertex mismatch"
                    );
                }
            }
        }

        // The first facet lies in the z = 0 plane and faces away from the solid.
        assert!((triangles[0].normal[2] + 1.0).abs() < 1e-6);

        let size = std::fs::metadata(&path).expect("metadata").len() as usize;
        assert_eq!(
            size,
            constants::STL_HEADER_BYTES + 4 + mesh.triangles.len() * constants::STL_TRIANGLE_BYTES
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn truncated_files_are_rejected() {
        let mesh = tetrahedron();
        let path = std::env::temp_dir().join("growforge_stl_truncated.stl");
        write(&path, &mesh).expect("write");
        let mut bytes = std::fs::read(&path).expect("read");
        bytes.truncate(bytes.len() - 10);
        std::fs::write(&path, &bytes).expect("write back");
        assert!(read(&path).is_err());
        std::fs::remove_file(&path).ok();
    }
}
