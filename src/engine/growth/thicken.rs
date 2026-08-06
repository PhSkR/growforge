//! Sizing the struts: how much load a branch carries, and how thick that makes
//! it.
//!
//! Flow is the load a segment has to carry down to its root. It enters the
//! skeleton at the tips - at a backbone's load attachment node it is the load
//! region's own magnitude, at every other tip it is the canopy's share - and
//! accumulates towards the roots, so a segment's flow is the sum of everything
//! hanging off it.
//!
//! Radii follow Murray's law, `r_parent^n = sum(r_child^n)`. Setting
//! `r = C * flow^(1/n)` satisfies it identically at every junction that carries
//! no load of its own, because flow is additive there; a junction that is
//! itself a load attachment point satisfies the physical form of the law,
//! `r_parent^n = sum(r_child^n) + C^n * load`. The single scale `C` is then
//! bisected until the rasterized field hits the requested `mass_fraction`.

use crate::constants;
use crate::engine::growth::skeleton::Skeleton;

/// Load entering the skeleton at every node, in newtons.
///
/// `region_loads` gives the magnitude attached to individual nodes (the
/// backbones' load ends); every leaf that carries none of those gets an equal
/// share of `canopy_load`, so the grown branches taper like the backbone
/// instead of collapsing to the minimum feature everywhere.
pub fn node_loads(
    skeleton: &Skeleton,
    region_loads: &[(usize, f64)],
    canopy_load: f64,
) -> Vec<f64> {
    let mut loads = vec![0.0; skeleton.len()];
    for (node, magnitude) in region_loads {
        loads[*node] += magnitude;
    }
    let leaves = skeleton.leaves();
    let tips: Vec<usize> = (0..skeleton.len())
        .filter(|&n| leaves[n] && loads[n] == 0.0)
        .collect();
    if !tips.is_empty() {
        let share = canopy_load / tips.len() as f64;
        for tip in tips {
            loads[tip] = share;
        }
    }
    loads
}

/// Accumulate the node loads down the forest towards the roots.
///
/// One reverse pass suffices: a node's parent always has a smaller index, so
/// walking from the last node to the first visits every child before its
/// parent.
pub fn accumulate_flow(skeleton: &Skeleton, node_loads: &[f64]) -> Vec<f64> {
    let mut flow = node_loads.to_vec();
    for index in (0..skeleton.len()).rev() {
        if let Some(parent) = skeleton.nodes[index].parent {
            flow[parent] += flow[index];
        }
    }
    flow
}

/// Radius of every node's segment: `clamp(C * flow^(1/n))`.
pub fn radii(
    flow: &[f64],
    scale: f64,
    exponent: f64,
    min_radius: f64,
    max_radius: f64,
) -> Vec<f64> {
    let inverse = 1.0 / exponent;
    flow.iter()
        .map(|f| (scale * f.max(0.0).powf(inverse)).clamp(min_radius, max_radius))
        .collect()
}

/// What the volume normalization settled on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scaling {
    /// The radius scale `C`.
    pub scale: f64,
    /// Volume fraction the resulting radii rasterize to.
    pub volume_fraction: f64,
    /// Set when the radius clamps stopped the search from reaching the target;
    /// the value is the fraction that is actually achievable.
    pub clamped: Option<f64>,
}

/// Bisect the radius scale until the rasterized volume fraction hits `target`.
///
/// `measure` rasterizes a radius array and returns its volume fraction, which
/// is monotone non-decreasing in the scale: a thicker strut can only add
/// material. The bracket is exact rather than searched, because both ends are
/// known in closed form - at a scale of zero every radius is clamped to the
/// minimum, and at `max_radius / min_flow^(1/n)` every radius is clamped to the
/// maximum.
///
/// A target outside that bracket is unreachable: growforge reports the fraction
/// the clamps do allow and carries on with the nearest end, because a design
/// that is a little too heavy or too light is still a design, while failing the
/// run leaves the user with nothing.
pub fn resolve_scale(
    flow: &[f64],
    exponent: f64,
    min_radius: f64,
    max_radius: f64,
    target: f64,
    mut measure: impl FnMut(&[f64]) -> f64,
) -> Scaling {
    let smallest_flow = flow
        .iter()
        .copied()
        .filter(|f| *f > 0.0)
        .fold(f64::INFINITY, f64::min);
    let low = 0.0;
    // A scale this large clamps even the thinnest branch to the maximum
    // radius, so it is the heaviest structure the clamps allow. A degenerate
    // flow (nothing carries anything) collapses the bracket to a point, which
    // the checks below then report as clamped.
    let mut high = if smallest_flow.is_finite() && smallest_flow > 0.0 {
        max_radius / smallest_flow.powf(1.0 / exponent)
    } else {
        max_radius
    };
    if high.is_nan() || high <= low {
        high = low;
    }

    let at = |scale: f64, measure: &mut dyn FnMut(&[f64]) -> f64| {
        measure(&radii(flow, scale, exponent, min_radius, max_radius))
    };
    let tolerance = constants::GROWTH_VOLUME_TOLERANCE * target;

    let thinnest = at(low, &mut measure);
    if thinnest > target + tolerance {
        return Scaling {
            scale: low,
            volume_fraction: thinnest,
            clamped: Some(thinnest),
        };
    }
    let thickest = at(high, &mut measure);
    if thickest < target - tolerance {
        return Scaling {
            scale: high,
            volume_fraction: thickest,
            clamped: Some(thickest),
        };
    }

    let mut best = thickest;
    let mut scale = high;
    let mut low = low;
    for _ in 0..constants::GROWTH_VOLUME_BISECTION_MAX_STEPS {
        scale = 0.5 * (low + high);
        best = at(scale, &mut measure);
        if (best - target).abs() <= tolerance {
            break;
        }
        if best < target {
            low = scale;
        } else {
            high = scale;
        }
    }
    Scaling {
        scale,
        volume_fraction: best,
        clamped: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::growth::skeleton::Skeleton;

    /// A root, two children of it, and two children of the first child.
    fn fork() -> Skeleton {
        let mut skeleton = Skeleton::new();
        let root = skeleton.add_root([0.0, 0.0, 0.0]);
        let a = skeleton.add_child(root, [1.0, 0.0, 0.0]);
        let _b = skeleton.add_child(root, [0.0, 1.0, 0.0]);
        skeleton.add_child(a, [2.0, 0.0, 0.0]);
        skeleton.add_child(a, [1.0, 1.0, 0.0]);
        skeleton
    }

    #[test]
    fn flow_accumulates_from_every_tip_to_the_root() {
        let skeleton = fork();
        let loads = node_loads(&skeleton, &[], 4.0);
        // Three leaves share the canopy load; the interior nodes carry none.
        assert_eq!(loads, vec![0.0, 0.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0]);
        let flow = accumulate_flow(&skeleton, &loads);
        assert!((flow[0] - 4.0).abs() < 1e-12, "the root carries everything");
        assert!((flow[1] - 8.0 / 3.0).abs() < 1e-12);
        assert!((flow[2] - 4.0 / 3.0).abs() < 1e-12);

        // A named load region lands on its node and is not diluted by the tips.
        let loads = node_loads(&skeleton, &[(3, 90.0)], 3.0);
        assert_eq!(loads[3], 90.0);
        assert_eq!(loads[2], 1.5);
        assert_eq!(loads[4], 1.5);
        let flow = accumulate_flow(&skeleton, &loads);
        assert!((flow[0] - 93.0).abs() < 1e-12);
    }

    #[test]
    fn murray_law_holds_at_every_unloaded_junction() {
        let skeleton = fork();
        let exponent = constants::DEFAULT_MURRAY_EXPONENT;
        let flow = accumulate_flow(&skeleton, &node_loads(&skeleton, &[], 6.0));
        // Unclamped: a wide enough band that nothing is limited by it.
        let r = radii(&flow, 0.7, exponent, 0.0, f64::INFINITY);

        // Node 1's segment carries its two children's segments.
        let parent = r[1].powf(exponent);
        let children = r[3].powf(exponent) + r[4].powf(exponent);
        assert!(
            (parent - children).abs() < 1e-9 * parent,
            "r_parent^n = {parent} but the children sum to {children}"
        );
        // And so does the root's, over both of its children.
        let root = r[0].powf(exponent);
        let below = r[1].powf(exponent) + r[2].powf(exponent);
        assert!((root - below).abs() < 1e-9 * root, "{root} against {below}");
        // A thicker parent than child is the whole point.
        assert!(r[0] > r[1] && r[1] > r[3]);
    }

    #[test]
    fn the_scale_is_bisected_until_the_volume_target_is_met() {
        let skeleton = fork();
        let flow = accumulate_flow(&skeleton, &node_loads(&skeleton, &[], 6.0));
        // A stand-in for the rasterizer: volume grows with the cube of the
        // radius, which is monotone in the scale just as the real one is.
        let measure = |r: &[f64]| r.iter().map(|r| r * r * r).sum::<f64>() / 100.0;
        let target = 0.35;
        let outcome = resolve_scale(&flow, 2.6, 0.0, 1e6, target, measure);
        assert_eq!(outcome.clamped, None);
        assert!(
            (outcome.volume_fraction - target).abs() <= constants::GROWTH_VOLUME_TOLERANCE * target,
            "settled on {} instead of {target}",
            outcome.volume_fraction
        );
        assert!(outcome.scale > 0.0);
    }

    #[test]
    fn clamped_radii_report_the_fraction_they_can_reach() {
        let skeleton = fork();
        let flow = accumulate_flow(&skeleton, &node_loads(&skeleton, &[], 6.0));
        let measure = |r: &[f64]| r.iter().map(|r| r * r * r).sum::<f64>() / 100.0;

        // Too heavy to be asked for: even everything at the maximum radius
        // cannot fill it.
        let outcome = resolve_scale(&flow, 2.6, 0.1, 0.5, 10.0, measure);
        let reachable = outcome.clamped.expect("the clamp has to be reported");
        assert!(reachable < 10.0 && reachable > 0.0);
        assert!((outcome.volume_fraction - reachable).abs() < 1e-12);

        // And too light: the minimum radius alone already overshoots.
        let outcome = resolve_scale(&flow, 2.6, 2.0, 4.0, 0.01, measure);
        let reachable = outcome.clamped.expect("the clamp has to be reported");
        assert!(reachable > 0.01);
        assert_eq!(outcome.scale, 0.0);
    }

    #[test]
    fn radii_stay_inside_their_clamps() {
        let flow = vec![0.0, 1e-9, 1.0, 1e9];
        let r = radii(&flow, 3.0, 2.6, 0.5, 2.0);
        assert!(r.iter().all(|r| (0.5..=2.0).contains(r)), "{r:?}");
        assert_eq!(r[0], 0.5);
        assert_eq!(r[3], 2.0);
    }
}
