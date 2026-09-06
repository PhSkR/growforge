//! What a GPU device says when it refuses, instead of the process dying of it.
//!
//! wgpu reports a refused request - a texture, a buffer, a swapchain the device
//! has no memory left for, and everything that is then made from what it did not
//! return - through an error handler rather than through a `Result`, and its
//! default handler panics. A machine whose video memory another program has
//! taken therefore killed a whole editing session mid-run, with the run's hours
//! in it.
//!
//! [`Scopes`] is the alternative: error scopes over the three kinds of refusal
//! there are, opened around the calls that allocate and read once they are made,
//! so a refusal becomes a value the caller decides about. Scopes rather than
//! [`wgpu::Device::on_uncaptured_error`] for three reasons. They are thread
//! local, which is exactly how the two devices here are used - the viewer's on
//! the window thread, the solver's on its run thread, never shared. They
//! attribute a refusal to the calls made inside them, where a handler installed
//! on the device answers for whatever happened last, wherever it happened. And
//! anything outside them keeps wgpu's own behaviour, so this says nothing about
//! code it was not put around. The pop is immediate - wgpu hands back a future
//! that is already resolved - so blocking on it costs nothing and needs no
//! device poll.

use anyhow::Result;

/// The three ways a device refuses, outermost first.
///
/// Out of memory outermost because it is the one that is a cause: an allocation
/// the device could not make hands back an invalid resource, and everything
/// built from that answers with a validation error about the resource rather
/// than about the memory. [`Scopes::close`] reads them in this order and keeps
/// the outermost, so what is reported is the refusal the others followed from.
const FILTERS: [wgpu::ErrorFilter; 3] = [
    wgpu::ErrorFilter::OutOfMemory,
    wgpu::ErrorFilter::Internal,
    wgpu::ErrorFilter::Validation,
];

/// Error scopes over every kind of refusal, open on a device until they are
/// read.
///
/// Not `Send`: wgpu keeps the scope stack per thread, so these are closed on the
/// thread that opened them.
pub struct Scopes(Vec<wgpu::ErrorScopeGuard>);

impl Scopes {
    /// Open a scope per filter on `device`.
    pub fn open(device: &wgpu::Device) -> Scopes {
        Scopes(
            FILTERS
                .iter()
                .map(|&filter| device.push_error_scope(filter))
                .collect(),
        )
    }

    /// Close them all and answer what the device refused with, if it refused.
    ///
    /// Every scope is popped, in the reverse of the order they were opened -
    /// wgpu requires that order, and one left open would catch the next frame's
    /// errors as well as this one's - so what comes back is the outermost
    /// refusal caught rather than the first.
    pub fn close(mut self) -> Option<Refusal> {
        let mut caught = None;
        while let Some(scope) = self.0.pop() {
            if let Some(error) = pollster::block_on(scope.pop()) {
                caught = Some(error);
            }
        }
        caught.map(Refusal::of)
    }
}

impl Drop for Scopes {
    /// Close what an early return left open, in the order wgpu requires.
    ///
    /// Each guard pops itself when it is dropped, but a `Vec` drops its elements
    /// front to back - the order wgpu panics on - so they are taken off the back
    /// here instead. A `?` between [`Scopes::open`] and [`Scopes::close`] is
    /// therefore just an unread scope rather than a second way to die.
    fn drop(&mut self) {
        while self.0.pop().is_some() {}
    }
}

/// What a device refused with: what it said, and whether what it said was that
/// it had run out of memory.
#[derive(Debug, Clone)]
pub struct Refusal {
    message: String,
    out_of_memory: bool,
}

impl Refusal {
    /// Read one out of wgpu's own error type.
    pub(crate) fn of(error: wgpu::Error) -> Refusal {
        Refusal {
            out_of_memory: matches!(error, wgpu::Error::OutOfMemory { .. }),
            // wgpu describes a validation failure over several indented lines,
            // and what this ends up in is one console line.
            message: error
                .to_string()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    /// True when the device said it had no memory left, which is the one cause
    /// worth naming to a user: something else on the machine is using the card.
    pub fn out_of_memory(&self) -> bool {
        self.out_of_memory
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

/// Ask `device` to do `work`, with a refusal caught rather than fatal.
///
/// `what` completes "the GPU refused to ...", so it reads as the thing that was
/// asked for: `"make the viewer's depth texture"`.
pub fn ask<T>(device: &wgpu::Device, what: &str, work: impl FnOnce() -> T) -> Result<T> {
    let scopes = Scopes::open(device);
    let value = work();
    match scopes.close() {
        None => Ok(value),
        Some(refusal) => Err(refused(what, &refusal)),
    }
}

/// A request a device refused: what was asked of it, what it said, and what the
/// user can do about it.
///
/// An error of its own rather than a message, so that a caller deciding what to
/// do about it - the solver's fallback above all - can ask whether the device
/// was out of memory instead of reading the words back.
#[derive(Debug, Clone)]
pub struct Refused {
    what: String,
    refusal: Refusal,
    advice: Option<String>,
}

impl Refused {
    /// True when the device refused because it had no memory left.
    pub fn out_of_memory(&self) -> bool {
        self.refusal.out_of_memory()
    }
}

impl std::fmt::Display for Refused {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Refused {
            what,
            refusal,
            advice,
        } = self;
        // wgpu's own two words for a device with nothing left are "Out of
        // Memory", which say nothing a user can act on and nothing this does
        // not.
        if refusal.out_of_memory() {
            write!(
                formatter,
                "the GPU refused to {what}: the device is out of memory"
            )?;
        } else {
            write!(formatter, "the GPU refused to {what}: {refusal}")?;
        }
        match advice {
            Some(advice) => write!(formatter, "; {advice}"),
            None => Ok(()),
        }
    }
}

impl std::error::Error for Refused {}

/// The error a refusal becomes: what was asked for, and what the device said
/// about it.
pub fn refused(what: &str, refusal: &Refusal) -> anyhow::Error {
    anyhow::Error::new(Refused {
        what: what.to_string(),
        refusal: refusal.clone(),
        advice: None,
    })
}

/// The same, ending in what the user can do about it - which is not the same
/// advice for a window as for a solve.
pub fn refused_with(what: &str, refusal: &Refusal, advice: &str) -> anyhow::Error {
    anyhow::Error::new(Refused {
        what: what.to_string(),
        refusal: refusal.clone(),
        advice: Some(advice.to_string()),
    })
}

/// Open the machine's own device for a test, or say why the test is being
/// skipped.
///
/// The pattern [`crate::fea::backend::gpu_or_skip`] sets, for the tests that
/// need a real device rather than a real solver: a runner without an adapter
/// must still pass the suite, and a silently skipped test is worse than no test.
#[cfg(test)]
pub(crate) fn device_or_skip(what: &str) -> Option<wgpu::Device> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })) {
        Ok(adapter) => adapter,
        Err(error) => {
            println!("skipping {what}: no GPU adapter on this machine ({error})");
            return None;
        }
    };
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(crate::constants::PROGRAM_NAME),
        required_limits: adapter.limits(),
        ..Default::default()
    })) {
        Ok((device, _queue)) => {
            println!("{what}: {}", adapter.get_info().name);
            Some(device)
        }
        Err(error) => {
            println!("skipping {what}: the adapter refused a device ({error})");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a refusal carries: one line of it, and whether the device said the
    /// memory had run out - which is the difference between "something else on
    /// this machine is using the card" and a bug.
    #[test]
    fn a_refusal_is_one_line_and_says_when_the_memory_is_gone() {
        let sprawling = wgpu::Error::Validation {
            source: Box::new(std::io::Error::other("in Texture::create_view")),
            description: "Validation Error\n\nCaused by:\n  In Texture::create_view\n    \
                          Texture with 'growforge_depth' label is invalid\n"
                .to_string(),
        };
        let refusal = Refusal::of(sprawling);
        assert!(!refusal.out_of_memory());
        assert_eq!(
            refusal.to_string(),
            "Validation Error Caused by: In Texture::create_view Texture with 'growforge_depth' \
             label is invalid"
        );
        assert_eq!(
            refused("make the depth texture", &refusal).to_string(),
            "the GPU refused to make the depth texture: Validation Error Caused by: In \
             Texture::create_view Texture with 'growforge_depth' label is invalid"
        );

        let exhausted = Refusal::of(wgpu::Error::OutOfMemory {
            source: Box::new(std::io::Error::other("no memory")),
        });
        assert!(exhausted.out_of_memory());
        let error = refused("make the depth texture", &exhausted);
        assert_eq!(
            error.to_string(),
            "the GPU refused to make the depth texture: the device is out of memory",
            "the cause worth naming is named rather than left as wgpu's two words"
        );
        assert!(
            error
                .downcast_ref::<Refused>()
                .is_some_and(Refused::out_of_memory),
            "a caller deciding what to do about it has to be able to ask, not to read"
        );
    }

    /// The mechanism itself, on this machine's own device: a request wgpu would
    /// otherwise take the process down over comes back as an error, and the
    /// scopes are left clean enough for the next request to succeed.
    ///
    /// A buffer past the device's own limit is the cheapest refusal there is -
    /// nothing is allocated to find out - and it is the same handler a device
    /// with no memory left answers through, which is the one this cannot stage.
    #[test]
    fn a_refused_request_comes_back_as_an_error_rather_than_a_panic() {
        let Some(device) = device_or_skip("the refused request test") else {
            return;
        };
        let past_the_limit = device.limits().max_buffer_size.saturating_add(1);
        let refused = ask(&device, "make an impossible buffer", || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("growforge_impossible"),
                size: past_the_limit,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let error = refused.expect_err("the device allowed a buffer past its own limit");
        assert!(
            error.downcast_ref::<Refused>().is_some(),
            "a refusal has to arrive as one: {error:#}"
        );
        println!("the device refused as expected: {error:#}");

        ask(&device, "make a buffer it can make", || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("growforge_possible"),
                size: size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        })
        .expect("a refusal left a scope behind that caught the next request");
    }
}
