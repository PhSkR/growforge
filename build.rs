//! Stamps the Windows VersionInfo resource onto the executable.
//!
//! Without it the file's Properties tab reports version 0.0.0.0 and no product
//! name at all, which is what an installer, an inventory tool or anyone looking
//! at a copied `growforge.exe` reads first. Everything written here comes from
//! the `CARGO_PKG_*` variables Cargo sets for build scripts, so the manifest
//! stays the single place the version and the description are written: nothing
//! here can name a version the package is not.
//!
//! A resource is a Windows PE concept. On any other target this script does
//! nothing at all, and the build is exactly what it was before.

use std::env;

fn main() {
    // The resource depends on this script and on the manifest values Cargo
    // hands it, so the only thing worth re-running on is the script itself:
    // Cargo already rebuilds the crate - and with it the build script - when
    // the manifest changes.
    println!("cargo::rerun-if-changed=build.rs");

    // The *target* OS, not the host: a Linux build hosted on Windows must not
    // get a PE resource compiled into it.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let version =
        env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION is set for build scripts");
    let mut resource = winresource::WindowsResource::new();
    // The strings are set explicitly rather than left to the crate's defaults
    // so that what the Properties tab shows is spelled out here: the version
    // exactly as the manifest writes it, not the four-part form the binary
    // FILEVERSION field is padded to.
    resource
        .set("FileVersion", &version)
        .set("ProductVersion", &version)
        .set(
            "ProductName",
            &env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME"),
        )
        .set(
            "FileDescription",
            &env::var("CARGO_PKG_DESCRIPTION").expect("CARGO_PKG_DESCRIPTION"),
        );
    // A failure here is a build failure, deliberately: a release binary that
    // silently lost its version is the thing this script exists to prevent, and
    // the message names the resource compiler that could not be run.
    resource
        .compile()
        .expect("compiling the Windows version resource (needs rc.exe from the Windows SDK)");
}
