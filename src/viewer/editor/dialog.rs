//! The native file dialogs behind the editor's "open" and "new" buttons.
//!
//! Two thin shells over the platform's own picker - the file dialog every other
//! program on the machine raises - filtered to configurations and pointed at the
//! directory the session's file came from. "Open" picks an existing file; "new"
//! is save-as shaped, so a name can be typed into a folder that has nothing in
//! it yet.
//!
//! **The call blocks.** An OS file dialog owns the input queue while it is up
//! and the window behind it stops drawing until it is answered, which is what a
//! modal dialog is. Nothing else is affected: the run behind the window is on
//! its own thread and keeps going, and the channels that feed the window are
//! single-slot ones that drop what nobody collects - see
//! [`crate::viewer::snapshot::LatestSlot`] - so a dialog left open costs dropped
//! preview frames and nothing else. The picker is raised unparented: the crate
//! takes the same `raw-window-handle` the winit and wgpu stack pins, so
//! parenting is available, but nothing here depends on that alignment holding.
//!
//! **Nothing here is unit tested, by construction.** A native dialog cannot be
//! answered by a test, so this module is kept down to the call itself: it takes
//! a directory and hands back a path or nothing. Everything that then happens
//! to that path - the unsaved-changes guard, the swap of the session, the
//! scaffold's refusal to overwrite - is reached by handing a path to
//! [`Editor::request_open`](super::Editor::request_open) and
//! [`Editor::request_new`](super::Editor::request_new) directly, and that is the
//! surface the tests drive. The one thing here with a decision in it - the
//! extension a typed name is given - is a pure function, and is tested.

use std::path::{Path, PathBuf};

use crate::constants;

/// Ask for an existing configuration to edit, starting in `directory`.
///
/// `None` when the dialog was cancelled, which is an answer: nothing was asked
/// for after all.
pub fn choose_open(directory: &Path) -> Option<PathBuf> {
    rfd::FileDialog::new()
        .set_title(constants::VIEW_EDIT_OPEN_DIALOG_TITLE)
        .add_filter(
            constants::VIEW_EDIT_FILE_FILTER,
            &[constants::VIEW_EDIT_FILE_EXTENSION],
        )
        .set_directory(directory)
        .pick_file()
}

/// Ask for a name to scaffold a new configuration at, starting in `directory`.
///
/// Save-as shaped, because the file does not exist yet and a name has to be
/// typed. What comes back is only ever a path: whether it may be created is the
/// editor's question, and the scaffold's - the dialog's own replace prompt is
/// not what decides it.
pub fn choose_new(directory: &Path) -> Option<PathBuf> {
    let chosen = rfd::FileDialog::new()
        .set_title(constants::VIEW_EDIT_NEW_DIALOG_TITLE)
        .add_filter(
            constants::VIEW_EDIT_FILE_FILTER,
            &[constants::VIEW_EDIT_FILE_EXTENSION],
        )
        .set_directory(directory)
        .set_file_name(constants::VIEW_EDIT_NEW_FILE_NAME)
        .save_file()?;
    Some(with_extension(chosen, constants::VIEW_EDIT_FILE_EXTENSION))
}

/// `path` with `extension` when it has none of its own.
///
/// Not every platform's save dialog appends the filter's extension to a name
/// typed without one, and a configuration the file manager does not recognize is
/// a worse thing to leave behind than one whose name gained four characters. A
/// name that already carries an extension - any extension - is the user's own
/// and is left exactly as it was typed.
fn with_extension(path: PathBuf, extension: &str) -> PathBuf {
    if path.extension().is_some() {
        path
    } else {
        path.with_extension(extension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_typed_without_an_extension_is_given_the_configuration_one() {
        let extension = constants::VIEW_EDIT_FILE_EXTENSION;
        assert_eq!(
            with_extension(PathBuf::from("parts/bracket"), extension),
            PathBuf::from("parts/bracket.toml")
        );
        // One that has an extension keeps it, whatever it is.
        assert_eq!(
            with_extension(PathBuf::from("parts/bracket.toml"), extension),
            PathBuf::from("parts/bracket.toml")
        );
        assert_eq!(
            with_extension(PathBuf::from("parts/bracket.cfg"), extension),
            PathBuf::from("parts/bracket.cfg")
        );
        // And a dotted stem is a stem, not an extensionless name.
        assert_eq!(
            with_extension(PathBuf::from("parts/bracket.v2"), extension),
            PathBuf::from("parts/bracket.v2")
        );
    }
}
