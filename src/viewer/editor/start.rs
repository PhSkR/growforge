//! `growforge edit` with no path: what a session opens on when nobody named a
//! file.
//!
//! A path on the command line is a terminal's way of starting the editor, and
//! it stays exactly what it was. An installed copy is started from a Start Menu
//! shortcut instead, where there is no path to type and no useful working
//! directory either - the program sits where installed programs sit, and
//! nothing may be written there - so the one thing it can do is ask, in the
//! platform's own file dialog, about the one place the answer sensibly lives:
//! [`configs_home`], the growforge folder under the user's Documents.
//!
//! Which dialog it raises is the whole of the decision here, and it is made by
//! looking at that folder:
//!
//! ```text
//!   holds at least one .toml   ->  "open"    pick one of the files you have
//!   empty, or not there yet    ->  save-as   name the first one
//! ```
//!
//! Both dialogs are the editor's own - the same titles, the same filter, the
//! same extension given to a name typed without one - only seeded at the
//! configurations home rather than at the session's directory, because there is
//! no session yet. What comes back is a path, and from there this is `growforge
//! edit <path>`: an existing file opens, a name that is not there yet is
//! scaffolded. **Nothing is created on the way to the question.** The home
//! folder comes into existence when a starter configuration is written into it
//! and not before, so a cancelled dialog leaves the filesystem exactly as it
//! was.
//!
//! The dialog is raised before the window exists, which the picker does not
//! mind: it is raised unparented in the editor too - see
//! [`super::dialog`].

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::constants;
use crate::viewer::editor::dialog;

/// Which dialog a pathless `growforge edit` opens with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    /// The home already holds configurations: pick one of them.
    Open,
    /// There is nothing to pick: name the first one.
    Create,
}

/// The canonical home of a user's growforge configurations: the platform's
/// Documents folder with [`constants::VIEW_EDIT_CONFIGS_HOME_DIR`] under it.
///
/// `None` when the platform has no Documents folder to name - a Linux account
/// without the XDG user directories set up, say. It is asked for by known
/// folder rather than assembled out of the home directory, because a Documents
/// folder redirected onto a cloud drive is not under the profile at all and a
/// guess would point at a folder the user's files are not in.
pub fn configs_home() -> Option<PathBuf> {
    Some(dirs::document_dir()?.join(constants::VIEW_EDIT_CONFIGS_HOME_DIR))
}

/// Whether `home` is a folder with configurations already in it, and so which
/// dialog to raise.
///
/// A path that is not there, one that is not a directory, one that cannot be
/// read and one holding nothing that ends in the configuration extension all
/// answer the same way: there is nothing to pick from, so the first file has to
/// be named. Only a *file* counts - a folder called `parts.toml` is not a
/// configuration - and the extension is matched without regard to case, because
/// the filesystem this runs on does not distinguish `.TOML` from `.toml`.
pub fn start_for(home: &Path) -> Start {
    let Ok(entries) = std::fs::read_dir(home) else {
        return Start::Create;
    };
    let holds_one = entries.flatten().any(|entry| {
        entry.file_type().is_ok_and(|kind| kind.is_file())
            && entry.path().extension().is_some_and(|extension| {
                extension.eq_ignore_ascii_case(constants::VIEW_EDIT_FILE_EXTENSION)
            })
    });
    if holds_one {
        Start::Open
    } else {
        Start::Create
    }
}

/// The folder a dialog aimed at `home` can actually be opened in.
///
/// `home` itself, once it is there. Until then - the first run on a machine -
/// the nearest folder above it that does exist, which is the Documents folder:
/// a dialog cannot show a folder that is not there, and one seeded at a missing
/// path is left wherever the shell was last, which is nowhere in particular.
/// This is the only thing the missing home changes; it is still not created
/// until a file is saved into it.
fn seed_directory(home: &Path) -> &Path {
    home.ancestors()
        .find(|ancestor| ancestor.is_dir())
        .unwrap_or(home)
}

/// Open the editor on a file the user picks, with nothing named on the command
/// line.
///
/// Cancelling is an answer: the command says so in one line and exits
/// successfully, having written nothing.
pub fn edit_without_a_path() -> Result<()> {
    // With no Documents folder to name, the folder the command was typed in is
    // the only place left that means anything to the user, and it is what the
    // dialog opens in.
    let home = match configs_home() {
        Some(home) => home,
        None => std::env::current_dir()?,
    };
    let seed = seed_directory(&home);
    let chosen = match start_for(&home) {
        Start::Open => dialog::choose_open(seed),
        Start::Create => dialog::choose_new(seed),
    };
    match chosen {
        // From here this is `growforge edit <path>`, down to the same call: an
        // existing file opens, a name that is not there yet is scaffolded, and
        // the folder it is scaffolded into is created by that write.
        Some(path) => super::edit(&path),
        None => {
            println!("{}", constants::VIEW_EDIT_NOTHING_OPENED);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::editor::tests::TempDir;

    /// A directory of this test's own that removes itself, created or not as
    /// the test asks.
    fn scratch(name: &str, create: bool) -> (TempDir, PathBuf) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "growforge_start_{name}_{}_{unique}",
            std::process::id()
        ));
        std::fs::remove_dir_all(&directory).ok();
        if create {
            std::fs::create_dir_all(&directory).expect("temp directory");
        }
        (TempDir::at(directory.clone()), directory)
    }

    #[test]
    fn a_home_that_is_not_there_yet_asks_for_the_first_name() {
        let (_guard, home) = scratch("missing", false);
        assert_eq!(start_for(&home), Start::Create);
        // And asking did not bring it into existence.
        assert!(
            !home.exists(),
            "{} was created by the question",
            home.display()
        );
    }

    #[test]
    fn an_empty_home_asks_for_the_first_name() {
        let (_guard, home) = scratch("empty", true);
        assert_eq!(start_for(&home), Start::Create);
    }

    #[test]
    fn a_home_of_other_files_still_asks_for_the_first_name() {
        let (_guard, home) = scratch("other_files", true);
        std::fs::write(home.join("notes.txt"), "not a configuration").expect("write");
        std::fs::write(home.join("bracket.stl"), "not a configuration").expect("write");
        // A folder whose name ends in the extension is not a configuration.
        std::fs::create_dir(home.join("archive.toml")).expect("directory");
        assert_eq!(start_for(&home), Start::Create);
    }

    #[test]
    fn a_home_holding_a_configuration_offers_the_ones_that_are_there() {
        let (_guard, home) = scratch("holds_one", true);
        std::fs::write(home.join("notes.txt"), "not a configuration").expect("write");
        std::fs::write(home.join("bracket.toml"), "[project]").expect("write");
        assert_eq!(start_for(&home), Start::Open);
    }

    #[test]
    fn the_extension_is_matched_whatever_case_it_was_written_in() {
        let (_guard, home) = scratch("upper_case", true);
        std::fs::write(home.join("BRACKET.TOML"), "[project]").expect("write");
        assert_eq!(start_for(&home), Start::Open);
    }

    #[test]
    fn a_home_that_is_there_is_what_the_dialog_opens_in() {
        let (_guard, home) = scratch("seed_present", true);
        assert_eq!(seed_directory(&home), home);
    }

    #[test]
    fn a_home_that_is_not_there_yet_opens_the_dialog_in_the_folder_above_it() {
        let (_guard, home) = scratch("seed_missing", false);
        assert_eq!(seed_directory(&home), std::env::temp_dir());
    }

    #[test]
    fn the_configurations_home_is_the_growforge_folder_under_documents() {
        let Some(home) = configs_home() else {
            // A platform with no Documents folder to name is exactly the case
            // `edit_without_a_path` falls back for; there is nothing to assert.
            return;
        };
        assert_eq!(
            home.file_name().and_then(|name| name.to_str()),
            Some(constants::VIEW_EDIT_CONFIGS_HOME_DIR)
        );
        assert!(
            home.is_absolute(),
            "the configurations home is not absolute: {}",
            home.display()
        );
        assert_eq!(
            home.parent(),
            dirs::document_dir().as_deref(),
            "the configurations home is not under the Documents folder"
        );
    }
}
