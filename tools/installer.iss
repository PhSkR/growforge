; growforge 3D Windows installer (Inno Setup 6).
;
; Compiled by tools\build_installer.ps1, which passes the version in:
;
;   ISCC.exe /DAppVersion=0.40.0 tools\installer.iss
;
; The version is never written here. It comes from the built executable's
; VersionInfo resource, which comes from Cargo.toml, so the manifest stays the
; one place a version is stated and a setup can never name a build it does not
; carry.
;
; Relative source paths below resolve against this file's own directory
; (tools\), which is what SourceDir defaults to, so the repository can sit
; anywhere.

#ifndef AppVersion
  #error AppVersion is required: compile through tools\build_installer.ps1, or pass /DAppVersion=x.y.z
#endif

[Setup]
; The upgrade identity. NEVER change this GUID: Windows recognises an install
; as an upgrade of an earlier one by this value alone. A new GUID would make
; every future setup a second, parallel installation with its own uninstall
; entry, leaving the old one behind.
AppId={{C1F0FF97-1FDE-4205-B122-627A9AD5C809}
; The displayed name - Apps & features, the wizard, the shortcuts. The machine
; names below it are deliberately the other one: the directory, the executable
; and the setup's own file name stay `growforge`, which is what an upgrade, a
; path and a typed command are matched on.
AppName=growforge 3D
AppVersion={#AppVersion}
DefaultDirName={autopf}\growforge
DefaultGroupName=growforge
UninstallDisplayIcon={app}\growforge.exe
DisableProgramGroupPage=yes
; Installs for all users, into Program Files, which is where a program that is
; never written to belongs - the configurations it edits live in the user's
; Documents. Overridable on the command line (`/CURRENTUSER`) so an unattended
; install - the build script's own test of this setup, a CI machine, an account
; without administrator rights - can put it under the profile instead without a
; dialog anyone has to answer.
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=commandline
; The executable is x86-64 only, so the installer refuses the architectures it
; would not run on and installs into the 64-bit Program Files rather than the
; 32-bit one.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; An upgrade over a running editor asks to close it rather than failing on a
; locked file or leaving a half-replaced install behind.
CloseApplications=yes
OutputDir=..\target\installer
OutputBaseFilename=growforge-setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; The shortcut names 0.40.0 changed. An upgrade does not remove the icons an
; earlier setup created, so without these two lines the old names sit beside the
; new ones until an uninstall clears both.
;
; The rule this block is held to: it only ever names *this installer's own
; previous shortcut files*, one literal path each, inside the group it created
; itself. No wildcards, no patterns, and never an entry under {autodesktop} or
; {userdesktop} - user shell state is untouchable, and a shortcut a user made
; and named themselves is exactly what a pattern here would delete.
Type: files; Name: "{group}\growforge.lnk"
Type: files; Name: "{group}\Uninstall growforge.lnk"

[Files]
Source: "..\target\release\growforge.exe"; DestDir: "{app}"; Flags: ignoreversion
; The shipped examples, as samples to open and copy rather than to work in:
; they sit beside the program in Program Files, and are marked read-only so an
; edit saved back over one is refused rather than silently changing what the
; documentation refers to. `uninsremovereadonly` is what makes an uninstall
; clean: the uninstaller does not delete a read-only file otherwise, and would
; leave the samples - and the folder holding them - behind.
;
; `overwritereadonly` is what makes an *upgrade* work, and it is not optional
; here: Setup refuses to replace a read-only file without it, so every upgrade
; over an install that has these samples aborted on the first one and rolled
; itself back. The attribute is re-applied by `Attribs` on the way in, so the
; samples land read-only again.
Source: "..\examples\*.toml"; DestDir: "{app}\examples"; Flags: ignoreversion overwritereadonly uninsremovereadonly; Attribs: readonly

[Icons]
; Both shortcuts start the editor with no path, which asks for the file in the
; platform's own dialog, starting in the growforge folder of the user's
; Documents. The working directory is that Documents folder rather than the
; install directory: a relative path typed into a configuration resolves
; somewhere writable, never into Program Files.
; The shortcut names are what a user reads, so they carry the product name; the
; file they point at is the executable, which does not.
Name: "{group}\growforge 3D"; Filename: "{app}\growforge.exe"; Parameters: "edit"; WorkingDir: "{userdocs}"
Name: "{group}\{cm:UninstallProgram,growforge 3D}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\growforge 3D"; Filename: "{app}\growforge.exe"; Parameters: "edit"; WorkingDir: "{userdocs}"; Tasks: desktopicon
