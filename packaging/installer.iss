; Audio2MIDI installer (Inno Setup 6.5+). Packs dist\audio2midi, built by build_runtime.ps1.
;
; Build:  ISCC.exe packaging\installer.iss            (best compression, slow)
;         ISCC.exe /DComp=lzma2/fast packaging\installer.iss   (quick, to try things out)

#ifndef Comp
  #define Comp "lzma2/max"
#endif
#define AppName "Audio2MIDI"
#define AppVersion "0.1.0"

[Setup]
AppId={{A92968EA-B683-4EAA-B2D5-E8E1DF466887}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=OzLoup
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename={#AppName}-Setup-{#AppVersion}
; per-user by default (no administrator needed, and the folder stays writable); the user may choose all users
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression={#Comp}
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=6
WizardStyle=modern
UninstallDisplayName={#AppName}
; room for the app (~5 GB). The models (~5 GB more) go to %LOCALAPPDATA%\audio2midi when they are downloaded.
ExtraDiskSpaceRequired=0

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "Crear un acceso directo en el escritorio"; Flags: unchecked

[Files]
Source: "..\dist\audio2midi\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\runtime\main\pythonw.exe"; Parameters: "-s ""{app}\launch.pyw"""; WorkingDir: "{app}"; Comment: "Audio a MIDI, pistas y hojas de acordes"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\runtime\main\pythonw.exe"; Parameters: "-s ""{app}\launch.pyw"""; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\runtime\main\pythonw.exe"; Parameters: "-s ""{app}\launch.pyw"""; WorkingDir: "{app}"; Description: "Abrir {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Python writes .pyc files while running, and the uninstaller only removes what it installed
Type: filesandordirs; Name: "{app}"
