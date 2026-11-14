#ifndef ProjectRoot
  #error ProjectRoot must be supplied by build_installer.py
#endif
#ifndef PythonRoot
  #error PythonRoot must be supplied by build_installer.py
#endif

[Setup]
AppId={{52D4A723-E623-4BAF-BE39-8EA970D00B7E}
AppName=焊视 PCB 焊接质量检测
AppVersion=1.0.0
AppPublisher=PCB Inspector
DefaultDirName={localappdata}\Programs\PCBInspector
DefaultGroupName=焊视 PCB 检测
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#ProjectRoot}\..\安装包
OutputBaseFilename=PCBInspector-Setup-1.0.0-win64
Compression=zip/1
SolidCompression=no
WizardStyle=modern
UninstallDisplayIcon={app}\runtime\pythonw.exe
CloseApplications=no
SetupLogging=yes
InfoBeforeFile={#ProjectRoot}\packaging\install-notes.txt
DiskSpanning=no

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; Flags: checkedonce

[Files]
Source: "{#ProjectRoot}\*.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\static\*"; DestDir: "{app}\static"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc"
Source: "{#ProjectRoot}\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#ProjectRoot}\training\*"; DestDir: "{app}\training"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc,runs\*,modelscope\*"
Source: "{#ProjectRoot}\training\missing_connection\runs\trial_fit-2\weights\best.pt"; DestDir: "{app}\training\missing_connection\runs\trial_fit-2\weights"; Flags: ignoreversion
Source: "{#PythonRoot}\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "Lib\site-packages\*,Lib\test\*,Lib\idlelib\*,Doc\*,Scripts\*,Tools\*,__pycache__\*,*.pyc"
; App-local Microsoft C++ runtime for native Torch/OpenCV extensions.
Source: "{#GetEnv('SystemRoot')}\System32\msvcp140*.dll"; DestDir: "{app}\runtime"; Flags: ignoreversion
Source: "{#GetEnv('SystemRoot')}\System32\vcruntime140*.dll"; DestDir: "{app}\runtime"; Flags: ignoreversion
Source: "{#GetEnv('SystemRoot')}\System32\concrt140.dll"; DestDir: "{app}\runtime"; Flags: ignoreversion
Source: "{#GetEnv('SystemRoot')}\System32\vcomp140.dll"; DestDir: "{app}\runtime"; Flags: ignoreversion
Source: "{#ProjectRoot}\.venv\Lib\site-packages\*"; DestDir: "{app}\runtime\Lib\site-packages"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc,torch\include\*,torch\share\*,*.lib"
Source: "{#ProjectRoot}\packaging\launcher.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\packaging\start-installed.vbs"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\packaging\install-notes.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\packaging\build\install-manifest.json"; DestDir: "{app}"; Flags: ignoreversion
; User records and model history survive both upgrades and uninstall.
Source: "{#ProjectRoot}\data\learning\*"; DestDir: "{app}\data\learning"; Flags: onlyifdoesntexist uninsneveruninstall recursesubdirs createallsubdirs; Excludes: "*.sqlite,*.sqlite-*,*.lock,__pycache__\*,*.pyc"
Source: "{#ProjectRoot}\data\platform\*"; DestDir: "{app}\data\platform"; Flags: onlyifdoesntexist uninsneveruninstall recursesubdirs createallsubdirs; Excludes: "*.sqlite,*.sqlite-*,*.log,*.lock,__pycache__\*,*.pyc"
Source: "{#ProjectRoot}\packaging\build\data\*"; DestDir: "{app}\data"; Flags: onlyifdoesntexist uninsneveruninstall recursesubdirs createallsubdirs

[Icons]
Name: "{group}\焊视 PCB 检测"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\start-installed.vbs"""; WorkingDir: "{app}"; IconFilename: "{app}\runtime\pythonw.exe"
Name: "{group}\停止焊视后台服务"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\start-installed.vbs"" --stop"; WorkingDir: "{app}"
Name: "{group}\打开数据文件夹"; Filename: "{app}\data"
Name: "{group}\卸载焊视 PCB 检测"; Filename: "{uninstallexe}"
Name: "{autodesktop}\焊视 PCB 检测"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\start-installed.vbs"""; WorkingDir: "{app}"; IconFilename: "{app}\runtime\pythonw.exe"; Tasks: desktopicon

[Run]
Filename: "{sys}\wscript.exe"; Parameters: """{app}\start-installed.vbs"""; Description: "启动焊视 PCB 检测"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launcher.py"" --stop"; Flags: runhidden waituntilterminated

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  Result := '';
  if FileExists(ExpandConstant('{app}\launcher.py')) then
    Exec(ExpandConstant('{app}\runtime\pythonw.exe'), '"' + ExpandConstant('{app}\launcher.py') + '" --stop', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;
