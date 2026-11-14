# Windows 安装包

输出文件：项目父目录的 `安装包/PCBInspector-Setup-1.0.0-win64.exe`。

在项目目录执行：

```powershell
.venv\Scripts\python.exe packaging\build_installer.py
```

编译工具使用 Inno Setup 7，安装位置为 `packaging/tools/InnoSetup`。
脚本直接读取现有 Python 3.11 和虚拟环境中的依赖，不复制虚拟环境、不依赖其绝对路径。
数据库通过 SQLite backup API 生成一致快照，包含 records、learning、platform；首次启动将数据库与 JSON/YAML 中的项目路径迁移到安装目录。

安装面向 Windows 10/11 x64，安装在当前用户目录，不需要管理员权限。
包含 CPU 版 Torch、Ultralytics、OpenCV、Flask、训练代码、模型、训练图库及检验记录。
原始下载数据集、重复压缩包、Git、开发虚拟环境和临时训练目录不分发。
Microsoft C++ 运行库随 Python 采用应用目录部署。

桌面/开始菜单快捷方式调用 `start-installed.vbs`，用安装目录中的 `runtime/pythonw.exe` 启动 `launcher.py`。
启动器使用安装目录专用互斥锁避免重复进程，等待 HTTP 服务就绪后打开浏览器。
开始菜单提供停止服务与打开数据目录入口，卸载和升级前会先停止本安装实例及其训练子进程。
卸载保留 data 文件夹，升级不会覆盖已有数据。

静默安装验证示例（请指定独立测试目录）：

```powershell
PCBInspector-Setup-1.0.0-win64.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="C:\path\to\test-install"
C:\path\to\test-install\runtime\python.exe C:\path\to\test-install\launcher.py --no-browser --port 18765
```

首次启动日志：安装目录 `data/desktop-logs/startup.log`。
本机使用独立目录验证了安装、数据路径迁移、模型加载、检测接口、重复启动和停止服务。
换硬件的相机与驱动兼容性仍需在目标电脑实测。
