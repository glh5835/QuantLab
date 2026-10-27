@echo off
chcp 65001 >nul
setlocal
title QuantLab Student
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHON=%~dp0.venv\Scripts\python.exe"

echo ==========================================
echo   QuantLab Student
echo ==========================================

if not exist "%PYTHON%" goto missing_venv
"%PYTHON%" -c "import sys" >nul 2>&1
if errorlevel 1 goto broken_venv

echo 正在检查项目虚拟环境和依赖...
"%PYTHON%" -c "import importlib.util, sys; names='streamlit pandas numpy plotly requests yfinance openpyxl xlrd sklearn joblib scipy matplotlib reportlab pytest pypdf'.split(); missing=[n for n in names if importlib.util.find_spec(n) is None]; print('Missing modules: ' + ', '.join(missing)) if missing else None; sys.exit(bool(missing))"
if errorlevel 1 goto missing_deps
"%PYTHON%" -m pip check
if errorlevel 1 goto missing_deps

echo 正在启动，请以终端显示的地址为准。
echo 关闭本窗口即退出应用。
"%PYTHON%" -m streamlit run app\Home.py
if errorlevel 1 goto launch_failed
exit /b 0

:missing_venv
echo.
echo 未找到项目虚拟环境 .venv，未使用全局 Python。
echo 请在项目文件夹打开 PowerShell，运行：
echo   py -3.11 -m venv .venv
echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
echo 如未安装 Python 3.11，请参阅 README 的 Windows 安装说明。
pause
exit /b 1

:broken_venv
echo.
echo 找到了 .venv，但其中的 Python 无法运行。
echo 请参阅 README 检查 Python 安装和虚拟环境。
echo 若需重建环境，请先备份或确认 .venv 中没有需要保留的内容。
pause
exit /b 1

:missing_deps
echo.
echo 项目虚拟环境缺少依赖、pip 不可用或存在依赖冲突。
echo 请在项目文件夹打开 PowerShell，运行：
echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
echo   .\.venv\Scripts\python.exe -m pip check
echo 如果仍失败，请查看上方错误并参阅 README 的常见安装问题。
pause
exit /b 1

:launch_failed
echo.
echo 应用启动失败，请查看上方错误信息，并参阅 README 的常见安装问题。
pause
exit /b 1
