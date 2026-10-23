@echo off
chcp 65001 >nul
title QuantLab Student
cd /d "%~dp0"
echo ==========================================
echo   QuantLab Student - 正在启动...
echo   浏览器将自动打开 http://localhost:8501
echo   关闭本窗口即退出应用
echo ==========================================
python -m streamlit run app\Home.py
pause
