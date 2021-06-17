@echo off
set build_dir=build
IF NOT "%1"=="" (
    set build_dir=%1
)
set config=Debug
IF NOT "%2"=="" (
    set config=%2
)
cd ../../%build_dir%/lib/windows-x64/%config%/python/venv/Scripts
call activate.bat
cd ../../../../../../../tests/swig