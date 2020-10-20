@echo off
set build_dir=build
IF NOT "%1"=="" (
    set build_dir=%1
)
cd ../../%build_dir%/lib/windows-x64/python/venv/Scripts
call activate.bat
cd ../../../../../../tests/swig