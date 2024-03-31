@echo off
echo Installing audio-separator workaround
runtime\python.exe -m pip install audio-separator
runtime\python.exe -m pip install audio-separator[gpu]
echo Ilaria RVC is starting...
start /B runtime\python.exe infer-web.py --pycmd runtime\python.exe --port 7897
