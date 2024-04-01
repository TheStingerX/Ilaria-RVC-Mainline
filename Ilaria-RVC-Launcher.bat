@echo off
echo Ensuring audio-separator is installed before startup. Ignore the following notices.
runtime\python.exe -m pip install --quiet audio-separator
runtime\python.exe -m pip install --quiet audio-separator[gpu]
echo Ilaria RVC Starting...

start /B runtime\python.exe infer-web.py --pycmd runtime\python.exe --port 7897
