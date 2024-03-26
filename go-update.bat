@echo off
echo Welcome to Ilaria RVC Mainline Updater!
echo.
set /p userinp= "Do you want to run the update? (y/n): "
if /i "%userinp%" equ "y" (
    python update.py
) else (
    echo Update not executed.
)
pause
