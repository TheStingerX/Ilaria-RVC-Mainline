@echo off
:menu
cls
echo Welcome to Ilaria RVC Mainline Assistant!
echo How can i help you today?
echo.
echo Please select an option:
echo 1. Run the update
echo 2. Download additional pretrain
echo 3. Exit
echo.
set /p userinp= "Enter your choice (1, 2 or 3): "
if /i "%userinp%" equ "1" (
    echo You have selected to run the update.
    python update.py
    pause
    goto menu
) else if /i "%userinp%" equ "2" (
    echo You have selected to download the additional pretrain.
    python download_pretrain.py
    pause
    goto menu
) else if /i "%userinp%" equ "3" (
    echo Exiting the program.
    exit
) else (
    echo Invalid choice. Please enter either 1, 2 or 3.
    pause
    goto menu
)
