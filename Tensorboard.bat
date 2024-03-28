@echo off
setlocal

REM Check if tensorboard_venv exists in the root directory
if not exist "%~dp0\tensorboard_venv\" (
    REM Print the step to the terminal
    echo Installing virtualenv...

    REM Install virtualenv
    %~dp0\runtime\python -m pip install virtualenv

    REM Print the step to the terminal
    echo Creating a virtual environment named tensorboard_venv...

    REM Create a virtual environment named tensorboard_venv in the root directory
    %~dp0\runtime\python -m virtualenv %~dp0\tensorboard_venv

    REM Print the step to the terminal
    echo Activating the virtual environment...

    REM Activate the virtual environment
    call %~dp0\tensorboard_venv\Scripts\activate

    REM Print the step to the terminal
    echo Installing TensorBoard into the virtual environment...

    REM Install TensorBoard into the virtual environment
    pip install tensorboard

    REM Downgrade problematic packages
    echo Downgrading packages for troubleshooting...
    pip install markdown==3.0
    pip install tensorboard==2.1.0
    pip install protobuf==3.11.0
    pip install numpy==1.19.5

) else (
    REM Print the step to the terminal
    echo tensorboard_venv already exists, skipping creation and activation...

    REM Activate the existing virtual environment
    call %~dp0\tensorboard_venv\Scripts\activate
)

REM Print the step to the terminal
echo Launching TensorBoard...

REM Launch TensorBoard
tensorboard --logdir="%~dp0\logs"

REM Print the step to the terminal
echo Keeping the command prompt open...

pause
endlocal
