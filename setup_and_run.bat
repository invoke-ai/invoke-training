@echo off
setlocal enabledelayedexpansion

REM --- Configuration ---
set VENV_DIR=.venv
set PYTHON_CMD=python
set REQUIRE_PYTHON_VERSION_MAJOR=3
set REQUIRE_PYTHON_VERSION_MINOR=10
set TORCH_PACKAGE=torch==2.6.0+cu124
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
REM --- End Configuration ---

REM --- Helper Function Labels ---

:CHECK_PYTHON
    echo Checking for Python...
    %PYTHON_CMD% --version > nul 2>&1
    if %errorlevel% neq 0 (
        echo Checking for py command instead...
        py --version > nul 2>&1
        if %errorlevel% neq 0 (
            echo Error: Neither 'python' nor 'py' found in PATH.
            echo Please install Python %REQUIRE_PYTHON_VERSION_MAJOR%.%REQUIRE_PYTHON_VERSION_MINOR%+ and add it to your PATH.
            goto :ERROR_EXIT
        ) else (
            set PYTHON_CMD=py
            echo Found Python using 'py' command.
        )
    ) else (
        echo Found Python using 'python' command.
    )

    echo Checking Python version...
    for /f "tokens=1,2 delims=." %%a in ('%PYTHON_CMD% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do (
        set PYTHON_MAJOR=%%a
        set PYTHON_MINOR=%%b
    )

    if !PYTHON_MAJOR! LSS %REQUIRE_PYTHON_VERSION_MAJOR% (
        echo Error: Python %REQUIRE_PYTHON_VERSION_MAJOR% or higher is required. Found Major Version: !PYTHON_MAJOR!
        goto :ERROR_EXIT
    )
    if !PYTHON_MAJOR! EQU %REQUIRE_PYTHON_VERSION_MAJOR% (
        if !PYTHON_MINOR! LSS %REQUIRE_PYTHON_VERSION_MINOR% (
            echo Error: Python %REQUIRE_PYTHON_VERSION_MAJOR%.%REQUIRE_PYTHON_VERSION_MINOR% or higher is required. Found Version: !PYTHON_MAJOR!.!PYTHON_MINOR!
            goto :ERROR_EXIT
        )
    )
    echo Python check passed (%PYTHON_CMD% reports version !PYTHON_MAJOR!.!PYTHON_MINOR!).
goto :EOF

:INSTALL_APP
    echo --- Starting Installation ---
    call :CHECK_PYTHON || goto :ERROR_EXIT

    if exist "%VENV_DIR%\Scripts\activate.bat" (
        echo Existing virtual environment found. Skipping creation.
    ) else (
        echo Creating Python virtual environment in %VENV_DIR%...
        %PYTHON_CMD% -m venv %VENV_DIR%
        if %errorlevel% neq 0 (
            echo Failed to create virtual environment.
            goto :ERROR_EXIT
        )
    )

    echo Activating virtual environment for installation...
    call "%VENV_DIR%\Scripts\activate.bat"
    if %errorlevel% neq 0 (
        echo Failed to activate virtual environment.
        goto :ERROR_EXIT
    )

    echo Upgrading pip...
    pip install --upgrade pip
    if %errorlevel% neq 0 (
        echo Failed to upgrade pip.
        call :DEACTIVATE_VENV
        goto :ERROR_EXIT
    )

    echo Installing PyTorch (%TORCH_PACKAGE%)...
    pip install %TORCH_PACKAGE% --extra-index-url %TORCH_INDEX_URL%
    if %errorlevel% neq 0 (
        echo Failed to install PyTorch. Check internet connection and CUDA compatibility.
        call :DEACTIVATE_VENV
        goto :ERROR_EXIT
    )

    echo Installing invoke-training and its dependencies...
    pip install .
    if %errorlevel% neq 0 (
        echo Failed to install invoke-training.
        call :DEACTIVATE_VENV
        goto :ERROR_EXIT
    )

    call :DEACTIVATE_VENV
    echo --- Installation complete! ---
goto :EOF

:RUN_APP
    echo --- Starting Application ---
    if not exist "%VENV_DIR%\Scripts\activate.bat" (
        echo Error: Virtual environment not found at '%VENV_DIR%'.
        echo Please run the installation option first.
        goto :PAUSE_AND_RETURN
    )

    echo Activating virtual environment...
    call "%VENV_DIR%\Scripts\activate.bat"
    if %errorlevel% neq 0 (
        echo Failed to activate virtual environment.
        goto :PAUSE_AND_RETURN
    )

    echo Starting the Invoke Training UI (invoke-train-ui)...
    REM Add any command-line arguments needed by invoke-train-ui here if necessary
    invoke-train-ui

    echo Invoke Training UI finished.
    call :DEACTIVATE_VENV
    echo --- Application finished ---
goto :EOF

:REINSTALL_APP
    echo --- Reinstalling Application ---
    if exist "%VENV_DIR%" (
        echo Removing existing virtual environment '%VENV_DIR%'...
        rmdir /s /q "%VENV_DIR%"
        if %errorlevel% neq 0 (
           echo Error: Failed to remove existing virtual environment. It might be in use.
           goto :PAUSE_AND_RETURN
        )
        echo Existing environment removed.
    ) else (
        echo No existing virtual environment found to remove.
    )
    call :INSTALL_APP
goto :EOF

:DEACTIVATE_VENV
    REM Check if deactivate command exists (it should if venv is active)
    where deactivate > nul 2>&1
    if %errorlevel% equ 0 (
        echo Deactivating virtual environment...
        call deactivate
    )
goto :EOF

:PAUSE_AND_RETURN
    pause
goto :EOF

:ERROR_EXIT
    echo.
    echo An error occurred. Please check the messages above.
    pause
    exit /b 1

:EXIT_SCRIPT
    echo Exiting.
    goto :EOF

REM --- Main Menu Logic ---
:MENU
    cls
    echo =========================================
    echo  Invoke Training Setup & Run Menu
    echo =========================================
    echo  Virtual Environment Path: %VENV_DIR%
    set "CHOICE_1="
    set "CHOICE_2="
    set "CHOICE_3="

    if exist "%VENV_DIR%\Scripts\activate.bat" (
        echo  Status: Virtual environment detected.
        echo.
        echo  1. Run Invoke Training UI
        echo  2. Reinstall (Deletes and rebuilds %VENV_DIR%)
        echo  3. Exit
        set "CHOICE_1=:RUN_APP"
        set "CHOICE_2=:REINSTALL_APP"
        set "CHOICE_3=:EXIT_SCRIPT"
    ) else (
        echo  Status: Virtual environment NOT detected.
        echo.
        echo  1. Install Invoke Training
        echo  2. Exit
        set "CHOICE_1=:INSTALL_APP"
        set "CHOICE_2=:EXIT_SCRIPT"
    )
    echo =========================================

    set /p "USER_CHOICE=Enter your choice: "

    if "%USER_CHOICE%"=="1" (
        if defined CHOICE_1 (
            call %CHOICE_1%
            if %errorlevel% neq 0 goto :PAUSE_AND_RETURN
        ) else (
            echo Invalid choice.
            goto PAUSE_AND_GOTO_MENU
        )
    ) else if "%USER_CHOICE%"=="2" (
        if defined CHOICE_2 (
            call %CHOICE_2%
            if "%CHOICE_2%"==":EXIT_SCRIPT" goto :EOF
            if %errorlevel% neq 0 goto :PAUSE_AND_RETURN
        ) else (
            echo Invalid choice.
            goto PAUSE_AND_GOTO_MENU
        )
    ) else if "%USER_CHOICE%"=="3" (
        if defined CHOICE_3 (
            call %CHOICE_3%
            if "%CHOICE_3%"==":EXIT_SCRIPT" goto :EOF
            if %errorlevel% neq 0 goto :PAUSE_AND_RETURN
        ) else (
            echo Invalid choice.
            goto PAUSE_AND_GOTO_MENU
        )
    ) else (
        echo Invalid choice.
        goto PAUSE_AND_GOTO_MENU
    )

:PAUSE_AND_GOTO_MENU
    pause
    goto MENU

:END
endlocal
exit /b 0 