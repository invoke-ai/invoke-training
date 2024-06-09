@echo off
setlocal enabledelayedexpansion

:: Read the virtual environment path from config.txt
for /f "tokens=1,2 delims==" %%i in (config.txt) do (
    if "%%i"=="VENV_PATH" (
        set "VENV_PATH=%%j"
        set "VENV_PATH=!VENV_PATH:~0,-1!"
    )
    if "%%i"=="PYTHON_CMD" (
        set "PYTHON_CMD=%%j"
    )
)

if exist "%VENV_PATH%\Scripts\activate" (
    call "%VENV_PATH%\Scripts\activate"
) else (
    echo Virtual environment path "%VENV_PATH%" is invalid.
)
setlocal enabledelayedexpansion

:menu
echo.
echo Select a merge method:
echo 1. Merge Models
echo 2. Merge LoRA into Model
echo 3. Merge Task Models to Base Model
echo 4. Extract LoRA from Model Diff
echo 5. Open Terminal with venv
echo 6. Update venv Path
echo 7. Update Python Command
echo 8. Exit
set /p choice="Enter your choice: "

if "%choice%"=="1" goto merge_models
if "%choice%"=="2" goto merge_lora_into_model
if "%choice%"=="3" goto merge_task_models_to_base_model
if "%choice%"=="4" goto extract_lora_from_model_diff
if "%choice%"=="5" goto open_terminal_with_venv
if "%choice%"=="6" goto update_venv_path
if "%choice%"=="7" goto update_python_cmd
if "%choice%"=="8" goto end

:merge_models
set /p model_type="Enter model type (SD/SDXL): "
set /p models="Enter models (space-separated): "
set /p weights="Enter weights (space-separated): "
set /p method="Enter method (LERP/SLERP) [LERP]: "
if "%method%"=="" set method=LERP
set /p out_dir="Enter output directory [./output]: "
if "%out_dir%"=="" set out_dir=./output
set /p dtype="Enter dtype (float32/float16/bfloat16) [float16]: "
if "%dtype%"=="" set dtype=float16
set cmd=%PYTHON_CMD% src\invoke_training\model_merge\scripts\merge_models.py --model-type %model_type% --models %models% --weights %weights% --method %method% --out-dir %out_dir% --dtype %dtype%
echo %cmd%
%cmd%
goto end

:merge_lora_into_model
set /p model_type="Enter model type (SD/SDXL): "
set /p base_model="Enter base model: "
set /p lora_models="Enter LoRA models (space-separated): "
set /p output="Enter output directory: "
set /p save_dtype="Enter save dtype (float32/float16/bfloat16) [float16]: "
if "%save_dtype%"=="" set save_dtype=float16
set cmd=%PYTHON_CMD% src\invoke_training\model_merge\scripts\merge_lora_into_model.py --model-type %model_type% --base-model %base_model% --lora-models %lora_models% --output %output% --save-dtype %save_dtype%
echo %cmd%
%cmd%
goto end

:merge_task_models_to_base_model
set /p model_type="Enter model type (SD/SDXL): "
set /p base_model="Enter base model: "
set /p task_models="Enter task models (space-separated): "
set /p task_weights="Enter task weights (space-separated): "
set /p method="Enter method (TIES/DARE_LINEAR/DARE_TIES) [TIES]: "
if "%method%"=="" set method=TIES
set /p density="Enter density (0-1) [0.2]: "
if "%density%"=="" set density=0.2
set /p out_dir="Enter output directory: "
set /p dtype="Enter dtype (float32/float16/bfloat16) [float16]: "
if "%dtype%"=="" set dtype=float16
set cmd=%PYTHON_CMD% src\invoke_training\model_merge\scripts\merge_task_models_to_base_model.py --model-type %model_type% --base-model %base_model% --task-models %task_models% --task-weights %task_weights% --method %method% --density %density% --out-dir %out_dir% --dtype %dtype%
echo %cmd%
%cmd%
goto end

:extract_lora_from_model_diff
set /p model_type="Enter model type (SD/SDXL): "
set /p model_orig="Enter original model: "
set /p model_tuned="Enter tuned model: "
set /p save_to="Enter save to path: "
set /p load_precision="Enter load precision (float32/float16/bfloat16) [bfloat16]: "
if "%load_precision%"=="" set load_precision=bfloat16
set /p save_precision="Enter save precision (float32/float16/bfloat16) [float16]: "
if "%save_precision%"=="" set save_precision=float16
set /p lora_rank="Enter LoRA rank [4]: "
if "%lora_rank%"=="" set lora_rank=4
set /p clamp_quantile="Enter clamp quantile (0-1) [0.99]: "
if "%clamp_quantile%"=="" set clamp_quantile=0.99
set /p device="Enter device (cuda/cpu) [cuda]: "
if "%device%"=="" set device=cuda
set cmd=%PYTHON_CMD% src\invoke_training\model_merge\scripts\extract_lora_from_model_diff.py --model-type %model_type% --model-orig %model_orig% --model-tuned %model_tuned% --save-to %save_to% --load-precision %load_precision% --save-precision %save_precision% --lora-rank %lora_rank% --clamp-quantile %clamp_quantile% --device %device%
echo %cmd%
%cmd%
goto end

:open_terminal_with_venv
echo Opening a new terminal with the virtual environment activated...
if exist "%VENV_PATH%\Scripts\activate" (
    start cmd /k "call %VENV_PATH%\Scripts\activate"
) else (
    echo Virtual environment path "%VENV_PATH%" is invalid. Please update the path.
)
goto end

:update_venv_path
set /p new_venv_path="Enter new virtual environment path: "
if not "%new_venv_path%"=="" (
    echo VENV_PATH=%new_venv_path:~0,-1% > config.txt
    set "VENV_PATH=%new_venv_path%"
    if exist "%VENV_PATH%\Scripts\activate" (
        call "%VENV_PATH%\Scripts\activate"
        echo Virtual environment path updated to %VENV_PATH%
    ) else (
        echo Virtual environment path "%VENV_PATH%" is invalid. Please update the path.
    )
)
goto menu

:update_python_cmd
set /p new_python_cmd="Enter new Python command: "
if not "%new_python_cmd%"=="" (
    echo PYTHON_CMD=%new_python_cmd:~0,-1% >> config.txt
    set "PYTHON_CMD=%new_python_cmd%"
    echo Python command updated to %PYTHON_CMD%
)

:end
echo Exiting...
