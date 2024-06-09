#!/bin/bash

VENV_PATH=$(echo "../venv" | xargs)
if [ ! -z "$1" ]; then
    VENV_PATH=$(echo "$1" | xargs)
fi

PYTHON_CMD=$(grep 'PYTHON_CMD' config.txt | cut -d '=' -f2 | tail -n 1 | xargs)
source "$VENV_PATH/bin/activate"

while true; do
    echo
    echo "Select a merge method:"
    echo "1. Merge Models"
    echo "2. Merge LoRA into Model"
    echo "3. Merge Task Models to Base Model"
    echo "4. Extract LoRA from Model Diff"
    echo "5. Open Terminal with venv"
    echo "6. Update venv Path"
    echo "7. Update Python Command"
    echo "8. Exit"
    read -p "Enter your choice: " choice

    case $choice in
        1)
            read -p "Enter model type (SD/SDXL): " model_type
            read -p "Enter models (space-separated): " models
            read -p "Enter weights (space-separated): " weights
            read -p "Enter method (LERP/SLERP) [LERP]: " method
            method=${method:-LERP}
            read -p "Enter output directory [./output]: " out_dir
            out_dir=${out_dir:-./output}
            read -p "Enter dtype (float32/float16/bfloat16) [float16]: " dtype
            dtype=${dtype:-float16}
            cmd="$PYTHON_CMD src/invoke_training/model_merge/scripts/merge_models.py --model-type $model_type --models $models --weights $weights --method $method --out-dir $out_dir --dtype $dtype"
            echo $cmd
            $cmd
            ;;
        2)
            read -p "Enter model type (SD/SDXL): " model_type
            read -p "Enter base model: " base_model
            read -p "Enter LoRA models (space-separated): " lora_models
            read -p "Enter output directory: " output
            read -p "Enter save dtype (float32/float16/bfloat16) [float16]: " save_dtype
            save_dtype=${save_dtype:-float16}
            cmd="$PYTHON_CMD src/invoke_training/model_merge/scripts/merge_lora_into_model.py --model-type $model_type --base-model $base_model --lora-models $lora_models --output $output --save-dtype $save_dtype"
            echo $cmd
            $cmd
            ;;
        3)
            read -p "Enter model type (SD/SDXL): " model_type
            read -p "Enter base model: " base_model
            read -p "Enter task models (space-separated): " task_models
            read -p "Enter task weights (space-separated): " task_weights
            read -p "Enter method (TIES/DARE_LINEAR/DARE_TIES) [TIES]: " method
            method=${method:-TIES}
            read -p "Enter density (0-1) [0.2]: " density
            density=${density:-0.2}
            read -p "Enter output directory: " out_dir
            read -p "Enter dtype (float32/float16/bfloat16) [float16]: " dtype
            dtype=${dtype:-float16}
            cmd="$PYTHON_CMD src/invoke_training/model_merge/scripts/merge_task_models_to_base_model.py --model-type $model_type --base-model $base_model --task-models $task_models --task-weights $task_weights --method $method --density $density --out-dir $out_dir --dtype $dtype"
            echo $cmd
            $cmd
            ;;
        4)
            read -p "Enter model type (SD/SDXL): " model_type
            read -p "Enter original model: " model_orig
            read -p "Enter tuned model: " model_tuned
            read -p "Enter save to path: " save_to
            read -p "Enter load precision (float32/float16/bfloat16) [bfloat16]: " load_precision
            load_precision=${load_precision:-bfloat16}
            read -p "Enter save precision (float32/float16/bfloat16) [float16]: " save_precision
            save_precision=${save_precision:-float16}
            read -p "Enter LoRA rank [4]: " lora_rank
            lora_rank=${lora_rank:-4}
            read -p "Enter clamp quantile (0-1) [0.99]: " clamp_quantile
            clamp_quantile=${clamp_quantile:-0.99}
            read -p "Enter device (cuda/cpu) [cuda]: " device
            device=${device:-cuda}
            cmd="$PYTHON_CMD src/invoke_training/model_merge/scripts/extract_lora_from_model_diff.py --model-type $model_type --model-orig $model_orig --model-tuned $model_tuned --save-to $save_to --load-precision $load_precision --save-precision $save_precision --lora-rank $lora_rank --clamp-quantile $clamp_quantile --device $device"
            echo $cmd
            $cmd
            ;;
        5)
            echo "Opening a new terminal with the virtual environment activated..."
            gnome-terminal -- bash -c "source $VENV_PATH/bin/activate; exec bash"
            ;;
        6)
            read -p "Enter new virtual environment path: " new_venv_path
            VENV_PATH=$(echo ${new_venv_path:-$VENV_PATH} | xargs)
            source "$VENV_PATH/bin/activate"
            echo "Virtual environment path updated to $VENV_PATH"
            ;;
        7)
            read -p "Enter new Python command: " new_python_cmd
            PYTHON_CMD=$(echo ${new_python_cmd:-$PYTHON_CMD} | xargs)
            echo "PYTHON_CMD=$PYTHON_CMD" > config.txt
            echo "Python command updated to $PYTHON_CMD"
            ;;
        8)
            echo "Exiting..."
            break
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
done
