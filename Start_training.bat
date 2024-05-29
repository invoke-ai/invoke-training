@echo off 
cd %~dp0 
start cmd /k ".\invoketraining\Scripts\activate && invoke-train-ui"
