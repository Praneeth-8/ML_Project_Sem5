@echo off
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python not found on PATH. > train_log.txt
  echo Python not found. If you use Anaconda, open "Anaconda Prompt", cd to this folder and run this file from there.
  pause
  exit /b 1
)
echo Training all models. This can take 10-40 minutes. Keep this window open.
echo Progress is also written to train_log.txt
%PY% -u train_classifiers.py --data all_classes_dataset_2048.npz > train_log.txt 2>&1
type train_log.txt
echo.
echo Training finished. Tell Claude "training done".
pause
