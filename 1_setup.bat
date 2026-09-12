@echo off
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python not found on PATH. > setup_log.txt
  echo Python not found. If you use Anaconda, open "Anaconda Prompt", cd to this folder and run this file from there.
  pause
  exit /b 1
)
echo Installing libraries (this can take a few minutes)...
(
  echo ===== python =====
  %PY% --version
  %PY% -c "import sys; print(sys.executable)"
  echo ===== pip install =====
  %PY% -m pip install --upgrade scikit-learn pandas matplotlib joblib opencv-python yt-dlp
  echo ===== check torch =====
  %PY% -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  echo ===== check sklearn =====
  %PY% -c "import sklearn, cv2; print('sklearn', sklearn.__version__, 'cv2', cv2.__version__)"
  echo ===== DONE =====
) > setup_log.txt 2>&1
type setup_log.txt
echo.
echo Setup finished. Tell Claude "setup done".
pause
