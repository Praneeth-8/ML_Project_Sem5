@echo off
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python not found on PATH. > video_log.txt
  echo Python not found. If you use Anaconda, open "Anaconda Prompt", cd to this folder and run this file from there.
  pause
  exit /b 1
)
%PY% -c "import torch, torchvision" >nul 2>nul || (
  echo Installing PyTorch (one time, about 200-300 MB^)...
  %PY% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)
set /p URL=Paste the YouTube link (or just press Enter if the video file is already in this folder): 
set /p NAME=Short name for this video (e.g. robbery or normal): 
if "%NAME%"=="" set NAME=test_video
if not "%URL%"=="" (
  echo Downloading...
  %PY% -m pip install -q -U "yt-dlp[default]" >nul 2>nul
  %PY% -m yt_dlp --force-overwrites -f "b[height<=720]/bv*[height<=720]/b/bv*" -o "%NAME%.mp4" "%URL%" > video_log_%NAME%.txt 2>&1
)
if not exist "%NAME%.mp4" (
  echo.
  echo Could not find %NAME%.mp4 - the download failed.
  echo Fix: save the video yourself into this folder as %NAME%.mp4, run this file again,
  echo      press Enter at the link question and type %NAME% as the name.
  echo Download failed >> video_log_%NAME%.txt
  pause
  exit /b 1
)
echo Classifying (first run downloads DenseNet weights)...
%PY% -u test_youtube_video.py --video "%NAME%.mp4" --models results\models --window 10 >> video_log_%NAME%.txt 2>&1
type video_log_%NAME%.txt
echo.
echo Done. Tell Claude "video done: %NAME%".
pause
