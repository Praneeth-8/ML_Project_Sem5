@echo off
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not exist test_videos mkdir test_videos
echo Predicting Normal / Abnormal for every video inside the test_videos folder...
%PY% -u predict_normal_abnormal.py --folder test_videos --seconds 30 > predict_log.txt 2>&1
type predict_log.txt
echo.
echo Finished. Results are also in predictions.csv. Tell Claude "predictions done".
pause
