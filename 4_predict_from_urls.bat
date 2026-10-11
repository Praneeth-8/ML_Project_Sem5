@echo off
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
echo Downloading the videos listed in urls.txt and predicting Normal / Abnormal for each...
%PY% -u predict_normal_abnormal.py --urls urls.txt --seconds 30 > predict_log.txt 2>&1
type predict_log.txt
echo.
echo Finished. Results are also in predictions.csv. Tell Claude "predictions done".
pause
