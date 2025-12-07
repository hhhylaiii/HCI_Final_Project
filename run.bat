@echo off
echo [Step 1] Running calibration.py ...
python Codes/calibration.py

echo.
echo [Step 2] Running main.py ...
python Codes/main.py

echo.
echo Done.
pause