@echo off
chcp 65001 >nul
setlocal
rem run_test.bat - 개발·검증용 실행 메뉴 (조원 4)
rem 시연용 원클릭 실행은 run_demo.bat (조원 5) 을 쓴다.
rem 번호를 인자로 주면 메뉴 없이 바로 실행한다.  예) run_test.bat 1   (push 전 코드 점검)

cd /d "%~dp0"
set PYTHONUTF8=1
set PORT=8000
set URL=http://localhost:%PORT%
set RC=0

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo [주의] venv 폴더가 없습니다. 시스템 파이썬으로 실행합니다.
    echo        README 2번 "코드 받고 가상환경 만들기"를 먼저 하세요.
    echo.
)

set "ONCE="
if not "%~1"=="" (
    set "choice=%~1"
    set "ONCE=1"
    goto dispatch
)

:menu
echo.
echo ============================================================
echo   SOAR 테스트 메뉴
echo ============================================================
echo   1. 코드 점검        (smoke_test, 카메라·모델 없이)
echo   2. 환경 점검        (check_env)
echo   3. 콘솔 실시간 인식 (realtime.py, q=종료)
echo   4. 서버 + 브라우저  (server.py --auto-start --debug)
echo   5. 서버 상태 보기   (GET /api/status)
echo   6. 워커 오류 주입   (NFR-05 시험, 4번 실행 중일 때)
echo   7. 세션 종료/시작   (S07 / S00 시험, 4번 실행 중일 때)
echo   0. 끝내기
echo ============================================================
set "choice="
set /p "choice=번호를 고르세요: "

:dispatch
if "%choice%"=="1" goto smoke
if "%choice%"=="2" goto env
if "%choice%"=="3" goto realtime
if "%choice%"=="4" goto server
if "%choice%"=="5" goto status
if "%choice%"=="6" goto fault
if "%choice%"=="7" goto session
if "%choice%"=="0" goto end
echo 잘못 고르셨습니다: "%choice%"
set RC=2
goto next

:smoke
python tests\smoke_test.py
if errorlevel 1 (
    echo.
    echo [FAIL] 실패 항목이 있습니다. push 하지 마세요.
    set RC=1
) else (
    echo.
    echo [ OK ] 전부 통과. push 해도 됩니다.
)
goto next

:env
python setup\check_env.py
goto next

:realtime
python src\realtime.py
goto next

:server
echo 서버를 새 창에서 켭니다. 끄려면 그 창에서 Ctrl+C.
start "SOAR server" cmd /k python src\server.py --auto-start --debug --port %PORT%
echo 서버 준비를 기다리는 중 (모델 불러오기에 10초 이상 걸릴 수 있음)...
for /l %%i in (1,1,60) do (
    curl -s -o nul %URL%/health && goto server_ready
    timeout /t 1 /nobreak >nul
)
echo [FAIL] 60초 안에 서버가 응답하지 않았습니다. 서버 창의 오류 메시지를 확인하세요.
set RC=1
goto next
:server_ready
echo [ OK ] 서버 준비 완료. 브라우저를 엽니다: %URL%
start "" %URL%
echo.
echo 예외 시험 순서 (NFR-05): 카메라 케이블 뽑기 -^> 안내 확인 -^> 다시 꽂기 -^> 자동 복구 확인
echo                          메뉴 6번으로 워커 오류 주입 -^> 자동 재시작 확인
goto next

:status
curl -s %URL%/api/status
if errorlevel 1 (
    echo [FAIL] 서버가 꺼져 있습니다. 메뉴 4번으로 먼저 켜세요.
    set RC=1
)
echo.
goto next

:fault
curl -s -X POST %URL%/api/debug/fault
if errorlevel 1 (
    echo [FAIL] 서버가 꺼져 있습니다. 메뉴 4번으로 먼저 켜세요.
    set RC=1
) else (
    echo.
    echo 주입했습니다. 서버 창에 오류가 찍히고 1초 뒤 "다시 정상 동작합니다"가 나와야 합니다.
)
goto next

:session
set "act=%~2"
if not defined act set /p "act=stop 또는 start 를 입력하세요: "
if /i not "%act%"=="stop" if /i not "%act%"=="start" (
    echo stop 또는 start 만 입력할 수 있습니다.
    set RC=2
    goto next
)
curl -s -X POST -H "Content-Type: application/json" -d "{\"action\":\"%act%\"}" %URL%/api/session
if errorlevel 1 (
    echo [FAIL] 서버가 꺼져 있습니다. 메뉴 4번으로 먼저 켜세요.
    set RC=1
)
echo.
goto next

:next
if defined ONCE goto end
goto menu

:end
endlocal & exit /b %RC%
