@echo off
title Iniciador del Simulador de Calidad del Aire
echo ========================================================
echo   Iniciando Simulador de Calidad del Aire...
echo ========================================================
echo.

:: Comprobar si Python esta instalado
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] No se ha encontrado Python en el sistema.
    echo Por favor, instala Python desde https://www.python.org/downloads/
    echo Asegurate de marcar la casilla "Add Python to PATH" durante la instalacion.
    pause
    exit /b
)

:: Definir el nombre del entorno virtual (se usa .venv o venv, elegiremos venv que esta en el directorio)
set VENV_DIR=venv

:: Crear entorno virtual si no existe
IF NOT EXIST "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creando el entorno virtual por primera vez...
    python -m venv %VENV_DIR%
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Hubo un problema al crear el entorno virtual.
        pause
        exit /b
    )
)

:: Activar el entorno virtual
echo [INFO] Activando el entorno virtual...
call "%VENV_DIR%\Scripts\activate.bat"

:: Instalar/actualizar dependencias de requirements.txt
IF EXIST "requirements.txt" (
    echo [INFO] Verificando e instalando dependencias. Puede tardar la primera vez.
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Hubo un problema al instalar las dependencias.
        pause
        exit /b
    )
) ELSE (
    echo [ADVERTENCIA] No se encontro el archivo requirements.txt.
)

:: Ejecutar la aplicacion
echo [INFO] Arrancando la aplicacion...
echo.
echo =======================================================================
echo    SIMULADOR DE CALIDAD DEL AIRE - CIUDAD UNIVERSITARIA UANL
echo =======================================================================
echo.
echo   Puedes ver la aplicacion en el navegador web.
echo.
echo    URL Local: http://localhost:8501
echo.
echo   ---------------------------------------------------------------------
echo    [IMPORTANTE] NO CIERRES ESTA VENTANA
echo    Cerrar la ventana apagará el servidor y el simulador
echo   ---------------------------------------------------------------------
echo.
echo    Para detener la aplicacion, presiona CTRL+C en esta ventana
echo    o simplemente cierra esta ventana.
echo.
echo =======================================================================
echo.

:: Redirigimos stdout a nul para ocultar los mensajes en ingles de Streamlit,
:: pero permitimos stderr para ver si ocurre algun error de ejecucion.
streamlit run app.py >nul

:: Si se cierra por algun motivo, pausar para ver errores
echo.
echo [INFO] La aplicacion se ha detenido.
echo.
pause
