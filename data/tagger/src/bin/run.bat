@echo off

set dataLoc=%1

:: Find Java binary
@set JAVA_BIN=%~dp0..\..\..\jre_x64\bin\java.exe
@if not exist "%JAVA_BIN%" >NUL 2>&1 set JAVA_BIN=%JAVA_HOME%\bin\java.exe
@if not exist "%JAVA_BIN%" >NUL 2>&1 set JAVA_BIN=%JAVA_HOME%\jre\bin\java.exe
@if not exist "%JAVA_BIN%" >NUL 2>&1 ( set JAVA_BIN=java.exe & where /q java.exe )
@if errorlevel 1 echo No Java found. Please put Java bin directory into your PATH environment or set JAVA_HOME environment variable to valid Java installation. && exit /b 1

:: Run the app
@"%JAVA_BIN%" -jar tagger-1.0-SNAPSHOT.jar --data.location=%dataLoc%