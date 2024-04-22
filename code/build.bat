@echo off

IF NOT EXIST ..\.venv (
	py -m venv ..\.venv
	CALL ..\.venv\Scripts\activate
	py -m pip install --upgrade pip
	py -m pip install -r ..\requirements.txt
    py -m spacy download en_core_web_sm
	CALL deactivate
)

CALL ..\.venv\Scripts\activate

py .\utsj.py ..\data\textdata.txt

CALL deactivate
