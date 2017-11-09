#Data tagged
Data tagger is a command line tool used for data tagging used to train a neural network.

The data strictly need to follow the structure of <http://athena.pef.czu.cz/ptacionline/134572snaps/>

## How to use
- Compile (if needed) the app with `mvn package`
- Run with `run.sh` on Linux or `run.bat` on Windows. Both scripts accept data location as
the first and only parameter.
    - For example: `run.bat C:\school\bakalarka\data`