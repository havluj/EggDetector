# Folder trimmer
Folder trimmer is a command line tool for deleting useless data the download
script downloads.

## What does it do
- Deletes files that do not end on either `.png`, `.xml`,
or `.txt`. 
- If a folder does not contain another folder or at least one of the files
listed above, the folder is deleted.

## How to use
- Compile (if needed) the app with `mvn clean package`
- Run with `target/run.sh` on Linux or `target\run.bat` on Windows. Both scripts accept data location as
the first and only parameter.
    - For example: `target\run.bat C:\school\bakalarka\data`