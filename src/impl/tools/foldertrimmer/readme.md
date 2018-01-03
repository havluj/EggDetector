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
- Run with `target/run.sh` on Linux or `target\run.bat` on Windows. Both scripts accept two parameters:
    - First parameter can be `true` or `false` and it determines whether you want to delete
    everything except manually tagged data (containing imgdata.xml) in case of `true` or if you just want to
    delete the useless data in case of `false`.
    - Second parameter specifies the root folder of the data that should be trimmed. 
    - For example to delete only useless data but keep all the folders with
    some image data, use: `target\run.bat false C:\school\bakalarka\data`
    - To delete everything except tagged data, run: `target\run.bat true C:\school\bakalarka\data`