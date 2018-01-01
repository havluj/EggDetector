#!/bin/bash

DIRECTORY=data
URL=http://athena.pef.czu.cz/ptacionline/

wget -o log.txt -nv --show-progress -c -P "$DIRECTORY" -r -np -nH --cut-dirs=1 -R index.html "$URL"
