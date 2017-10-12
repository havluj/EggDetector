#!/bin/bash

DIRECTORY=data
URL=http://athena.pef.czu.cz/ptacionline/134572snaps/

if [ -d "$DIRECTORY" ]; then
  rm -rf "$DIRECTORY"
fi

mkdir "$DIRECTORY"
#cd "$DIRECTORY"
wget -o log.txt -nv -c -P "$DIRECTORY" -r -np -nH --cut-dirs=2 -R index.html "$URL"
#cd ..
