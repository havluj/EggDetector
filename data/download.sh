#!/bin/bash

DIRECTORY=data
URL=http://athena.pef.czu.cz/ptacionline/134572snaps/

if [ -d "$DIRECTORY" ]; then
  rm -rf "$DIRECTORY"
fi

mkdir "$DIRECTORY"
wget -o log.txt -nv --show-progress -c -P "$DIRECTORY" -r -np -nH --cut-dirs=2 -R index.html "$URL"
