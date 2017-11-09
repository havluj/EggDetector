# How to obtain the data
- Data location: [link](http://athena.pef.czu.cz/ptacionline/134572snaps/) http://athena.pef.czu.cz/ptacionline/134572snaps/ 


## Downloading in bulk
- Download everything in the directory listing using wget: `wget -o log.txt -nv --show-progress -c -P "$DIRECTORY" -r -np -nH --cut-dirs=2 -R index.html "$URL"`
	- It will download all files and subfolders in `$DIRECTORY`:
		- saving a log file to `log.txt` (-o log.txt)
		- with non-verbose output (showing only errors) (-nv)
		- showing progress bar (--show-progress)
		- continue downloading not finished files from previous download (-c)
		- recursively (-r),
		- not going to upper directories, like `ptacionline` (-np),
		- not saving files to hostname folder (-nH),
		- but to `$DIRECTORY` by omitting first 2 folders `ptacionline`, `134572snaps` (--cut-dirs=3)
		- excluding index.html files (-R index.html)
		
## Use the data tagger to tag the data
Use the application to tag the downloaded data. The information will then be used to train a neural network: [Data Tagger](tagger/readme.md)
