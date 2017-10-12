# How to obtain the data

- Data location: [link](http://athena.pef.czu.cz/ptacionline/134572snaps/) http://athena.pef.czu.cz/ptacionline/134572snaps/ 


## Downloading in bulk
- Download everything in the directory listing using wget: `wget -r -np -nH --cut-dirs=1 -R index.html http://athena.pef.czu.cz/ptacionline/134572snaps/`
	- Explanation of `wget -r -np -nH --cut-dirs=3 -R index.html http://hostname/aaa/bbb/ccc/ddd/`:
		- It will download all files and subfolders in ddd directory:
			- recursively (-r),
			- not going to upper directories, like ccc/… (-np),
			- not saving files to hostname folder (-nH),
			- but to ddd by omitting first 3 folders aaa, bbb, ccc (--cut-dirs=3)
			- excluding index.html files (-R index.html)
