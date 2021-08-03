#!/usr/bin/env python
# coding: utf-8

# Imports
import globals
import os

# Save all prints to a file
# This step is important. It serves to check the behavior
# of the algorithm later and collect the respective results

# Call function openFile
def openFile(filename):
	# Delete a file if it exists
	if os.path.exists(filename):
	    os.remove(filename)

	# Open a file
	globals.file = open(filename, 'a')

# Call function closeFile
def closeFile():
	# Close a file
	globals.file.close()