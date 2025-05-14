SHELL := /bin/bash
.ONESHELL: # Make runs every command in a subshell so cd doesn't work unless used with && pipe.

get_data:
	@echo "Warning: kaggle.json API token needs to be placed to ~/.kaggle"
	@echo "Installing kaggle package..."
	pip install kaggle 
	cd data
	@echo "Getting original data"
	kaggle datasets download -d abhishek/aaamlp
	unzip aaamlp.zip
	rm aaamlp.zip
	cd ..
	@echo "Finished without any problems"

install:
	conda create -n aaamlp python=3.7.6
	conda activate aaamlp
 	pip install -r requirements.txt

clean:
	# If errors occur in setting up environment run this to clean conda
	conda remove -n aaamlp --all
	conda clean --all # This might break other environments, so do it at your own peril
