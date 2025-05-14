.PHONY: clean setup create_environment run-spark stop-spark

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

ifeq (,$(shell which uv))
HAS_UV=False
else
HAS_UV=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Setup whole project
setup:
	$(MAKE) create_environment

## Setup virtual environment, check uv
create_environment:
ifeq (True,$(HAS_UV))
	@echo "uv found, installing"
	uv sync
	@echo "Repo is installed in editable mode in virtual environment"
else
	@echo "uv NOT found, installing"
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync
endif

## Clean compiled files and strip notebook outputs preserving metadata
clean:
	find ./books/ -type f -name  "*.py[co]" -not -path "./.venv/*" -exec rm -rf {} \;
	find ./books/ -type d -name "__pycache__"  -not -path "./.venv/*" -exec rm -rf {} \;
	find ./books/ -type d -name ".ipynb_checkpoints" -not -path "./.venv/*" -exec rm -rf {} \;
	uv run nb-clean clean -n ./books/**

## Run a standalone spark
run-spark:
	@echo "requires 'spark-3.5.5-bin-hadoop3' download from apache spark websites"
	./spark-3.5.5-bin-hadoop3/sbin/start-all.sh

## Stop a standalone spark
stop-spark:
	./spark-3.5.5-bin-hadoop3/sbin/stop-all.sh
	

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
