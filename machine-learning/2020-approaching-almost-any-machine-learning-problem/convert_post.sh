#!/bin/bash

DIR_NAME=${1::-6}
FIG="_files"
MD=".md"
date=$(date '+%Y-%m-%d')

if [[ $1 -eq 0 ]]; then
        echo "No notebook name is supplied"
    else
        jupyter nbconvert $1 --to markdown --output-dir=docs/_posts
        find docs/_posts/"$DIR_NAME$FIG"/ -type f -exec mv {} docs/fig \;
        rm -rf docs/_posts/"$DIR_NAME$FIG"/
        cd docs/_posts
        sed -i "s|$DIR_NAME$FIG|{{ site.baseurl }}/fig|g" "$DIR_NAME$MD"
        mv $DIR_NAME$MD "$date-$DIR_NAME$MD"
fi

echo "Markdown conversion completed."

