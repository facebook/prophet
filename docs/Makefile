notebooks:
	for f in ../notebooks/*.ipynb; \
	do \
	    NAME=$$(basename $$f .ipynb); \
	    jupyter nbconvert --to markdown ../notebooks/$$NAME.ipynb --template=nbconvert_template.tpl; \
	    mv -f ../notebooks/"$$NAME".md _docs/; \
	    rm -rf static/"$$NAME"_files; \
	    mv ../notebooks/"$$NAME"_files static/; \
	done
