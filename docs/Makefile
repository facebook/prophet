.PHONY: api serve-api notebooks

API_MODULES = \
	../python/prophet \
	../python/prophet/diagnostics.py \
	../python/prophet/make_holidays.py \
	../python/prophet/models.py \
	../python/prophet/plot.py \
	../python/prophet/serialize.py \
	../python/prophet/utilities.py

api:
	rm -rf api
	python -m pdoc --docformat numpy --output-directory api $(API_MODULES)
	cp _includes/api-index.html api/index.html

serve-api: api
	cd api && python -m http.server 8080

notebooks:
	for f in ../notebooks/*.ipynb; \
	do \
	    NAME=$$(basename $$f .ipynb); \
	    jupyter nbconvert --to markdown ../notebooks/$$NAME.ipynb --template=nbconvert_template.tpl; \
	    mv -f ../notebooks/"$$NAME".md _docs/; \
	    rm -rf static/"$$NAME"_files; \
	    mv ../notebooks/"$$NAME"_files static/; \
	done
