build:
	docker-compose build

py-shell:
	docker-compose run package ipython

shell:
	docker-compose run package bash

test:
	docker-compose run package python -m unittest