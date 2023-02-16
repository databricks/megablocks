clean:
	rm -rf dist/*

dist: clean
	python3 setup.py sdist

upload: dist
	twine upload dist/*

upload-test: dist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
