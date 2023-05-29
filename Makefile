.PHONY: run
run:
	python -m main

.PHONY: data
data:
	python -m data $(filter-out $@, $(MAKECMDGOALS))

%:
	@true