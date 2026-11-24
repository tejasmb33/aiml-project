PYTHON ?= python
CONSENT ?= data/consent.json
EVAL_MANIFEST ?= data/evaluation.json
TRAIN_MANIFEST ?= data/training.json
OUTPUT ?= artifacts/evaluation
DETECTOR ?= hog
TOLERANCE ?= 0.6
CONFIDENCE ?= 0.4

.PHONY: run reproduce encode recognize evaluate train summary test
run: encode
	$(PYTHON) -m src.evaluate --manifest $(EVAL_MANIFEST) --consent $(CONSENT) --output $(OUTPUT) --detector $(DETECTOR) --tolerance $(TOLERANCE) --confidence-threshold $(CONFIDENCE)
encode:
	$(PYTHON) -m src.encode --dataset dataset --consent $(CONSENT)
recognize:
	$(PYTHON) -m src.recognize --consent-confirmed --detector $(DETECTOR) --tolerance $(TOLERANCE) --confidence-threshold $(CONFIDENCE)
evaluate:
	$(PYTHON) -m src.evaluate --manifest $(EVAL_MANIFEST) --consent $(CONSENT) --output $(OUTPUT) --detector $(DETECTOR) --tolerance $(TOLERANCE) --confidence-threshold $(CONFIDENCE)
train:
	$(PYTHON) -m src.vgg_localizer --manifest $(TRAIN_MANIFEST) --consent $(CONSENT)
summary:
	$(PYTHON) -m src.vgg_localizer --summary
test:
	$(PYTHON) -m pytest -q

# Complete protocol after all consented manifests and independent captures exist.
reproduce:
	$(PYTHON) -m pytest -q
	$(PYTHON) -m src.vgg_localizer --manifest $(TRAIN_MANIFEST) --consent $(CONSENT)
	$(PYTHON) -m src.encode --dataset dataset --consent $(CONSENT)
	$(PYTHON) -m src.evaluate --manifest $(EVAL_MANIFEST) --consent $(CONSENT) --training-split artifacts/localizer/split.json --output $(OUTPUT) --detector $(DETECTOR) --tolerance $(TOLERANCE) --confidence-threshold $(CONFIDENCE)
