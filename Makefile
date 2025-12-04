PYTHON := python

CONFIG := configs/config.yaml
CHECKPOINT := artifacts/best.ckpt

.PHONY: install train lr_finder grid_search eval clean

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.train --config $(CONFIG)

lr_finder:
	$(PYTHON) -m src.lr_finder --config $(CONFIG)

grid_search:
	$(PYTHON) -m src.grid_search --config $(CONFIG)

eval:
	$(PYTHON) -m src.evaluate --config $(CONFIG) --checkpoint $(CHECKPOINT)

clean:
	rm -rf runs/* artifacts/*