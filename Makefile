# ----------------- #
#     Variables     #
# ----------------- #

PYTHON=uv run python

FLAKE8=uv run flake8
MYPY=uv run mypy

CHECK_UV=command -v uv
INSTALL_UV = curl -LsSf https://astral.sh/uv/install.sh | sh

MYPY_FLAGS=--warn-return-any --warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs \

# ----------------- #
#       Rules       #
# ----------------- #

.PHONY: all install run debug clean lint lint-strict


all: install run


install:
	@clear
	@if	! $(CHECK_UV) > /dev/null 2>&1; then \
			echo "UV not installed. Installing..."; \
			$(INSTALL_UV); \
	fi
	@echo "$(BROWN)Installing project dependencies using uv...$(END)"
	uv sync --link-mode=copy


run:
	@clear
	@echo "$(BLUE)Running the project...$(END)"
	$(PYTHON) -m src


debug:
	@clear
	@echo "$(BLUE)Running the project in debug...$(END)"
	$(PYTHON) -m pdb src $(JSON_FLAGS)


clean:
	@clear
	@echo "$(RED)Removing unecessary files from the folder...$(END)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf data/output
	rm -rf .venv
	@echo "\n$(GREEN)Folder cleaned!$(END)"


lint:
	@clear
	@status=0; \
	$(FLAKE8) src/ || status=$$?; \
	$(MYPY) src/ $(MYPY_FLAGS) || status=$$?; \
	exit $$status


lint-strict:
	@clear
	@status=0; \
	$(FLAKE8) src/ || status=$$?; \
	$(MYPY) src/ $(MYPY_FLAGS) --strict || status=$$?; \
	exit $$status


# ----------------- #
#       Colors      #
# ----------------- #

BROWN=\033[1;33m
BLUE=\033[1;94m
RED=\033[1;31m
GREEN=\033[1;92m
END=\033[0m
