# Build the GRSL letter and enforce its two hard gates: the five-page
# limit, and every number matching its source file in results/.
#
#   make          build, then check page count and numbers
#   make clean    remove build artifacts

PAPER = aegisbench_grsl

all: $(PAPER).pdf check

$(PAPER).pdf: $(PAPER).tex $(PAPER).bib figures/lowlight_collapse.pdf
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

figures/lowlight_collapse.pdf:
	cd ../.. && python3 scripts/grsl_figures.py

# GRSL rejects over-length submissions outright, so this is a build gate,
# not advice. Same for the number check: the letter's claim is traceability.
check: $(PAPER).pdf
	@pages=$$(pdfinfo $(PAPER).pdf | awk '/^Pages:/ {print $$2}'); \
	 if [ "$$pages" -gt 5 ]; then \
	   echo "FAIL: $$pages pages, GRSL allows 5"; exit 1; \
	 else echo "OK: $$pages pages"; fi
	@if grep -qE "Overfull \\\\hbox" $(PAPER).log; then \
	   echo "FAIL: overfull hbox, something runs past the column"; exit 1; \
	 else echo "OK: no overfull boxes"; fi
	@cd ../.. && python3 scripts/verify_grsl_numbers.py

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf pg-*.png

.PHONY: all check clean
