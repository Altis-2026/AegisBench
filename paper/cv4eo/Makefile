# Build the CV4EO @ WACV 2027 submission with three hard gates:
#
#   1. body must end by page 8 (CV4EO full track: 8 pages excluding
#      references), and the PDF must not run past 9 pages total
#   2. no overfull boxes (a line past its column is a camera-ready defect
#      and reviewers read it as carelessness)
#   3. every number printed in the paper must match its source record in
#      results/, checked by scripts/verify_cv4eo_numbers.py
#
# A failure in any of the three fails the build rather than warning, so
# the PDF that exists is always one that passed all three.

PAPER  = main
LATEX  = pdflatex -interaction=nonstopmode -halt-on-error
VERIFY = ../../scripts/verify_cv4eo_numbers.py

.PHONY: all check clean

all: $(PAPER).pdf check

$(PAPER).pdf: $(PAPER).tex preamble.tex $(wildcard sec/*.tex) aegisbench.bib
	$(LATEX) $(PAPER) > /dev/null
	bibtex $(PAPER) > /dev/null || true
	$(LATEX) $(PAPER) > /dev/null
	$(LATEX) $(PAPER) > /dev/null

check: $(PAPER).pdf
	@python3 $(VERIFY)
	@total=$$(pdfinfo $(PAPER).pdf | awk '/^Pages:/{print $$2}'); \
	refpage=$$(for p in $$(seq 1 $$total); do \
	    if pdftotext -f $$p -l $$p $(PAPER).pdf - 2>/dev/null | grep -q '^References$$'; \
	    then echo $$p; break; fi; done); \
	if [ -z "$$refpage" ]; then echo "FAIL: no References section found"; exit 1; fi; \
	if [ $$refpage -gt 8 ]; then \
	    echo "FAIL: body runs past page 8 (references start on page $$refpage)"; exit 1; fi; \
	if [ $$total -gt 9 ]; then \
	    echo "FAIL: $$total pages total"; exit 1; fi; \
	echo "pages: body ends on p$$refpage, $$total total (limit: body <= 8)"
	@n=$$(grep -c "Overfull" $(PAPER).log || true); \
	if [ "$$n" -ne 0 ]; then \
	    echo "FAIL: $$n overfull box(es):"; grep -n "Overfull" $(PAPER).log; exit 1; fi; \
	echo "no overfull boxes"
	@if grep -qE '(^|[^\\])(Prem|Kamasani|Amogh|Vinaykumar|Flower Mound|Altis)' \
	    $(PAPER).tex sec/*.tex; then \
	    echo "FAIL: identifying text found (CV4EO review is double-blind)"; exit 1; fi; \
	echo "anonymity: no identifying text in sources"

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).brf $(PAPER).pdf
