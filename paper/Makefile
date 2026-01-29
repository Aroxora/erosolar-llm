# LaTeX Paper Makefile
# Author: Bo Shang <bo@shang.software>

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean arxiv

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Create arXiv submission package
arxiv: $(MAIN).tex
	mkdir -p arxiv_submission
	cp $(MAIN).tex arxiv_submission/
	cp *.bib arxiv_submission/ 2>/dev/null || true
	cd arxiv_submission && tar -czvf ../arxiv_submission.tar.gz *
	@echo ""
	@echo "========================================"
	@echo "arXiv submission package created!"
	@echo "Upload: arxiv_submission.tar.gz"
	@echo "========================================"
