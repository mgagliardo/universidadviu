
TEX = pdflatex
TEXFILE = report.tex

all: pdf

pdf: $(TEXFILE)
	$(TEX) $<
	$(TEX) $<

clean:
	rm -f \
	$(TEXFILE:.tex=.pdf) \
	$(TEXFILE:.tex=.aux) \
	$(TEXFILE:.tex=.log) \
	$(TEXFILE:.tex=.toc) \
	texput.log
