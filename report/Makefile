FIGS := pipeline owner_nn hmm results

Build_Up_Viz.pdf: Build_Up_Viz.tex $(FIGS:%=figures/%.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error Build_Up_Viz.tex
	latexmk -c

figures/%.pdf: figures/%.svg
	rsvg-convert -f pdf $< -o $@

$(FIGS:%=figures/%.svg) &: figures/make_figures.mjs
	cd figures && npm ci --no-bin-links --no-audit --no-fund && node make_figures.mjs
