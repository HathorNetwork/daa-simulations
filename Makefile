sim_files := $(wildcard *.bin)
plot_files := $(patsubst %.bin,%.png,$(sim_files))
cli := target/release/hathor-daa-sim
open := open
srcs = $(wildcard fast/src/*.rs) $(wildcard Cargo.*)

all:
	@echo targets:
	@echo - plots

cli: $(cli)

$(cli): $(srcs)
	cargo build --release

plots: $(plot_files)

%.png: %.bin $(cli)
	$(cli) plot --load-sim $< --save-plot $@ && $(open) $@
