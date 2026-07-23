Bernardo, Andrea — the modeling side is unblocked, and I'm sorry it took this long.

Andrea, buy the Raspberry Pi 5, 8GB, about $150 all in. Don't buy the Jetson and
don't touch the FPGA. The reason is that the control loop is limited by how fast
the pumps and tubing settle, which is five to thirty seconds, so the loop runs at
0.1 to 10 Hz no matter how fast the camera or the chip is. At that rate inference
is nothing — every model I benchmarked runs in well under a millisecond, and the
regime classifier tops out around 1,700 frames per second on the Pi's CPU alone.
The step that actually costs time is the classical CV measurement, and even that
is 40 to 150 milliseconds for 250 frames at a sane ROI, a fraction of a five
second interval. One camera never saturates the Pi's USB3; two do, which is the
only reason we'll move to a NUC for double emulsions later. The Jetson only earns
its price if we're forced onto a learned high-res detector because classical CV
can't segment the droplets — that's the single trigger, and it's written down. Run
`python bench/benchmark.py` on the Pi and on your Mac and you'll get the real
numbers on your own hardware; everything degrades gracefully if a backend's
missing.

Bernardo, the device table is in `data/device_matrix.csv`, eight flow-focusing
PDMS devices, every row already checked against our fab limits — 8 µm minimum
feature, 15–200 µm depth, roof-collapse and mold-release guards. It should go
straight to the cleanroom. The geometries and fluids are paired on purpose so the
ratio W/ℓ* spans about four orders of magnitude, which is what we need to test the
normalization idea. `docs/device_spec.md` has the predicted diameter, frequency
and dripping-to-jetting boundary per device; those are model estimates, clearly
marked, and the chips' own data will sharpen them.

On the normalization itself: I audited the DAFD dataset and it can't test our
claim. It has no interfacial tension or density columns, so ℓ* isn't even
computable from it, and 55% of the rows are one fluid system. Identical models
show zero difference between the two normalizations on this data — not because
we're wrong, but because the data can't see it. The fix is our own device matrix
with measured fluid properties, which is exactly what Bernardo's table is for.

One thing for you, Antonio, when you're back: the W/ℓ* = Oh² line in the source
doc is inverted, it's Oh⁻², and the fluorocarbon ℓ* is ~5.3 µm not 14 — their
number implies an impossible density. Both matter before Bernardo emails the
authors. What I need from each of you: Andrea, the benchmark numbers off the real
Pi; Bernardo, the measured γ, η and ρ for three fluid systems once chips exist.
