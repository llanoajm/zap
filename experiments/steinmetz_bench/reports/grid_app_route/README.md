# grid-app route bundle (human mount step)

This directory is a ready-to-mount bundle for the Steinmetz whitepaper. It is
produced inside the zap repo by `reports/build_whitepaper.py`; mounting it into
grid-app is a human cross-repo step (the zap loop's verify does not cover the
grid-app TypeScript).

Contents:
- `STEINMETZ_WHITEPAPER.md` — a verbatim copy of the generated whitepaper.
- `page.tsx` — a minimal Next.js server-component scaffold that renders the copy.

To mount (by hand):
1. Copy this directory to a gated route, e.g. grid-app `app/app/whitepaper/`.
2. Swap the `<pre>` for grid-app's markdown renderer if it has one.
3. Verify with grid-app's own `tsc --noEmit && npm run test:unit`.
