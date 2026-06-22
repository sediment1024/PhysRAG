# README media

- `main.pdf`: editable/source overview figure supplied for the release.
- `physrag_overview.png`: GitHub-renderable overview exported from `main.pdf`.
- `demos/`: three compressed animated WebP comparison panels and their metadata.
- `dingliang.png`: quantitative comparison table from the paper.

The original MP4 ZIP is intentionally stored outside the GitHub package under
`../../demo_sources/` in the local staging tree. Upload it as a release asset or to
Hugging Face rather than committing it alongside the compressed previews.

Before publication, reconcile one terminology mismatch in the supplied figure:
the artwork labels the adapter activation as `GLU`, while the released
`physical_adapter.py` currently implements `Linear-GELU-Linear`.
