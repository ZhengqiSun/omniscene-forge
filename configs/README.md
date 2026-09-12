# Configuration

`interaction/` contains JSON recipes with validated argument names and path resolution. The `train` and `infer` examples belong to the interaction variant; they do not define a default model for the entire project.

For other routes, inspect the original arguments with `python multiview.py run ROUTE -- --help` and configure inputs according to the [project map](../docs/PROJECT_MAP.md). Machine-specific launchers, watchers, and experiment authorization configurations remain in the original backup.

`*.local.json` files are ignored by Git. Example training parameters are not validated recipes for reproducing historical quality results.
