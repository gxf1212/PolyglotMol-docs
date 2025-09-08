

To build the docs, run: `make html`

- try to write .md if possible for usage and rst for API (if no complex notes). 
- one canvas per file 
- follows sphinx and myst as markdown parser
- you can use sphinx-design to beautify (but not full of them), e.g. use Grids, Cards, Dropdowns, Tabs, Badges, Buttons & Icons. cards are for display of various submodules. tabs might be used for switching modes
- and also myst Admonitions
- no try except in ``` code blocks since user will test themselves
- similarly, no if emb is not None:print(f"CLS Rep Shape for '{smiles_batch[i]}': {emb.shape}"). just directly print...
- try to list all featurizers (parameters) in this submodule in a table. and users can easily copy and paste keys into the example code to use
- Minimize empty lines to improve readability without sacrificing clarity.
- abundant comments, like possible output format
- clear and simple for users to understand. the first principle is user-friendly. users read this and quickly write their own application
- use $ around equations
- keep the title concise since they appear in the doctree! just the keyword.
- put dependency installation after introduction (if any; no need for required dependencies like rdkit and deepchem). a single bash code block is enough
- no unrelated usages, just all possible fundamental usage about this file
- add links to original software packages (related doc/repo) for reference in the end
- do not use markers like {py:func} in inside a table
- but always use sth like this to refer to api: {doc}`/api/representations/fingerprints/deepchem`
- always add this to accept toctree (order of subpages are optional)
```{toctree}
:maxdepth: 1
:hidden:
```

for submodule index pages, should use this to list individual pages
```{toctree}
:maxdepth: 1
:hidden:

rdkit
cdk
deepchem
datamol
```
