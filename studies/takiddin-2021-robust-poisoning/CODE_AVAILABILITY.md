# Code-availability search

**Checked:** 2026-09-20

**Paper:** Robust Electricity Theft Detection Against Data Poisoning Attacks
in Smart Grids, DOI `10.1109/TSG.2020.3047864`.

**Outcome:** no matching public implementation located in the checked
sources. Publisher supplementary-material inspection remains incomplete
because IEEE presented a browser-verification barrier. No author was contacted,
and no code was downloaded or executed.

## Primary records inspected

| Source | Observation |
|---|---|
| Complete local published PDF, pp. 2675-2684, SHA-256 `03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff` | No repository or code-availability link was found. Text search for repository, source-code, supplementary-material, and URL references corroborated the complete visual reading. |
| [Author publication listing](https://web1.eng.famu.fsu.edu/~takiddin/publications.html) | Exact paper entry links [paper13.pdf](https://web1.eng.famu.fsu.edu/~takiddin/publications/paper13.pdf) and its DOI, with no code link in the entry. The HTML was retrieved directly after the web reader timed out. |
| [Author biography](https://web1.eng.famu.fsu.edu/~takiddin/takiddin.html) | Visible navigation links publications, teaching, Scholar, ORCID, and CV; no associated code repository was identified. |
| [Crossref DOI metadata](https://api.crossref.org/works/10.1109/TSG.2020.3047864) | Title/DOI match. Primary resource is IEEE document 9310227. The `relation` object is empty; the only `link` entry is a version-of-record PDF for similarity checking. Metadata absence does not exclude unregistered software. |
| [IEEE article](https://ieeexplore.ieee.org/document/9310227/) | Page returned a JavaScript/browser-verification challenge. Supplementary material could not be verified. |
| [Zenodo DOI search](https://zenodo.org/api/records?q=%2210.1109%2FTSG.2020.3047864%22&size=10) | API returned zero records. |

## Direct GitHub searches

Commands used the authenticated GitHub CLI or public search API. Search limits
and indexing apply; this is not an exhaustive search of every repository.

```bash
gh search code '"10.1109/TSG.2020.3047864"' --limit 50 --json repository,path,url
gh search code '"Robust Electricity Theft Detection Against Data Poisoning"' --limit 50 --json repository,path,url
gh search repos 'Takiddin' --limit 50 --json fullName,description,url
```

The exact DOI and title-phrase code searches each returned `[]`. The surname
repository search returned unrelated similarly named repositories, with no
verified paper implementation. The public repository search for
`"electricity theft" poisoning` returned `total_count: 0` and
`incomplete_results: false`.

An initial code search for only the digits `3047864` returned unrelated numeric
data files. It was discarded as nonspecific and replaced by the full DOI and
title searches above. Numeric matches were not treated as candidate software.

## Web-search queries

- `"Robust Electricity Theft Detection Against Data Poisoning Attacks in Smart Grids" code`
- `"10.1109/TSG.2020.3047864" github`
- `"Takiddin" github electricity poisoning`
- `"Robust Electricity Theft" "code" github zenodo`
- `"Abdulrahman Takiddin" GitHub code`
- `site:github.com "Robust Electricity Theft Detection"`
- `site:zenodo.org "Takiddin" poisoning`
- `site:codeocean.com "3047864"`

These located the paper, citations, author pages, and other research, but no
matching implementation. The Code Ocean check was a web-index search only,
not a direct exhaustive catalog inspection.

## Consequence for the work

Proceed with the source-derived reconstruction, with every necessary choice
recorded. If a repository is found later, verify its paper identity, author
association, release date, license, data requirements, and what experiments it
actually implements before executing it. Inspect any differences from the
published method explicitly.

Use the wording “no public implementation located in the sources checked on
20 September 2026.” Do not turn this result into a claim that the authors have
no code, never release code, or fabricated their results.
