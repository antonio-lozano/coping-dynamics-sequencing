# Publication release: DOIs, licensing, and journal statements

What to do, in order, to take this repository from its current state to a
citable archive that satisfies Nature Portfolio policy. Journal requirements
quoted here are from the Nature Neuroscience / Nature Portfolio pages listed at
the end.

## Two records, not one

Code and data are archived as **separate Zenodo records**, cross-linked to each
other:

| Record | Zenodo type | License | How it is created |
| --- | --- | --- | --- |
| Coping Dynamics Sequencing (code) | `software` | MIT | GitHub release, archived automatically |
| Coping dynamics dataset | `dataset` | CC BY 4.0 | Manual upload with a reserved DOI |

The split matters for three reasons: a Zenodo record carries exactly one
`upload_type`, so bundling forces one of the two to be mislabeled; only
`software` records are mirrored to Software Heritage; and the two need
different licenses, which this repository already declares separately in
`LICENSE` and `LICENSE-DATA`.

## Reserving a DOI before submission

Zenodo can reserve a DOI on a draft record so it can be quoted in a manuscript
before the record is public. Two constraints shape the order of operations:

- **The GitHub integration cannot pre-reserve.** Zenodo states plainly that a
  DOI cannot be reserved ahead of a GitHub release; a repository linked through
  the integration gets its DOI when the release is archived. Writing a `doi`
  field into `.zenodo.json` does not change this.
- **A reserved DOI is a version DOI.** The concept DOI, which always resolves to
  the newest version, is minted only when the record is first published, so it
  cannot be known in advance for a first deposit.
- **A deleted draft loses its reserved DOI permanently.** Do not delete the
  draft once its DOI has been quoted anywhere.

For the code this is not a problem: a GitHub release mints both DOIs within
minutes, so publish the release rather than reserving. For the dataset, which
is uploaded by hand, reserve first so the identifier can go into the manuscript
and into the code record's metadata.

## Order of operations

1. **Merge to main and tag.** Zenodo archives the default branch state of a
   release, so the reproducible layout must be on `main` first. The merge also
   puts the project page on its permanent trigger: `docs/web/index.html` deploys
   to <https://antonio-lozano.github.io/coping-dynamics-sequencing/> through
   `.github/workflows/pages.yml`, which needs *Settings → Pages → Source: GitHub
   Actions* enabled once. Remove the pre-merge branch from that workflow's
   `branches` list at the same time.
2. **Reserve the dataset DOI.** Log in to Zenodo, *New upload*, set
   *Resource type* to **Dataset**, answer **No** to "Do you already have a
   DOI?", and press **Get a DOI now!**. Fill in title, authors, description and
   set the license to **CC BY 4.0**. Save the draft; do not publish or delete
   it. Note the reserved DOI.
3. **Record the dataset DOI in the code metadata.** Add it to `.zenodo.json`
   under `related_identifiers` so the archived code points at the data:

   ```json
   {
     "identifier": "10.5281/zenodo.XXXXXXX",
     "relation": "isSourceOf",
     "resource_type": "dataset",
     "scheme": "doi"
   }
   ```

4. **Link the repository to Zenodo.** Zenodo profile menu → *GitHub* →
   *Sync now* → toggle the repository on. The repository must be public and must
   contain a license file; both already hold.
5. **Cut release `v1.0.0` on GitHub.** Zenodo archives it and mints the version
   DOI and the concept DOI. Read both from the record page.
6. **Write the DOIs into `CITATION.cff`.** Uncomment the `identifiers` block at
   the bottom of the file and fill in both values, bare (no `https://doi.org/`
   prefix). Add the DOI badge to `README.md`.
7. **Finish the dataset record.** Add the code DOI to the dataset draft as
   `isDerivedFrom`, attach the data files, and publish. Publish before the paper
   goes to press so the reserved DOI resolves.
8. **Re-release if the code changes during review.** A later release becomes a
   new version under the same concept DOI, so a concept DOI quoted in the
   manuscript stays valid.

### Metadata precedence

If both `.zenodo.json` and `CITATION.cff` are present, Zenodo reads **only**
`.zenodo.json` and ignores `CITATION.cff` entirely. Both files ship here
because GitHub's "Cite this repository" widget reads the CFF file; keep the two
in agreement by hand. Note also that the two use different case conventions for
license identifiers: `.zenodo.json` wants lowercase SPDX (`mit`, `cc-by-4.0`),
CFF wants canonical SPDX (`MIT`, `CC-BY-4.0`).

## What the journal requires

**Data availability statement — mandatory.** Every Nature Portfolio research
paper must carry one, as its own headed section at the end of the Methods,
before the Extended Data legends. It must make the conditions of access to the
minimum dataset transparent, give identifiers for public data, and state any
restrictions. Where figure source data are supplied, the statement must say so.

**Code availability statement — mandatory when custom code is central**, as a
separate headed section immediately after the data statement. It must state
whether and how the code can be accessed, any restrictions, and — a requirement
that is easy to miss — **the license by name**. Nature's code guidelines expect
the paper's version of the code to carry a DOI or other persistent identifier
rather than a bare repository URL.

**Repository choice.** Behavioral and pose data are not in Nature's
mandatory-deposition list, so no community repository is prescribed. Nature's
own text names Zenodo, figshare and Dryad as acceptable general repositories,
and names Zenodo alongside Code Ocean for code.

**Licensing.** Springer Nature recommends licenses that maximize reuse, naming
CC0 and CC BY, and warns that non-commercial or share-alike clauses can block
legitimate reuse. Restrictions must be disclosed to the editor at submission.
This repository therefore uses MIT for code and CC BY 4.0 for data.

**Source data.** Nature Neuroscience encourages source data for figures and
asks that statistics source data be provided in Excel, one file per figure,
with the linked figure named in the file. `report/raw_data.xlsx` already has
this shape: one sheet per figure panel-set, each named for its figure. The
per-panel plotted values are additionally available as CSVs in
`figure_source_data/`.

**Reporting summary — mandatory.** Nature Neuroscience is a life-sciences
journal, so the Reporting Summary must be completed in Adobe Reader and is
published with the paper. Its statistics section asks for exact sample sizes,
one- versus two-sidedness, covariates, corrections for multiple comparisons,
test statistics with effect sizes and exact p-values, and — the item that
matters most here — the level of analysis for hierarchical designs. The
litter-clustered models in `report/statistical_report.xlsx` supply all of this;
the unit of analysis is the animal, with standard errors clustered by litter.

**Animal ethics.** The manuscript must name the approving institutional or
licensing committee and describe animal sex, housing and husbandry. ARRIVE 2.0
reporting is recommended.

## Draft statements

Fill in the bracketed identifiers once minted.

**Data availability**

> The data supporting the findings of this study are available in the Zenodo
> repository under DOI [10.5281/zenodo.XXXXXXX], and are also bundled in the
> analysis repository archived under DOI [10.5281/zenodo.XXXXXXX]. The bundled
> data comprise the per-animal supervised freezing predictions, the per-frame
> keypoint-MoSeq syllable assignments, the time-binned syllable usage tables,
> and the behavioral flexibility scores used for resilience grouping — the
> minimum dataset required to reproduce every figure and statistic reported
> here. Data are released under a Creative Commons Attribution 4.0
> International license. Source data for Figures 2–7 are provided with the
> paper.

**Code availability**

> All analysis code is available at
> https://github.com/antonio-lozano/coping-dynamics-sequencing and archived at
> Zenodo under DOI [10.5281/zenodo.XXXXXXX], released under the MIT license.
> Every figure, statistical model output and report regenerates from the
> bundled data with a single command; artifact integrity is enforced by a
> SHA-256 manifest and verified by continuous integration on Windows and Linux.
> No access restrictions apply.

## Sources

- Reporting standards and availability of data, materials, code and protocols —
  https://www.nature.com/neuro/editorial-policies/reporting-standards
- Data availability statements and data citations policy —
  https://www.nature.com/documents/nr-data-availability-statements-data-citations.pdf
- Guidelines for authors submitting code and software —
  https://www.nature.com/documents/GuidelinesCodePublication.pdf
- Nature Neuroscience initial submission and formatting —
  https://www.nature.com/neuro/submission-guidelines/aip-and-formatting
- Copyright and data —
  https://support.springernature.com/en/support/solutions/articles/6000237622-copyright-and-data
- Reserve a DOI — https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/
- DOI reservation and the GitHub integration —
  https://support.zenodo.org/help/en-gb/24-github-integration/73-can-i-pre-reserved-a-doi-before-a-github-release
- DOI versioning — https://support.zenodo.org/help/en-gb/1-upload-deposit/97-what-is-doi-versioning
- `.zenodo.json` — https://help.zenodo.org/docs/github/describe-software/zenodo-json/
