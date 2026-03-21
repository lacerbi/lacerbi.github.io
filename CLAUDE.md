# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Luigi Acerbi's personal academic website built on the [al-folio](https://github.com/alshedivat/al-folio) Jekyll theme. Hosted at https://lacerbi.github.io via GitHub Pages.

## Build & Serve Commands

```bash
# Local development (requires Ruby + Bundler)
bundle exec jekyll serve

# With Docker
docker compose up

# Production build
JEKYLL_ENV=production bundle exec jekyll build

# Install dependencies
bundle install

# Format code (Prettier with Liquid plugin)
npx prettier --write .
```

**Deployment:** Pushing to `master` automatically triggers CI (`deploy.yml`): builds with `JEKYLL_ENV=production`, purges unused CSS, and deploys to GitHub Pages. The `bin/deploy` script exists as a manual alternative but is generally unnecessary.

## Architecture

### Jekyll Site Structure

- **`_config.yml`** — Central configuration (644 lines). Controls site metadata, Jekyll Scholar settings, plugin config, feature flags, and third-party library CDN versions.
- **`_pages/`** — Main site pages (about, publications, projects, cv, blog, teaching, contact). Written in Markdown with YAML front matter.
- **`_posts/`** — Blog posts. Permalink pattern: `/blog/:year/:title/`.
- **`_bibliography/papers.bib`** — BibTeX file processed by jekyll-scholar for the publications page. Entries use custom keywords (`abbr`, `preview`, `pdf`, `code`, `selected`, etc.) that are filtered from output but drive the publication card UI.
- **`_news/`** — Short announcement items displayed on the homepage.
- **`_projects/`** — Project showcase pages with category support.
- **`_data/`** — YAML data files: `cv.yml`, `coauthors.yml`, `repositories.yml`, `venues.yml`.
- **`assets/json/resume.json`** — JSON Resume standard format, loaded via `jekyll-get-json`.

### Templating

- **`_layouts/`** — Page layouts (about, post, page, distill, cv, bib, etc.). Written in Liquid.
- **`_includes/`** — Reusable components (header, footer, nav, citations, social links, scripts).
- **`_sass/`** — SCSS modules. Key files: `_base.scss`, `_layout.scss`, `_themes.scss`, `_variables.scss`.

### Custom Plugins

Ruby plugins in `_plugins/` extend Jekyll: cache busting, custom `<details>` tags, BibTeX field hiding, Google Scholar citation counts, accent removal, and third-party library downloading.

### Key Feature Flags in `_config.yml`

These `enable_*` booleans control site features: `enable_math` (MathJax), `enable_darkmode`, `enable_masonry` (project card layout), `enable_medium_zoom` (image lightbox), `enable_progressbar`, `enable_publication_badges` (Altmetric/Dimensions/Scholar badges on publications).

### Third-Party Libraries

Library versions and SRI hashes are configured in `_config.yml` under `third_party_libraries`. Setting `download: true` downloads them locally instead of using CDN.

## CI/CD

GitHub Actions workflows handle: deployment (`deploy.yml`), Prettier formatting checks (`prettier.yml`), broken link validation (`broken-links.yml`), Lighthouse performance audits, and accessibility testing (`axe.yml`).

## Formatting

Prettier is mandatory. Config in `.prettierrc` uses the `@shopify/prettier-plugin-liquid` plugin with 150 char print width.
