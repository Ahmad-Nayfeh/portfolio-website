# PROJECT_MAP — Portfolio Website

> **Owner:** Ahmad Nayfeh
> **Last updated:** 2026-10-23

## [TECH_STACK]

| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | Next.js (App Router) | 15.2.6 |
| Language | TypeScript | ^5 |
| Styling | TailwindCSS | ^3.4.17 |
| UI Library | shadcn/ui (Radix primitives) | — |
| Fonts | IBM Plex Sans, Plex Serif (article body), Plex Mono, Plex Sans Arabic | via next/font |
| MDX Rendering | next-mdx-remote | ^6.0.0 |
| Remark | remark-gfm, remark-math | ^4.0.1, ^6.0.0 |
| Rehype | rehype-katex ^7.0.1, rehype-pretty-code ^0.14.3, rehype-raw ^7.0.0 | — |
| Syntax Highlight | Shiki | ^4.0.2 |
| Math | KaTeX | ^0.16.45 |
| Tailwind Plugin | @tailwindcss/typography | ^0.5.16 |
| Package Manager | pnpm | 9.15.9 |
| Node | — | 22.x |
| Python | Scripts pipeline | 3.12 |
| Python Key Deps | anthropic, openai, matplotlib, numpy, scipy, scikit-learn, seaborn | — |

## [SYSTEM_FLOW]

### Content Pipeline (site build)
```
content/blog/*.mdx  →  gray-matter (frontmatter parsing)  →  next-mdx-remote/rsc (MDX → HTML)
                          └─ lib/content.ts (getAllPosts, getPostBySlug, etc.)
                          └─ components/MdxRenderer.tsx (remark + rehype plugins)
```

### Blog Automation Pipeline (GitHub Actions)
```
┌─ schedule (cron) ─┐
│ workflow_dispatch  │→ main.py → load_streams (check cron) → discover_papers (HF daily)
└───────────────────┘                                       → select_papers (Claude)
                                                             → generate_post (stages)
                                                             → generate_cover (DALL-E)
                                                             → write_post (MDX)
                                                             → generate_inline_images (DALL-E)
                                                             → run_demo (matplotlib subprocess)
                                                             → validate_build (next build)
                                                             → Create PR → Auto-merge
```

### Page Routing
```
/                  → Home (Hero, FeaturedProjects, RecentPosts, Manifesto, CTA)
/about             → About (bio, skills, experience, "Reading" section)
/blog              → Blog list (filterable grid with URL-based tag filter)
/blog/[slug]       → Blog post (editorial layout, ToC sidebar, related posts)
/projects          → Projects list
/projects/[slug]   → Project detail
```

## [ARCHITECTURE]

### Component Hierarchy
```
RootLayout (layout.tsx) — ThemeProvider (light "paper" default, dark "blueprint")
├── Navbar (client; sheet-numbered nav, CV link, ThemeToggle)
├── <main>
│   ├── HomePage (app/page.tsx)
│   │   ├── home/Hero → home/SingleLineDiagram (animated SVG SLD; feeders link to sections)
│   │   ├── home/NowPanel        ← content/home/intro.md `now:`
│   │   ├── home/WorkSection     ← content/home/work.yaml
│   │   ├── projects/ProductFeature (products tier, featured)
│   │   ├── ProjectCard (research tier), BlogCard rows (latest posts)
│   │   └── Contact block
│   ├── AboutPage — story.md, TitleBlock spec, busbar timeline, skillGroups
│   ├── BlogPage — BlogGrid (client, ?tag= filter) → BlogCard rows
│   ├── BlogPostPage — MdxRenderer + TableOfContents (leading "# Title" stripped)
│   ├── ProjectsPage — tiers (Products / Research & ML / Lab) from config/site.ts
│   └── ProjectDetailPage — TitleBlock spec, gallery (phone frames), MarkdownRenderer
└── Footer (title block)
```

### Project tiers
`config/site.ts → projectTiers / tierOf(category)`. Frontmatter `category: "Products"` or
`"Research"` places a project; anything else (incl. pipeline pages with
`category: "AI Implementation"`) falls into **Lab**. Optional frontmatter:
`status`, `gallery: [paths]`, `highlights: [strings]`.

## [CURRENT_DESIGN_SYSTEM]

### Theme Name: "Drawing Sheet" (October 2026 redesign)
**Concept:** an electrical drawing set. Drafting paper with a fine grid, ink linework,
title blocks, sheet numbers, and one red-pen markup colour. Signature element: the
animated single-line diagram in the home hero.

### Tokens (app/globals.css)
| Token | Light ("paper") | Dark ("blueprint") |
|-------|-----------------|--------------------|
| Background | 45 22% 95% (#F5F3EE) | 213 38% 9% (#0E1622) |
| Foreground (ink) | 210 16% 11% | 210 25% 92% |
| Accent (red pen) | 14 82% 46% (#D5471A) | 20 92% 60% |
| Lamp (status) | 142 64% 36% | 142 70% 50% |
| Phase colours | brown / black / grey (details only) | lifted variants |
Radius 2px. Body background draws a 24px / 120px grid.

### Primitives
`.sheet` (card with corner registration marks), `.sheet-hover`, `.label` (mono caps),
`.chip`, `.btn` / `.btn-primary` / `.btn-ghost`, `.lamp`, `.redline` (underline on hover),
`.phone` (screenshot frame), `components/drawing/TitleBlock`, `components/drawing/SectionHeader`.

### Motion
CSS only (no Framer Motion). SLD: stroke-draw, switch blades close, then a dashed
"current" flows along conductors. All motion is disabled under prefers-reduced-motion.

## [TASK_PROGRESS]

- [x] Task 1A: Cover image aspect ratio fix — BlogCard changed from `aspect-[4/3]` to `aspect-[16/9]`
- [x] Task 1B: ToC navigation fix — installed `rehype-slug` v6, added to MdxRenderer, updated `extractHeadings` for duplicate IDs
- [x] Task 1C: Code blocks — verified rehype-pretty-code v0.14.3 compatible with shiki v4.0.2 (peer dep). All posts render code blocks.
- [x] Task 1D: Figure/plot quality — improved `plot_style.py` (200 DPI, larger font sizes, `subplots()` helper, `editorial_tick_params()`, better legend defaults) + `run_demo.py` auto-capture at 200 DPI
- [x] Task 2: Full redesign — **Analog Laboratory** theme implemented (amber/forest palette, Cormorant Garamond display font)
- [x] Task 3: Blog automation — uncommented `schedule: */30 * * * *` in `publish.yml`; stream cron at `17 0 * * 0` (Sunday 00:17 UTC)
- [x] Task 4: Python plotting pipeline (merged with Task 1D above)
- [x] Task 5 (Oct 2026): "Drawing Sheet" redesign of every page; projects curated into tiers (removed MNIST, Digital-Image-Processing, Edge-Vision; added Shadhra, Khatwa, Distill-SSL); removed the three auto-generated W19 paper-note posts; CV regenerated from `resume-src/resume.html`.

## [ORPHANS_AND_PENDING]

### Resolved This Session
1. ✅ `rehype-slug` installed and added to MdxRenderer (ToC navigation)
2. ✅ Shiki v4 / rehype-pretty-code confirmed compatible (peer dep relationship)
3. ✅ BlogCard aspect ratio changed to `aspect-[16/9]`
4. ✅ Workflow cron uncommented for weekly automation
5. ✅ Design tokens, plot_style, DALL-E style suffixes updated to Analog Laboratory palette
6. ✅ `plot_style.py` DPI increased to 200, added `subplots()` and `editorial_tick_params()` helpers

### Still Present
1. **`rehype-raw` installed but unused** — listed in deps but omitted from MdxRenderer plugins. Not harmful but unnecessary.
2. **`a-comprehensive-guide-to-ai-agents.mdx` has 0 code blocks** — hand-written guide post; intentional (no demo code needed for a survey article).
