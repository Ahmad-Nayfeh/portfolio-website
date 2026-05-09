# PROJECT_MAP — Portfolio Website

> **Owner:** Ahmad Nayfeh
> **Last updated:** 2026-05-09

## [TECH_STACK]

| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | Next.js (App Router) | 15.2.6 |
| Language | TypeScript | ^5 |
| Styling | TailwindCSS | ^3.4.17 |
| UI Library | shadcn/ui (Radix primitives) | — |
| Fonts | Inter (sans), Newsreader (serif), JetBrains Mono (mono) | via next/font |
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
RootLayout (layout.tsx)
├── Navbar (sticky, client component, scroll-aware)
├── <main>
│   ├── HomePage (page.tsx)
│   │   ├── Hero (server)
│   │   ├── FeaturedProjects (server)
│   │   ├── RecentPosts (server)
│   │   ├── Manifesto (server)
│   │   └── CTA (server)
│   ├── AboutPage
│   │   └── SectionMasthead, FadeIn, etc.
│   ├── BlogPage
│   │   └── BlogGrid (client, useSearchParams for tag filter)
│   ├── BlogPostPage
│   │   ├── Article header (cover image, title, meta)
│   │   ├── MdxRenderer (server, MDX → HTML)
│   │   └── TableOfContents (client, IntersectionObserver)
│   ├── ProjectsPage
│   └── ProjectDetailPage
└── Footer
```

### Server vs Client Components
- **Server:** Hero, FeaturedProjects, RecentPosts, Manifesto, CTA, Footer, MdxRenderer, SectionMasthead, BlogPage, BlogPostPage
- **Client:** Navbar, TableOfContents, BlogGrid, FadeIn, ThemeToggle

### Design Token Flow
```
lib/design-tokens.ts (named hex palette) ─→ Python scripts (plot_style.py, generate_cover.py)
                                ─→ streams/ai-papers.yaml (DALL-E briefs)

app/globals.css (CSS custom properties, HSL vars) ─→ tailwind.config.ts (semantic tokens)
                                                   → Components (via @apply, utility classes)
```

## [CURRENT_DESIGN_SYSTEM]

### Theme Name: "Analog Laboratory" (May 2026 redesign)
**Vibe:** Scientific instrument, oscilloscope glow, laboratory notebook. Amber and forest green.

### Color Palette
| Token | Light (HSL) | Dark (HSL) | Hex Ref |
|-------|-------------|-------------|---------|
| Background | parchment 45 25% 93% | forest black 120 42% 8% | #f5f0e8 (lt) / #0d1f0d (dk) |
| Foreground | forest 120 28% 14% | warm cream 40 22% 88% | #1a2e1a (lt) |
| Accent | amber 38 69% 50% | amber glow 36 80% 63% | #d4942a (lt) / #ffb347 (dk) |
| Muted | sage 100 12% 48% | sage 100 10% 54% | #6b8a6b |
| Border | cream 45 17% 82% | dark forest 120 16% 18% | #d8d5c8 (lt) |
| Destructive | rust 2 55% 41% | rust lifted 2 55% 50% | #a1322f |

### Typography
- **UI/Body:** Inter (sans-serif, variable font)
- **Display/Headings:** Cormorant Garamond (serif, 400-700 weight, italic) — replaces Newsreader
- **Code:** JetBrains Mono (monospace, 400-600 weight)
- **Scale:** display-2xl through display-md (clamped fluid sizes)
- **Prose body:** 1.0625rem, line-height 1.75

### Motion (unchanged)
- No Framer Motion. CSS transitions + IntersectionObserver only.
- Fade-up: 360ms cubic-bezier(0.22, 1, 0.36, 1), 8px translate
- Lift: 280ms, 2px hover translate
- Underline-grow: 280ms, scaleX 0→1
- prefers-reduced-motion: all durations 0.01ms

### Spacing (unchanged)
- Max content width: 1400px
- Page padding: px-6 (mobile) / px-10 (md) / px-16 (lg)
- Container: mx-auto with 12-column grid

## [TASK_PROGRESS]

- [x] Task 1A: Cover image aspect ratio fix — BlogCard changed from `aspect-[4/3]` to `aspect-[16/9]`
- [x] Task 1B: ToC navigation fix — installed `rehype-slug` v6, added to MdxRenderer, updated `extractHeadings` for duplicate IDs
- [x] Task 1C: Code blocks — verified rehype-pretty-code v0.14.3 compatible with shiki v4.0.2 (peer dep). All posts render code blocks.
- [x] Task 1D: Figure/plot quality — improved `plot_style.py` (200 DPI, larger font sizes, `subplots()` helper, `editorial_tick_params()`, better legend defaults) + `run_demo.py` auto-capture at 200 DPI
- [x] Task 2: Full redesign — **Analog Laboratory** theme implemented (amber/forest palette, Cormorant Garamond display font)
- [x] Task 3: Blog automation — uncommented `schedule: */30 * * * *` in `publish.yml`; stream cron at `17 0 * * 0` (Sunday 00:17 UTC)
- [x] Task 4: Python plotting pipeline (merged with Task 1D above)

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
