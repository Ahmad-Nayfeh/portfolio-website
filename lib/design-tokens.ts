/**
 * lib/design-tokens.ts — Locked laboratory palette.
 *
 * Single source of truth for the visual identity shared across:
 *   • Website (Tailwind classes, inline styles)
 *   • Blog cover images (generate_cover.py STYLE_SUFFIX)
 *   • Inline figures (scripts/plot_style.py PALETTE)
 *   • DALL-E illustration briefs (streams/ai-papers.yaml)
 *
 * Rule: never deviate from these hexes on any surface. The brand identity
 * holds because one palette appears on covers, charts, and the site.
 */

export const PALETTE = {
  /** Dominant background — warm paper. */
  parchment: "#f5f0e8",
  /** Primary text and ruled lines — deep forest green. */
  forest:    "#1a2e1a",
  /** Warm amber — the single accent color. Sparingly. */
  amber:     "#d4942a",
  /** Secondary text, captions, muted UI — cool sage. */
  sage:      "#6b8a6b",
  /** Fills, grid lines, soft card backgrounds — near-bg tint. */
  cream:     "#d8d5c8",
  /** Warm copper accent — one highlight per composition, maximum. */
  copper:    "#c87a4f",
  /** Earthy red — error states, warning thresholds. */
  rust:      "#a1322f",
} as const

export type PaletteKey = keyof typeof PALETTE

/**
 * CSS variable equivalents for use in inline styles.
 * Use Tailwind's `text-amber` / `bg-parchment` classes wherever possible;
 * reach for these only in dynamic inline styles.
 */
export const CSS_VARS = {
  background:  "hsl(var(--background))",
  foreground:  "hsl(var(--foreground))",
  accent:      "hsl(var(--accent))",
  muted:       "hsl(var(--muted))",
  mutedFg:     "hsl(var(--muted-foreground))",
  border:      "hsl(var(--border))",
  card:        "hsl(var(--card))",
} as const
