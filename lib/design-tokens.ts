/**
 * lib/design-tokens.ts — Nocturne palette (May 2026 redesign).
 *
 * Single source of truth for the visual identity shared across:
 *   • Website (Tailwind classes, inline styles)
 *   • Blog cover images (generate_cover.py STYLE_SUFFIX)
 *   • Inline figures (scripts/plot_style.py PALETTE)
 *   • DALL-E illustration briefs (streams/ai-papers.yaml)
 *
 * Deep dark canvas with electric teal, amber, and magenta accents.
 */

export const PALETTE = {
  /** Deep near-black background with a blue hint. */
  bg:       "#080812",
  /** Card / surface background. */
  surface:  "#12121e",
  /** Subtle border colour. */
  border:   "#1e1e30",
  /** Primary text — warm white. */
  text:     "#e8e8ed",
  /** Secondary text — muted cool grey. */
  muted:    "#8888a0",
  /** Electric teal — primary accent (technical, modern). */
  teal:     "#00d4aa",
  /** Warm amber — secondary accent (energy, highlights). */
  amber:    "#ffba08",
  /** Hot magenta — tertiary accent (creativity, surprise). */
  magenta:  "#ff3cac",
  /** Error / destructive red. */
  red:      "#ff3366",
} as const

export type PaletteKey = keyof typeof PALETTE

export const CSS_VARS = {
  background:  "hsl(var(--background))",
  foreground:  "hsl(var(--foreground))",
  accent:      "hsl(var(--accent))",
  muted:       "hsl(var(--muted))",
  mutedFg:     "hsl(var(--muted-foreground))",
  border:      "hsl(var(--border))",
  card:        "hsl(var(--card))",
} as const
