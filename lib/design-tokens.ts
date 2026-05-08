/**
 * lib/design-tokens.ts — Locked editorial palette.
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
  /** Dominant background — warm off-white paper. */
  cream:   "#fbf6ec",
  /** Primary text and ruled lines — deep navy. */
  navy:    "#0c1e3e",
  /** Electric cobalt — the single accent color. Sparingly. */
  cobalt:  "#2754d8",
  /** Secondary text, captions, muted UI — cool slate. */
  slate:   "#6b7a99",
  /** Fills, grid lines, soft card backgrounds — near-bg tint. */
  mist:    "#c9d2e3",
  /** Warm rust accent — one highlight per composition, maximum. */
  rust:    "#d4884a",
  /** Earthy red — error states, warning thresholds. */
  oxblood: "#a1322f",
} as const

export type PaletteKey = keyof typeof PALETTE

/**
 * CSS variable equivalents for use in inline styles.
 * Use Tailwind's `text-cobalt` / `bg-cream` classes wherever possible;
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
