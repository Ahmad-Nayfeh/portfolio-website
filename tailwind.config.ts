// tailwind.config.ts — Engineering Editorial design system (May 2026).
import type { Config } from "tailwindcss"

const config: Config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./content/**/*.{md,mdx}",
  ],
  theme: {
    extend: {
      // Three font families bound to the CSS variables loaded by next/font in
      // app/layout.tsx. Inter for UI, Newsreader for editorial display, and
      // JetBrains Mono for code.
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ["var(--font-serif)", "Georgia", "Times New Roman", "serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
        display: ["var(--font-serif)", "Georgia", "serif"],
      },
      // Editorial-scale type ramp. Display sizes get tighter tracking and
      // a leading suited to large serif headlines.
      fontSize: {
        "display-2xl": ["clamp(3.5rem, 8vw, 6rem)", { lineHeight: "1.02", letterSpacing: "-0.022em" }],
        "display-xl":  ["clamp(2.75rem, 6vw, 4.5rem)", { lineHeight: "1.05", letterSpacing: "-0.02em" }],
        "display-lg":  ["clamp(2rem, 4.5vw, 3.25rem)", { lineHeight: "1.08", letterSpacing: "-0.018em" }],
        "display-md":  ["clamp(1.625rem, 3vw, 2.25rem)", { lineHeight: "1.15", letterSpacing: "-0.014em" }],
      },
      letterSpacing: {
        tightest: "-0.025em",
        editorial: "-0.015em",
        wider: "0.04em",
        kicker: "0.18em",
      },
      colors: {
        // ── Locked editorial palette ─────────────────────────────────────
        // Exact hexes from the visual identity shared by blog cover images,
        // inline figures (plot_style.py), and the website. Use these named
        // tokens in new components; the semantic tokens below map to them.
        cream:   "#fbf6ec",   // dominant background — warm off-white
        navy:    "#0c1e3e",   // primary text + lines — deep navy
        cobalt:  "#2754d8",   // electric cobalt — the single accent color
        slate:   "#6b7a99",   // secondary text + muted elements
        mist:    "#c9d2e3",   // fills, grid lines, soft backgrounds
        rust:    "#d4884a",   // warm accent — use sparingly (one per page)
        oxblood: "#a1322f",   // earthy red — error states, thresholds
        // ── Semantic tokens (shadcn-compatible, HSL CSS variables) ───────
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        chart: {
          "1": "hsl(var(--chart-1))",
          "2": "hsl(var(--chart-2))",
          "3": "hsl(var(--chart-3))",
          "4": "hsl(var(--chart-4))",
          "5": "hsl(var(--chart-5))",
        },
        sidebar: {
          DEFAULT: "hsl(var(--sidebar-background))",
          foreground: "hsl(var(--sidebar-foreground))",
          primary: "hsl(var(--sidebar-primary))",
          "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
          accent: "hsl(var(--sidebar-accent))",
          "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
          border: "hsl(var(--sidebar-border))",
          ring: "hsl(var(--sidebar-ring))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      // Editorial-grade prose tweaks for the @tailwindcss/typography plugin.
      // Body uses Inter; first-level headings switch to Newsreader.
      typography: ({ theme }: { theme: (k: string) => string }) => ({
        DEFAULT: {
          css: {
            "--tw-prose-body": "hsl(var(--foreground))",
            "--tw-prose-headings": "hsl(var(--foreground))",
            "--tw-prose-links": "hsl(var(--accent))",
            "--tw-prose-bold": "hsl(var(--foreground))",
            "--tw-prose-quotes": "hsl(var(--foreground))",
            "--tw-prose-quote-borders": "hsl(var(--accent))",
            "--tw-prose-code": "hsl(var(--foreground))",
            "--tw-prose-hr": "hsl(var(--border))",
            "--tw-prose-bullets": "hsl(var(--muted-foreground))",
            "--tw-prose-counters": "hsl(var(--muted-foreground))",
            color: "hsl(var(--foreground))",
            fontSize: "1.0625rem",
            lineHeight: "1.75",
            "h1, h2, h3, h4": {
              fontFamily: "var(--font-serif), Georgia, serif",
              letterSpacing: "-0.015em",
              fontWeight: "600",
            },
            h1: { fontSize: "2.5rem", lineHeight: "1.1" },
            h2: {
              fontSize: "1.875rem",
              lineHeight: "1.2",
              marginTop: "2.5em",
              marginBottom: "0.6em",
              paddingBottom: "0.35em",
              borderBottom: "1px solid hsl(var(--border))",
            },
            h3: { fontSize: "1.375rem", lineHeight: "1.3", marginTop: "1.8em" },
            "a": { textDecoration: "none", borderBottom: "1px solid hsl(var(--accent) / 0.45)" },
            "a:hover": { borderBottomColor: "hsl(var(--accent))" },
            blockquote: {
              fontStyle: "italic",
              fontFamily: "var(--font-serif), Georgia, serif",
              borderLeftWidth: "3px",
              fontWeight: "400",
              color: "hsl(var(--foreground))",
            },
            "code::before": { content: "none" },
            "code::after": { content: "none" },
          },
        },
        invert: {
          css: {
            "--tw-prose-body": "hsl(var(--foreground))",
            "--tw-prose-headings": "hsl(var(--foreground))",
            "--tw-prose-links": "hsl(var(--accent))",
            "--tw-prose-bold": "hsl(var(--foreground))",
            "--tw-prose-quotes": "hsl(var(--foreground))",
            "--tw-prose-quote-borders": "hsl(var(--accent))",
            "--tw-prose-code": "hsl(var(--foreground))",
            "--tw-prose-hr": "hsl(var(--border))",
            "--tw-prose-bullets": "hsl(var(--muted-foreground))",
            "--tw-prose-counters": "hsl(var(--muted-foreground))",
          },
        },
      }),
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height