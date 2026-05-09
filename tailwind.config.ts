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
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ["var(--font-serif)", "Georgia", "Times New Roman", "serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
        display: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      fontSize: {
        "display-2xl": ["clamp(3.5rem, 8vw, 6rem)", { lineHeight: "1.02", letterSpacing: "-0.025em" }],
        "display-xl":  ["clamp(2.75rem, 6vw, 4.5rem)", { lineHeight: "1.05", letterSpacing: "-0.022em" }],
        "display-lg":  ["clamp(2rem, 4.5vw, 3.25rem)", { lineHeight: "1.08", letterSpacing: "-0.02em" }],
        "display-md":  ["clamp(1.625rem, 3vw, 2.25rem)", { lineHeight: "1.15", letterSpacing: "-0.018em" }],
      },
      colors: {
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
              fontFamily: '"Inter", ui-sans-serif, system-ui, sans-serif',
              letterSpacing: "-0.02em",
              fontWeight: "700",
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
            "a": {
              textDecoration: "none",
              borderBottom: "1px solid hsl(var(--accent) / 0.35)",
            },
            "a:hover": { borderBottomColor: "hsl(var(--accent))" },
            blockquote: {
              fontStyle: "italic",
              borderLeftWidth: "3px",
              fontWeight: "400",
              color: "hsl(var(--foreground))",
              borderLeftColor: "hsl(var(--accent))",
            },
            "code::before": { content: "none" },
            "code::after": { content: "none" },
          },
        },
        invert: {
          css: {},
        },
      }),
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "fade-up": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "underline-grow": {
          from: { backgroundSize: "0% 1px" },
          to: { backgroundSize: "100% 1px" },
        },
        "pulse-dot": {
          "0%, 100%": { opacity: "0.6", transform: "scale(1)" },
          "50%": { opacity: "1", transform: "scale(1.3)" },
        },
        "glow-pulse": {
          "0%, 100%": { boxShadow: "0 0 8px var(--section-glow), 0 0 16px var(--section-glow)" },
          "50%": { boxShadow: "0 0 16px var(--section-glow), 0 0 32px var(--section-glow)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "fade-up": "fade-up 0.6s cubic-bezier(0.22, 1, 0.36, 1) both",
        "underline-grow": "underline-grow 0.35s ease-out forwards",
        "pulse-dot": "pulse-dot 2.5s ease-in-out infinite",
        "glow-pulse": "glow-pulse 2s ease-in-out infinite",
      },
    },
  },
  plugins: [
    require("tailwindcss-animate"),
    require("@tailwindcss/typography"),
  ],
}

export default config
