import Link from "next/link"
import { ArrowUpRight } from "lucide-react"
import type { HomeContent } from "@/types"
import MarkdownRenderer from "@/components/MarkdownRenderer"

interface HeroProps {
  content: HomeContent
}

/**
 * Editorial hero.
 *
 * The visual structure is borrowed from the opening spread of a long-form
 * magazine: a tiny mono kicker (the issue/topic), a large display-serif
 * title that owns the page, then a smaller sans body for the description.
 * CTAs are restrained — a single solid primary, and a quieter ghost link.
 */
export default function Hero({ content }: HeroProps) {
  return (
    <section className="relative pt-12 pb-20 md:pt-20 md:pb-28">
      {/* Faint left rule — adds a magazine column edge without dominating. */}
      <div
        aria-hidden
        className="pointer-events-none absolute left-0 top-12 hidden h-32 w-px bg-foreground/15 md:block"
      />

      <div className="grid grid-cols-12 gap-x-6">
        <div className="col-span-12 lg:col-span-10 xl:col-span-9">
          {/* Kicker */}
          <div className="mb-8 flex items-center gap-3 animate-fade-up">
            <span className="h-px w-8 bg-accent" />
            <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
              From the workbench · Saudi Arabia
            </span>
          </div>

          {/* Display title — split with an italic emphasis to give it bite.
              We render whatever comes from content.md but layer typographic
              richness on top with the font-display class. */}
          <h1
            className="font-display text-display-xl text-foreground text-balance animate-fade-up [animation-delay:60ms]"
            style={{ animationFillMode: "both" }}
          >
            {content.title}
          </h1>

          {/* Subtitle */}
          {content.subtitle && (
            <p
              className="mt-6 max-w-2xl font-display text-xl italic text-muted-foreground md:text-2xl animate-fade-up [animation-delay:120ms]"
              style={{ animationFillMode: "both" }}
            >
              {content.subtitle}
            </p>
          )}

          {/* Body description in editorial prose. */}
          <div
            className="prose prose-lg dark:prose-invert mt-8 max-w-2xl animate-fade-up [animation-delay:180ms]"
            style={{ animationFillMode: "both" }}
          >
            <MarkdownRenderer content={content.description} />
          </div>

          {/* CTAs — a single solid action, then a quiet underlined link. */}
          <div
            className="mt-10 flex flex-col items-start gap-5 sm:flex-row sm:items-center animate-fade-up [animation-delay:240ms]"
            style={{ animationFillMode: "both" }}
          >
            <Link
              href="/projects"
              className="group inline-flex items-center gap-2 bg-foreground px-5 py-3 font-mono text-[11px] uppercase tracking-[0.2em] text-background transition-colors hover:bg-accent"
            >
              See selected work
              <ArrowUpRight
                size={14}
                className="transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
              />
            </Link>
            <Link
              href="/about"
              className="group inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-foreground"
            >
              <span className="border-b border-foreground/30 transition-colors group-hover:border-accent">
                About the engineer
              </span>
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}