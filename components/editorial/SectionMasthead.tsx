import Link from "next/link"
import type { Route } from "next"
import { ArrowUpRight } from "lucide-react"

/**
 * SectionMasthead — the standard "section opener" used on the home page and
 * on list pages. Renders a small mono kicker, a serif display title, an
 * optional inline description, and an optional "view all" link, all sitting
 * above a thin rule that establishes the editorial column.
 *
 * Keeping this in one place prevents the homepage and list pages from
 * drifting apart visually.
 */
export interface SectionMastheadProps {
  /** Tiny uppercase label sitting above the title. */
  kicker: string
  /** Display-serif title. */
  title: string
  /** Optional one-line description rendered in the right column on desktop. */
  description?: string
  /**
   * Optional "view all" link. Pass a route + label and the chevron is added
   * automatically. Omit to render no link at all.
   */
  link?: { href: Route | string; label: string }
  /** Optional id for jump links / aria-labelledby. */
  id?: string
}

export default function SectionMasthead({
  kicker,
  title,
  description,
  link,
  id,
}: SectionMastheadProps) {
  return (
    <header className="mb-10">
      {/* Kicker row: short rule + small caps label. */}
      <div className="mb-5 flex items-center gap-3">
        <span aria-hidden className="h-px w-8 bg-accent" />
        <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
          {kicker}
        </span>
      </div>

      <div className="grid grid-cols-12 items-end gap-x-6 gap-y-4 border-b border-border pb-5">
        <h2
          id={id}
          className="font-display text-display-lg text-foreground col-span-12 lg:col-span-7 text-balance"
        >
          {title}
        </h2>

        {description && (
          <p className="col-span-12 max-w-md text-base text-muted-foreground lg:col-span-4 lg:col-start-8">
            {description}
          </p>
        )}

        {link && (
          <div
            className={`col-span-12 ${
              description ? "lg:col-span-1 lg:col-start-12" : "lg:col-span-5 lg:col-start-8"
            } flex lg:justify-end`}
          >
            <Link
              href={link.href as Route}
              className="group inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground transition-colors hover:text-foreground"
            >
              <span className="border-b border-border transition-colors group-hover:border-accent">
                {link.label}
              </span>
              <ArrowUpRight
                size={12}
                className="transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
              />
            </Link>
          </div>
        )}
      </div>
    </header>
  )
}
