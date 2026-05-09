import Link from "next/link"
import type { Route } from "next"
import { ArrowUpRight } from "lucide-react"

export interface SectionMastheadProps {
  kicker: string
  title: string
  description?: string
  link?: { href: Route | string; label: string }
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
      {/* Kicker row */}
      <div className="mb-5 flex items-center gap-3">
        <span aria-hidden className="h-px w-8" style={{ backgroundColor: "hsl(var(--section-accent))" }} />
        <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
          {kicker}
        </span>
      </div>

      <div className="grid grid-cols-12 items-end gap-x-6 gap-y-4 border-b pb-5"
        style={{ borderColor: "hsl(var(--border))" }}
      >
        <h2
          id={id}
          className="text-display-lg text-foreground col-span-12 lg:col-span-7 text-balance font-bold tracking-tight"
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
              <span
                className="border-b transition-colors"
                style={{
                  borderColor: "hsl(var(--section-accent) / 0.35)",
                }}
              >
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
