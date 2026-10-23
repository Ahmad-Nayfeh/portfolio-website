import type { Route } from "next"
import Link from "next/link"
import { ArrowRight, FileDown } from "lucide-react"
import SingleLineDiagram, { type Feeder } from "@/components/home/SingleLineDiagram"
import type { HomeContent } from "@/types"
import { siteConfig } from "@/config/site"

const feeders: Feeder[] = [
  { tag: "F1", title: "AT WORK", lines: ["Switchgear design", "& automation"], href: "#work" },
  { tag: "F2", title: "PRODUCTS", lines: ["Shadhra, Khatwa", "and more"], href: "#products" },
  { tag: "F3", title: "RESEARCH", lines: ["Deep learning", "& signals"], href: "#research" },
  { tag: "F4", title: "NOTEBOOK", lines: ["Essays and", "notes"], href: "/blog" },
]

export default function Hero({ content }: { content: HomeContent }) {
  return (
    <section className="relative mx-auto w-full max-w-[1240px] px-5 pb-16 pt-10 md:px-8 md:pb-24 md:pt-16">
      <div className="grid grid-cols-1 items-center gap-x-10 gap-y-12 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <div className="fade-up flex items-center gap-3" style={{ animationDelay: "40ms" }}>
            <span className="lamp" aria-hidden />
            <span className="label">{siteConfig.location}</span>
          </div>

          <h1
            className="fade-up mt-6 font-display text-[clamp(2.75rem,6.5vw,4.75rem)] leading-[0.98]"
            style={{ animationDelay: "100ms" }}
          >
            {content.title}
          </h1>

          <p
            className="fade-up mt-5 font-mono text-[13px] font-medium uppercase tracking-[0.16em] text-accent"
            style={{ animationDelay: "160ms" }}
          >
            {content.subtitle}
          </p>

          <p
            className="fade-up mt-6 max-w-[34rem] text-[1.125rem] leading-relaxed text-foreground/85 md:text-[1.2rem]"
            style={{ animationDelay: "220ms" }}
          >
            {content.description}
          </p>

          <div className="fade-up mt-9 flex flex-wrap gap-3" style={{ animationDelay: "280ms" }}>
            <Link href={"/projects" as Route} className="btn btn-primary">
              View projects <ArrowRight size={14} />
            </Link>
            <a href={siteConfig.resume} target="_blank" rel="noopener noreferrer" className="btn btn-ghost">
              <FileDown size={14} /> Download CV
            </a>
          </div>
        </div>

        {/* Diagram: desktop / tablet */}
        <div className="hidden md:block lg:col-span-7">
          <div className="sheet p-4 md:p-6">
            <SingleLineDiagram feeders={feeders} source={["SOURCE", "Electrical Engineering, KFUPM"]} />
          </div>
        </div>

        {/* Feeder list: phones */}
        <nav aria-label="Sections" className="md:hidden">
          <div className="label mb-3">Outgoing feeders</div>
          <ul className="border-[1.5px] border-foreground bg-card">
            {feeders.map((f) => (
              <li key={f.tag} className="border-b border-border last:border-b-0">
                {f.href.startsWith("#") ? (
                  <a href={f.href} className="flex items-center gap-4 px-4 py-3.5">
                    <FeederGlyph />
                    <span className="font-mono text-[11px] text-muted-foreground">{f.tag}</span>
                    <span className="flex-1">
                      <span className="block text-sm font-semibold tracking-wide">{f.title}</span>
                      <span className="block text-[13px] text-muted-foreground">{f.lines.join(" ")}</span>
                    </span>
                    <ArrowRight size={15} className="text-muted-foreground" />
                  </a>
                ) : (
                  <Link href={f.href as Route} className="flex items-center gap-4 px-4 py-3.5">
                    <FeederGlyph />
                    <span className="font-mono text-[11px] text-muted-foreground">{f.tag}</span>
                    <span className="flex-1">
                      <span className="block text-sm font-semibold tracking-wide">{f.title}</span>
                      <span className="block text-[13px] text-muted-foreground">{f.lines.join(" ")}</span>
                    </span>
                    <ArrowRight size={15} className="text-muted-foreground" />
                  </Link>
                )}
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </section>
  )
}

/** Tiny load-break-switch glyph for the phone feeder list. */
function FeederGlyph() {
  return (
    <svg width="14" height="30" viewBox="0 0 14 30" aria-hidden className="shrink-0">
      <path d="M7 0 V7 M2 7 H12 M7 21 V30" stroke="currentColor" strokeWidth="1.4" fill="none" />
      <line x1="7" y1="21" x2="7" y2="8" stroke="hsl(var(--accent))" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="7" cy="21" r="1.8" fill="currentColor" />
    </svg>
  )
}
