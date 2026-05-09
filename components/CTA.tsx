import Link from "next/link"
import { ArrowUpRight, FileDown } from "lucide-react"

export default function CTA() {
  return (
    <section className="py-20 md:py-28">
      <div className="py-16 md:py-24 rounded-lg glass-card-strong">
        <div className="mx-auto max-w-3xl text-center">
          <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            — Open to work —
          </span>

          <h2 className="mt-6 text-display-lg font-bold tracking-tight text-balance">
            Looking for an engineer who can ship the whole stack of an
            <em style={{ color: "hsl(var(--section-accent))" }}> intelligent system</em>?
          </h2>

          <p className="mx-auto mt-6 max-w-xl text-base text-muted-foreground md:text-lg">
            I build end-to-end — from the data layer up to the model and the
            interface around it. If that sounds useful, the resume and a fuller
            biography are one click away.
          </p>

          <div className="mt-10 flex flex-col items-center justify-center gap-5 sm:flex-row">
            <Link
              href="/about"
              className="group inline-flex items-center gap-2 rounded-lg px-5 py-3 font-mono text-[11px] uppercase tracking-[0.2em] transition-all duration-300"
              style={{
                backgroundColor: "hsl(var(--section-accent))",
                color: "hsl(var(--primary-foreground))",
              }}
            >
              About the engineer
              <ArrowUpRight
                size={14}
                className="transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
              />
            </Link>
            <a
              href="/resume.pdf"
              target="_blank"
              rel="noreferrer"
              className="group inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-foreground"
            >
              <FileDown size={14} className="transition-colors" />
              <span
                className="border-b transition-colors"
                style={{
                  borderColor: "hsl(var(--section-accent) / 0.35)",
                }}
              >
                Download resume
              </span>
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
