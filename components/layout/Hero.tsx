import Link from "next/link"
import { ArrowUpRight } from "lucide-react"
import type { HomeContent } from "@/types"
import MarkdownRenderer from "@/components/MarkdownRenderer"
import HeroAmbient from "@/components/HeroAmbient"

interface HeroProps {
  content: HomeContent
}

export default function Hero({ content }: HeroProps) {
  return (
    <section className="relative pt-12 pb-20 md:pt-20 md:pb-28">
      <HeroAmbient />

      <div className="grid grid-cols-12 gap-x-6">
        <div className="col-span-12 lg:col-span-10 xl:col-span-9">
          {/* Kicker */}
          <div className="mb-8 flex items-center gap-3 animate-fade-up">
            <span className="h-px w-8" style={{ backgroundColor: "hsl(var(--section-accent))" }} />
            <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
              From the workbench · Saudi Arabia
            </span>
          </div>

          {/* Display title — gradient text */}
          <h1
            className="gradient-text text-display-xl text-balance animate-fade-up [animation-delay:60ms]"
            style={{ animationFillMode: "both" }}
          >
            {content.title}
          </h1>

          {/* Subtitle */}
          {content.subtitle && (
            <p
              className="mt-6 max-w-2xl text-xl italic text-muted-foreground md:text-2xl animate-fade-up [animation-delay:120ms]"
              style={{ animationFillMode: "both" }}
            >
              {content.subtitle}
            </p>
          )}

          {/* Body description */}
          <div
            className="prose prose-lg mt-8 max-w-2xl animate-fade-up [animation-delay:180ms]"
            style={{ animationFillMode: "both" }}
          >
            <MarkdownRenderer content={content.description} />
          </div>

          {/* CTAs */}
          <div
            className="mt-10 flex flex-col items-start gap-5 sm:flex-row sm:items-center animate-fade-up [animation-delay:240ms]"
            style={{ animationFillMode: "both" }}
          >
            <Link
              href="/projects"
              className="group inline-flex items-center gap-2 rounded-lg px-5 py-3 font-mono text-[11px] uppercase tracking-[0.2em] transition-all duration-300"
              style={{
                backgroundColor: "hsl(var(--section-accent))",
                color: "hsl(var(--primary-foreground))",
              }}
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
              <span
                className="border-b transition-colors"
                style={{
                  borderColor: "hsl(var(--section-accent) / 0.35)",
                }}
              >
                About the engineer
              </span>
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}
