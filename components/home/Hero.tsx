import type { Route } from "next"
import Link from "next/link"
import { ArrowRight, FileDown } from "lucide-react"
import TitleBlock, { type TitleBlockCell } from "@/components/drawing/TitleBlock"
import type { HomeContent } from "@/types"
import { siteConfig } from "@/config/site"

/** Who I am in one screen: name, plain-language role, one sentence, two buttons, and the key facts. */
export default function Hero({ content }: { content: HomeContent }) {
  const facts: TitleBlockCell[] = content.now.map((item) => ({ label: item.label, value: item.detail, span: 2 }))
  facts.push({ label: "Based in", value: siteConfig.location, span: 2 })

  return (
    <section className="relative mx-auto w-full max-w-[1240px] px-5 pb-16 pt-10 md:px-8 md:pb-24 md:pt-16">
      <div className="grid grid-cols-1 items-center gap-x-12 gap-y-10 lg:grid-cols-12">
        <div className="lg:col-span-7">
          <h1
            className="fade-up font-display text-[clamp(2.75rem,6.5vw,4.75rem)] leading-[0.98]"
            style={{ animationDelay: "60ms" }}
          >
            {content.title}
          </h1>

          <p
            className="fade-up mt-5 font-mono text-[13px] font-medium uppercase tracking-[0.16em] text-accent"
            style={{ animationDelay: "120ms" }}
          >
            {content.subtitle}
          </p>

          <p
            className="fade-up mt-6 max-w-[36rem] text-[1.125rem] leading-relaxed text-foreground/85 md:text-[1.2rem]"
            style={{ animationDelay: "180ms" }}
          >
            {content.description}
          </p>

          <div className="fade-up mt-9 flex flex-wrap gap-3" style={{ animationDelay: "240ms" }}>
            <Link href={"/projects" as Route} className="btn btn-primary">
              View projects <ArrowRight size={14} />
            </Link>
            <a href={siteConfig.resume} target="_blank" rel="noopener noreferrer" className="btn btn-ghost">
              <FileDown size={14} /> Download CV
            </a>
          </div>
        </div>

        <div className="fade-up lg:col-span-5" style={{ animationDelay: "300ms" }}>
          <div className="label mb-3">At a glance</div>
          <TitleBlock cols={2} cells={facts} />
        </div>
      </div>
    </section>
  )
}
