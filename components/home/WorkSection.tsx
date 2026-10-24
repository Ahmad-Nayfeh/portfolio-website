import SectionHeader from "@/components/drawing/SectionHeader"
import type { WorkContent } from "@/types"

/** Current job in brief: role, products, what the work involves, and the automation side. */
export default function WorkSection({ work }: { work: WorkContent }) {
  return (
    <section id="experience" className="mx-auto w-full max-w-[1240px] scroll-mt-24 px-5 py-20 md:px-8 md:py-28">
      <SectionHeader kicker="Experience" title={work.title} description={work.intro} />

      <div className="mb-8 flex flex-wrap items-center gap-2">
        <span className="label mr-2">Products</span>
        {work.products.map((p) => (
          <span key={p} className="chip text-foreground/80">
            {p}
          </span>
        ))}
      </div>

      <ol className="grid grid-cols-1 border-[1.5px] border-foreground bg-card md:grid-cols-4">
        {work.steps.map((step, i) => (
          <li key={step.title} className="border-b border-border p-5 last:border-b-0 md:border-b-0 md:border-r md:last:border-r-0">
            <div className="font-mono text-[11px] font-semibold text-accent">{i + 1}</div>
            <h3 className="mt-2 text-lg font-semibold tracking-tight">{step.title}</h3>
            <p className="mt-2 text-[14.5px] leading-relaxed text-muted-foreground">{step.detail}</p>
          </li>
        ))}
      </ol>

      {work.automation && (
        <p className="mt-6 max-w-4xl border-l-[3px] border-accent pl-5 text-[16px] leading-relaxed text-foreground/90 md:text-[17px]">
          {work.automation}
        </p>
      )}
    </section>
  )
}
