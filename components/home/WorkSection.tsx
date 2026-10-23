import SectionHeader from "@/components/drawing/SectionHeader"
import type { WorkContent } from "@/types"

export default function WorkSection({ work }: { work: WorkContent }) {
  return (
    <section id="work" className="mx-auto w-full max-w-[1240px] scroll-mt-24 px-5 py-20 md:px-8 md:py-28">
      <SectionHeader
        number="01"
        kicker="At work · Alfanar"
        title={work.title}
        description={work.intro}
      />

      <div className="mb-10 flex flex-wrap items-center gap-2">
        <span className="label mr-2">Product line</span>
        {work.products.map((p) => (
          <span key={p} className="chip text-foreground/80">
            {p}
          </span>
        ))}
      </div>

      {/* Process: design → approval → release → support */}
      <ol className="grid grid-cols-1 gap-0 md:grid-cols-4">
        {work.steps.map((step, i) => (
          <li key={step.title} className="relative flex md:block">
            {/* connector */}
            <div aria-hidden className="relative flex w-10 shrink-0 justify-center md:hidden">
              <span className="absolute bottom-0 top-0 w-px bg-foreground/60" />
            </div>
            <div className="sheet relative my-2 flex-1 p-5 md:my-0 md:mr-6 md:h-full">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] font-semibold text-accent">S{i + 1}</span>
                <span className="label text-[9.5px]">Step {i + 1} / {work.steps.length}</span>
              </div>
              <h3 className="mt-4 text-lg font-semibold tracking-tight">{step.title}</h3>
              <p className="mt-2 text-[14.5px] leading-relaxed text-muted-foreground">{step.detail}</p>
            </div>
            {i < work.steps.length - 1 && (
              <svg
                aria-hidden
                className="absolute right-0 top-1/2 hidden -translate-y-1/2 text-foreground md:block"
                width="24"
                height="12"
                viewBox="0 0 24 12"
              >
                <path d="M0 6 H20 M15 1 L21 6 L15 11" stroke="currentColor" strokeWidth="1.4" fill="none" />
              </svg>
            )}
          </li>
        ))}
      </ol>

      {work.automation && (
        <div className="mt-8 grid grid-cols-1 gap-6 border-[1.5px] border-foreground bg-card p-6 md:grid-cols-[auto_1fr] md:items-center md:p-7">
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center bg-accent font-mono text-[12px] font-semibold text-accent-foreground">
              AI
            </span>
            <span className="label text-foreground">Automation</span>
          </div>
          <p className="text-[16px] leading-relaxed text-foreground/90 md:text-[17px]">{work.automation}</p>
        </div>
      )}
    </section>
  )
}
