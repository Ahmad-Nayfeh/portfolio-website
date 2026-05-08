/**
 * Manifesto — "How I think about the work" section.
 *
 * Replaces the chip-cloud Skills section on the homepage with three short
 * editorial paragraphs that describe Ahmad's engineering philosophy. Keeps
 * the section masthead pattern used elsewhere on the page.
 *
 * The three pillars are intentionally terse — each fits in three sentences.
 * Padding and a top rule give them weight without needing imagery.
 */

import SectionMasthead from "@/components/editorial/SectionMasthead"

const PILLARS = [
  {
    number: "01",
    heading: "Build from the ground up",
    body: "Real systems don't behave like textbook models. The gap between theory and the factory floor is where most of the interesting engineering happens — and where the simplest solution is usually the one that actually ships.",
  },
  {
    number: "02",
    heading: "Measure twice, model once",
    body: "Every good ML model starts as a data-quality problem. Before reaching for a neural network, I want to know the signal's origin, its noise floor, and what a domain expert says about the edge cases. The model is the last step, not the first.",
  },
  {
    number: "03",
    heading: "Write it down",
    body: "Documentation is a form of engineering. A system that can't be explained can't be trusted, extended, or handed off. Writing forces clarity — and clarity is what separates a prototype from a product.",
  },
] as const

export default function Manifesto() {
  return (
    <section className="py-16 md:py-24">
      <SectionMasthead
        kicker="Engineering philosophy"
        title="How I think about the work"
      />

      <div className="grid grid-cols-1 gap-x-8 gap-y-12 md:grid-cols-3">
        {PILLARS.map(({ number, heading, body }) => (
          <div
            key={number}
            className="group flex flex-col gap-4 border-t border-border pt-6 transition-colors hover:border-accent"
          >
            <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground transition-colors group-hover:text-accent">
              {number}
            </span>
            <h3 className="font-display text-xl leading-snug tracking-editorial text-foreground">
              {heading}
            </h3>
            <p className="text-sm leading-relaxed text-muted-foreground">
              {body}
            </p>
          </div>
        ))}
      </div>
    </section>
  )
}
