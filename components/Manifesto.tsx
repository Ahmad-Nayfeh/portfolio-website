import SectionMasthead from "@/components/editorial/SectionMasthead"
import StaggerGroup from "@/components/ui/StaggerGroup"

const PILLARS = [
  {
    number: "01",
    heading: "Build from the ground up",
    body: "Real systems don't behave like textbook models. The gap between theory and practice is where most of the interesting engineering happens — and where the simplest solution is usually the one that actually ships.",
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
    <section className="zone-teal py-16 md:py-24">
      <SectionMasthead
        kicker="Engineering philosophy"
        title="How I think about the work"
      />

      <StaggerGroup
        staggerDelay={120}
        direction="up"
        distance={16}
        className="grid grid-cols-1 gap-x-8 gap-y-12 md:grid-cols-3"
      >
        {PILLARS.map(({ number, heading, body }) => (
          <div
            key={number}
            className="group flex flex-col gap-4 pt-6 transition-all duration-300"
            style={{
              borderTop: "1px solid hsl(var(--border))",
            }}
          >
            <span className="font-mono text-[10px] uppercase tracking-[0.22em] transition-colors"
              style={{
                color: "hsl(var(--section-accent))",
              }}
            >
              {number}
            </span>
            <h3 className="text-xl font-bold leading-snug tracking-tight text-foreground">
              {heading}
            </h3>
            <p className="text-base leading-relaxed text-muted-foreground">
              {body}
            </p>
          </div>
        ))}
      </StaggerGroup>
    </section>
  )
}
