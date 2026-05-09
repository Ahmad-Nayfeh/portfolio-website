import type { EducationItem } from "@/types"

interface EducationProps {
  education: EducationItem[]
}

export default function Education({
  education = [],
}: EducationProps) {
  return (
    <section className="mt-20">
      <div className="mb-8 flex items-center gap-3">
        <span aria-hidden className="h-px w-8" style={{ backgroundColor: "hsl(var(--section-accent))" }} />
        <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
          Studies · Formal training
        </span>
      </div>
      <h2 className="text-display-md mb-8 font-bold tracking-tight text-balance">
        Education
      </h2>

      <ol className="border-t" style={{ borderColor: "hsl(var(--border))" }}>
        {education.map((item, index) => (
          <li
            key={index}
            className="grid grid-cols-12 gap-x-4 gap-y-2 py-6"
            style={{ borderBottom: "1px solid hsl(var(--border))" }}
          >
            <div className="col-span-12 md:col-span-4">
              <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                {item.startDate} — {item.endDate}
              </div>
              {item.location && (
                <div className="mt-2 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground/80">
                  {item.location}
                </div>
              )}
            </div>
            <div className="col-span-12 md:col-span-8">
              <h3 className="text-xl font-bold leading-snug tracking-tight text-foreground">
                {item.degree}
              </h3>
              {item.institution && (
                <div className="mt-1 font-mono text-[11px] uppercase tracking-[0.2em]"
                  style={{ color: "hsl(var(--section-accent))" }}
                >
                  {item.institution}
                </div>
              )}
              {item.description && (
                <p className="mt-3 text-base leading-relaxed text-muted-foreground">
                  {item.description}
                </p>
              )}
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}
