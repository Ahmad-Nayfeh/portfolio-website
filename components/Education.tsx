import type { EducationItem } from "@/types"

interface EducationProps {
  education: EducationItem[]
}

/**
 * Editorial education block. Mirrors the Experience component layout so both
 * sit side-by-side on the about page with the same visual cadence.
 */
export default function Education({
  education = [
    {
      degree: "Master of Computer Science",
      institution: "University of Technology",
      location: "San Francisco, CA",
      startDate: "2016",
      endDate: "2018",
      description:
        "Specialized in Human-Computer Interaction and Web Technologies.",
    },
    {
      degree: "Bachelor of Science in Computer Science",
      institution: "State University",
      location: "Boston, MA",
      startDate: "2012",
      endDate: "2016",
      description:
        "Graduated with honors. Focused on software engineering and web development.",
    },
  ],
}: EducationProps) {
  return (
    <section className="mt-20">
      <div className="mb-8 flex items-center gap-3">
        <span aria-hidden className="h-px w-8 bg-accent" />
        <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
          Studies · Formal training
        </span>
      </div>
      <h2 className="font-display text-display-md mb-8 text-balance">
        Education
      </h2>

      <ol className="border-t border-border">
        {education.map((item, index) => (
          <li
            key={index}
            className="grid grid-cols-12 gap-x-4 gap-y-2 border-b border-border py-6"
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
              <h3 className="font-display text-xl leading-snug text-foreground">
                {item.degree}
              </h3>
              {item.institution && (
                <div className="mt-1 font-mono text-[11px] uppercase tracking-[0.2em] text-accent">
                  {item.institution}
                </div>
              )}
              {item.description && (
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
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
