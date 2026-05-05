import type { ExperienceItem } from "@/types"

interface ExperienceProps {
  experience: ExperienceItem[]
}

/**
 * Editorial CV block. A small mono kicker, a serif heading, then a vertical
 * stack of entries separated by hairlines — no rounded cards, no shadows.
 */
export default function Experience({
  experience = [
    {
      title: "Senior Frontend Developer",
      company: "Example Corp",
      location: "Remote",
      startDate: "2021-01",
      endDate: "Present",
      description:
        "Led the development of the company's main product using React and TypeScript. Implemented new features and improved performance.",
    },
    {
      title: "Frontend Developer",
      company: "Tech Startup",
      location: "New York, NY",
      startDate: "2018-06",
      endDate: "2020-12",
      description:
        "Developed and maintained multiple web applications using React, Redux, and CSS-in-JS.",
    },
  ],
}: ExperienceProps) {
  return (
    <section className="mt-20">
      <div className="mb-8 flex items-center gap-3">
        <span aria-hidden className="h-px w-8 bg-accent" />
        <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
          Career · Selected experience
        </span>
      </div>
      <h2 className="font-display text-display-md mb-8 text-balance">
        Work experience
      </h2>

      <ol className="border-t border-border">
        {experience.map((item, index) => (
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
                {item.title}
              </h3>
              {item.company && (
                <div className="mt-1 font-mono text-[11px] uppercase tracking-[0.2em] text-accent">
                  {item.company}
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
