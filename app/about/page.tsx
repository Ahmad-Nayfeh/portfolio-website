import type { Metadata } from "next"
import { FileDown, Mail } from "lucide-react"
import { getAboutContent } from "@/lib/content"
import MarkdownRenderer from "@/components/MarkdownRenderer"
import SectionHeader from "@/components/drawing/SectionHeader"
import TitleBlock from "@/components/drawing/TitleBlock"
import { siteConfig } from "@/config/site"
import type { EducationItem, ExperienceItem } from "@/types"

export const metadata: Metadata = {
  title: "About",
  description: "Electrical design engineer in medium-voltage switchgear who builds AI-assisted tools and ships software products.",
}

type TimelineEntry = {
  kind: "Work" | "Education"
  title: string
  org: string
  location: string
  start: string
  end: string
  description: string
}

function toEntries(experience: ExperienceItem[], education: EducationItem[]): TimelineEntry[] {
  const work = experience.map((e) => ({
    kind: "Work" as const,
    title: e.title,
    org: e.company,
    location: e.location,
    start: String(e.startDate),
    end: String(e.endDate),
    description: e.description,
  }))
  const edu = education.map((e) => ({
    kind: "Education" as const,
    title: e.degree,
    org: e.institution,
    location: e.location,
    start: String(e.startDate),
    end: String(e.endDate),
    description: e.description,
  }))
  // Work first within the same start year, then newest first.
  return [...work, ...edu].sort((a, b) => {
    const d = Number(b.start.slice(0, 4)) - Number(a.start.slice(0, 4))
    return d !== 0 ? d : a.kind === "Work" ? -1 : 1
  })
}

function period(start: string, end: string) {
  const fmt = (s: string) => {
    if (/^\d{4}-\d{2}/.test(s)) {
      const d = new Date(`${s.slice(0, 7)}-01T00:00:00Z`)
      return new Intl.DateTimeFormat("en-US", { month: "short", year: "numeric", timeZone: "UTC" }).format(d)
    }
    return s
  }
  return start === end ? fmt(start) : `${fmt(start)} – ${fmt(end)}`
}

export default async function AboutPage() {
  const about = await getAboutContent()
  const entries = toEntries(about.experience, about.education)
  const current = about.experience[0]

  return (
    <div className="mx-auto w-full max-w-[1240px] px-5 pb-16 pt-10 md:px-8 md:pt-16">
      <header className="mb-14">
        <span className="label">Sheet 04 · About</span>
        <h1 className="mt-4 font-display text-[clamp(2.5rem,6vw,4.25rem)] leading-[1]">{about.name}</h1>
        <p className="mt-4 font-mono text-[13px] font-medium uppercase tracking-[0.16em] text-accent">{about.tagline}</p>
      </header>

      <section className="grid grid-cols-1 gap-x-14 gap-y-10 lg:grid-cols-12">
        <div className="lg:col-span-7">
          <div className="about-story text-[1.1rem]">
            <MarkdownRenderer content={about.bio} />
          </div>
        </div>
        <div className="lg:col-span-5">
          <TitleBlock
            cols={2}
            cells={[
              { label: "Role", value: current?.title ?? siteConfig.role, emphasis: true, span: 2 },
              { label: "Employer", value: current?.company ?? "", span: 2 },
              { label: "Based in", value: siteConfig.location },
              { label: "Since", value: current ? period(current.startDate, current.endDate).split(" – ")[0] : "" },
              { label: "Education", value: "B.Sc. EE, KFUPM · M.Sc. EE (in progress), KFUPM", span: 2 },
              { label: "Languages", value: about.languages.join(" · "), span: 2 },
            ]}
          />
          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <a href={siteConfig.resume} target="_blank" rel="noopener noreferrer" className="btn btn-primary">
              <FileDown size={14} /> Download CV
            </a>
            <a href={`mailto:${siteConfig.contact.email}`} className="btn btn-ghost">
              <Mail size={14} /> Email me
            </a>
          </div>
        </div>
      </section>

      {/* Timeline drawn as a vertical busbar with taps */}
      <section className="mt-24">
        <SectionHeader number="01" kicker="Timeline" title="Work and education" />
        <ol className="relative ml-2 border-l-[3px] border-foreground md:ml-[10.5rem]">
          {entries.map((e) => (
            <li key={`${e.kind}-${e.title}-${e.start}`} className="relative pb-10 pl-8 last:pb-0 md:pl-10">
              {/* tap node */}
              <span aria-hidden className="absolute -left-[9px] top-1.5 h-[15px] w-[15px] rounded-full border-[2.5px] border-foreground bg-background" />
              <span aria-hidden className="absolute left-[6px] top-[13px] h-px w-5 bg-foreground md:w-7" />
              <div className="md:absolute md:-left-[11rem] md:top-0.5 md:w-[9rem] md:text-right">
                <span className="label text-[10.5px]">{period(e.start, e.end)}</span>
              </div>
              <div className="sheet mt-2 p-5 md:mt-0">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className={e.kind === "Work" ? "font-mono text-[10.5px] font-semibold uppercase tracking-[0.14em] text-accent" : "label text-[10.5px]"}>
                    {e.kind}
                  </span>
                  <span className="label text-[10px]">{e.location}</span>
                </div>
                <h3 className="mt-2 text-lg font-semibold leading-snug tracking-tight">{e.title}</h3>
                <p className="text-[14.5px] font-medium text-foreground/75">{e.org}</p>
                {e.description && <p className="mt-3 text-[14.5px] leading-relaxed text-muted-foreground">{e.description}</p>}
              </div>
            </li>
          ))}
        </ol>
      </section>

      {about.skillGroups.length > 0 && (
        <section className="mt-24">
          <SectionHeader number="02" kicker="Skills" title="What I work with" />
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            {about.skillGroups.map((g) => (
              <div key={g.name} className="sheet p-5">
                <h3 className="label text-foreground">{g.name}</h3>
                <ul className="mt-4 border-t border-border">
                  {g.items.map((item, i) => (
                    <li key={item} className="flex gap-3 border-b border-border py-2.5 text-[14.5px] leading-snug last:border-b-0">
                      <span className="mt-[2px] font-mono text-[10.5px] text-muted-foreground">{String(i + 1).padStart(2, "0")}</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
