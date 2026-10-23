import type { Metadata, Route } from "next"
import Link from "next/link"
import { ArrowUpRight } from "lucide-react"
import ProjectCard from "@/components/ProjectCard"
import SectionHeader from "@/components/drawing/SectionHeader"
import TitleBlock, { type TitleBlockCell } from "@/components/drawing/TitleBlock"
import { getAllProjects } from "@/lib/content"
import { projectTiers, tierOf } from "@/config/site"
import { projectYear, splitTitle } from "@/components/projects/projectName"
import type { Project } from "@/types"

export const metadata: Metadata = {
  title: "Projects",
  description: "Products, research, and smaller experiments by Ahmad Nayfeh.",
}

const prefix: Record<string, string> = { products: "P", research: "R", lab: "L" }

export default async function ProjectsPage() {
  const projects = await getAllProjects()
  const byTier = projectTiers.map((tier) => ({
    tier,
    items: projects.filter((p) => tierOf(p.category) === tier.id),
  }))
  const latest = projects[0]
  const summaryCells: TitleBlockCell[] = byTier.map(({ tier, items }) => ({
    label: tier.name,
    value: <span className="font-mono text-xl font-semibold">{String(items.length).padStart(2, "0")}</span>,
  }))
  if (latest) {
    summaryCells.push({ label: "Latest", value: <span className="font-medium">{splitTitle(latest.title).name}</span>, span: 3 })
  }

  return (
    <div className="mx-auto w-full max-w-[1240px] px-5 pb-16 pt-10 md:px-8 md:pt-16">
      <header className="mb-16 grid grid-cols-1 gap-8 lg:grid-cols-12 lg:items-end">
        <div className="lg:col-span-7">
          <span className="label">Sheet 02 · Projects</span>
          <h1 className="mt-4 font-display text-[clamp(2.5rem,6vw,4.25rem)] leading-[1]">Projects</h1>
          <p className="mt-5 max-w-xl text-[1.1rem] leading-relaxed text-foreground/80">
            Grouped by how far they went: products that real people use, research with measured results, and a lab of smaller tools.
          </p>
        </div>
        <div className="lg:col-span-5">
          <TitleBlock cols={3} cells={summaryCells} />
        </div>
      </header>

      <nav aria-label="Tiers" className="mb-14 flex flex-wrap gap-2">
        {byTier.map(({ tier, items }) => (
          <a key={tier.id} href={`#${tier.id}`} className="btn btn-ghost px-3 py-2 text-[11px]">
            {tier.name} <span className="text-muted-foreground">{items.length}</span>
          </a>
        ))}
      </nav>

      <div className="space-y-24">
        {byTier.map(({ tier, items }, ti) =>
          items.length === 0 ? null : (
            <section key={tier.id} id={tier.id} className="scroll-mt-24">
              <SectionHeader number={String(ti + 1).padStart(2, "0")} kicker={`${items.length} ${items.length === 1 ? "project" : "projects"}`} title={tier.name} description={tier.blurb} />
              {tier.id === "lab" ? (
                <LabList items={items} />
              ) : (
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
                  {items.map((p, i) => (
                    <ProjectCard key={p.slug} project={p} sheetNo={`${prefix[tier.id]}-${String(i + 1).padStart(2, "0")}`} />
                  ))}
                </div>
              )}
            </section>
          ),
        )}
      </div>
    </div>
  )
}

function LabList({ items }: { items: Project[] }) {
  return (
    <ul className="border-t-[1.5px] border-foreground">
      {items.map((p, i) => {
        const { name, tagline } = splitTitle(p.title)
        return (
          <li key={p.slug}>
            <Link
              href={`/projects/${p.slug}` as Route}
              className="group grid grid-cols-[3rem_1fr_auto] items-baseline gap-x-4 border-b border-border py-5 transition-colors hover:bg-card/70 md:grid-cols-[4rem_1fr_14rem_4rem_auto] md:px-3"
            >
              <span className="font-mono text-[11px] font-semibold text-accent">L-{String(i + 1).padStart(2, "0")}</span>
              <span>
                <span className="block text-[1.05rem] font-semibold tracking-tight">
                  <span className="redline">{name}</span>
                  {tagline && <span className="font-normal text-muted-foreground">: {tagline}</span>}
                </span>
                <span className="mt-1 block line-clamp-2 text-[14px] leading-relaxed text-muted-foreground">{p.excerpt}</span>
              </span>
              <span className="hidden flex-wrap gap-1.5 md:flex">
                {p.tags.slice(0, 2).map((t) => (
                  <span key={t} className="chip">
                    {t}
                  </span>
                ))}
              </span>
              <span className="label hidden md:block">{projectYear(p)}</span>
              <ArrowUpRight size={16} className="text-muted-foreground transition-colors group-hover:text-accent" />
            </Link>
          </li>
        )
      })}
    </ul>
  )
}
