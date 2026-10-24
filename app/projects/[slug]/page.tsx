import type { Metadata, Route } from "next"
import Image from "next/image"
import Link from "next/link"
import { notFound } from "next/navigation"
import { ArrowLeft, ArrowRight, ArrowUpRight, Github } from "lucide-react"
import { getProjectBySlug, getAllProjects } from "@/lib/content"
import { formatDate } from "@/lib/utils"
import MarkdownRenderer from "@/components/MarkdownRenderer"
import ProjectCard from "@/components/ProjectCard"
import TitleBlock, { type TitleBlockCell } from "@/components/drawing/TitleBlock"
import { hasLink, splitTitle } from "@/components/projects/projectName"
import { projectTiers, tierOf } from "@/config/site"

interface PageProps {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  const projects = await getAllProjects()
  return projects.map((project) => ({ slug: project.slug }))
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params
  const project = await getProjectBySlug(slug)
  if (!project) return { title: "Project not found" }
  return {
    title: splitTitle(project.title).name,
    description: project.excerpt,
    openGraph: { images: project.coverImage ? [project.coverImage] : [] },
  }
}

/** Project write-ups start with their own "# Title"; the page header already shows it. */
function stripLeadingH1(md: string) {
  return md.replace(/^\s*#\s+[^\n]+\n/, "")
}

export default async function ProjectPage({ params }: PageProps) {
  const { slug } = await params
  const project = await getProjectBySlug(slug)
  if (!project) notFound()

  const all = await getAllProjects()
  const tierId = tierOf(project.category)
  const tier = projectTiers.find((t) => t.id === tierId)!
  const idx = all.findIndex((p) => p.slug === project.slug)
  const prev = idx > 0 ? all[idx - 1] : null
  const next = idx >= 0 && idx < all.length - 1 ? all[idx + 1] : null
  const related = all.filter((p) => p.slug !== project.slug && tierOf(p.category) === tierId).slice(0, 3)

  const isArabic = project.lang === "ar"
  const { name, tagline } = splitTitle(project.title)
  const hasCover = project.coverImage && !project.coverImage.startsWith("/placeholder")

  const spec: TitleBlockCell[] = [
    { label: "Type", value: tier.name },
    { label: "Date", value: formatDate(project.date) },
  ]
  if (project.status) {
    spec.push({
      label: "Status",
      span: 2,
      value: (
        <span className="inline-flex items-center gap-2">
          <span className="lamp scale-75" aria-hidden />
          {project.status}
        </span>
      ),
    })
  }
  if (project.role) spec.push({ label: "My role", value: project.role, span: 2 })
  if (project.result) spec.push({ label: "Result", value: project.result, span: 2, emphasis: true })
  spec.push({
    label: "Tools",
    span: 2,
    value: (
      <span className="flex flex-wrap gap-1.5">
        {project.tags.map((t) => (
          <span key={t} className="chip">
            {t}
          </span>
        ))}
      </span>
    ),
  })
  if (hasLink(project.liveDemoUrl) || hasLink(project.githubLink)) {
    spec.push({
      label: "Links",
      span: 2,
      value: (
        <span className="flex flex-wrap gap-x-5 gap-y-1">
          {hasLink(project.liveDemoUrl) && (
            <a href={project.liveDemoUrl} target="_blank" rel="noopener noreferrer" className="redline inline-flex items-center gap-1.5 font-medium">
              {project.linkLabel ?? "Live"} <ArrowUpRight size={13} />
            </a>
          )}
          {hasLink(project.githubLink) && (
            <a href={project.githubLink} target="_blank" rel="noopener noreferrer" className="redline inline-flex items-center gap-1.5 font-medium">
              <Github size={13} /> Source
            </a>
          )}
        </span>
      ),
    })
  }

  return (
    <div className="mx-auto w-full max-w-[1240px] px-5 pb-16 pt-8 md:px-8 md:pt-12">
      <Link
        href={"/projects" as Route}
        className="group inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground transition-colors hover:text-foreground"
      >
        <ArrowLeft size={12} className="transition-transform group-hover:-translate-x-0.5" />
        All projects
      </Link>

      <article>
        <header className="mt-8 grid grid-cols-1 gap-x-12 gap-y-8 border-b-[1.5px] border-foreground pb-10 lg:grid-cols-12">
          <div className="lg:col-span-7" dir={isArabic ? "rtl" : undefined} lang={isArabic ? "ar" : undefined}>
            <span className="label" dir="ltr">
              {tier.name}
            </span>
            <h1 className="mt-4 font-display text-[clamp(2.25rem,5.5vw,4rem)] leading-[1.02] text-balance">{name}</h1>
            {tagline && <p className="mt-4 text-xl font-medium text-foreground/80 md:text-2xl">{tagline}</p>}
            {project.excerpt && <p className="mt-5 max-w-2xl text-[1.05rem] leading-relaxed text-muted-foreground">{project.excerpt}</p>}
          </div>
          <div className="lg:col-span-5 lg:pt-8">
            <TitleBlock cols={2} cells={spec} />
          </div>
        </header>

        {project.gallery.length > 0 ? (
          <div className="sheet mt-10 overflow-hidden">
            <div className="flex items-center justify-between border-b border-border px-4 py-2">
              <span className="label text-[9.5px]">Screenshots</span>
              <span className="label text-[9.5px]">{project.gallery.length}</span>
            </div>
            <div className="flex flex-wrap items-end justify-center gap-4 px-4 py-8 md:gap-8 md:px-10 md:py-12">
              {project.gallery.map((src, i) => (
                <div key={src} className="phone w-[42%] max-w-[240px] sm:w-[28%]">
                  <Image src={src} alt={`${name} screen ${i + 1}`} width={540} height={1140} sizes="(max-width: 640px) 42vw, 240px" priority={i === 0} />
                </div>
              ))}
            </div>
          </div>
        ) : (
          hasCover && (
            <div className="sheet relative mt-10 aspect-[16/9] overflow-hidden">
              <Image src={project.coverImage} alt="" fill className="object-cover" priority sizes="(max-width: 1240px) 100vw, 1240px" />
            </div>
          )
        )}

        <div
          className="mx-auto mt-12 max-w-[820px] md:mt-16"
          dir={isArabic ? "rtl" : undefined}
          lang={isArabic ? "ar" : undefined}
        >
          {project.content && (
            <div className="project-prose-container">
              <MarkdownRenderer content={stripLeadingH1(project.content)} />
            </div>
          )}
        </div>
      </article>

      {/* Previous / next sheet */}
      {(prev || next) && (
        <nav aria-label="More projects" className={`mt-20 grid grid-cols-1 border-[1.5px] border-foreground ${prev && next ? "md:grid-cols-2" : ""}`}>
          {prev && (
            <Link href={`/projects/${prev.slug}` as Route} className="group border-b border-border p-5 transition-colors last:border-b-0 hover:bg-card md:border-b-0 md:border-r md:last:border-r-0">
              <span className="label inline-flex items-center gap-2">
                <ArrowLeft size={12} /> Newer
              </span>
              <span className="mt-2 block text-lg font-semibold tracking-tight">
                <span className="redline">{splitTitle(prev.title).name}</span>
              </span>
            </Link>
          )}
          {next && (
            <Link href={`/projects/${next.slug}` as Route} className="group p-5 text-right transition-colors hover:bg-card">
              <span className="label inline-flex items-center gap-2">
                Older <ArrowRight size={12} />
              </span>
              <span className="mt-2 block text-lg font-semibold tracking-tight">
                <span className="redline">{splitTitle(next.title).name}</span>
              </span>
            </Link>
          )}
        </nav>
      )}

      {related.length > 0 && (
        <section className="mt-20">
          <div className="mb-8 flex items-center gap-3">
            <span className="label">More in {tier.name}</span>
            <span aria-hidden className="h-px flex-1 bg-border" />
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            {related.map((p) => (
              <ProjectCard key={p.slug} project={p} />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
