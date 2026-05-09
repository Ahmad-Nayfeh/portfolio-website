import type { Metadata } from "next"
import Image from "next/image"
import Link from "next/link"
import { notFound } from "next/navigation"
import { ArrowLeft, ArrowUpRight, Github, ExternalLink } from "lucide-react"
import { getProjectBySlug, getRelatedProjects, getAllProjects } from "@/lib/content"
import { formatDate } from "@/lib/utils"
import MarkdownRenderer from "@/components/MarkdownRenderer"
import type { FullProject } from "@/types"
import RelatedProjects from "@/components/RelatedProjects"

interface PageProps {
  params: Promise<{ slug: string }>
  searchParams?: Promise<{ [key: string]: string | string[] | undefined }>
}

export async function generateStaticParams() {
  const projects = await getAllProjects()
  return projects.map((project) => ({ slug: project.slug }))
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params
  const project = await getProjectBySlug(slug)
  if (!project) return { title: "Project Not Found" }
  return {
    title: `${project.title} | Ahmad Nayfeh`,
    description: project.excerpt,
  }
}

export default async function ProjectPage({ params }: PageProps) {
  const { slug } = await params
  const project: FullProject | null = await getProjectBySlug(slug)
  if (!project) notFound()

  const relatedProjects = await getRelatedProjects(project, 3)
  const direction = project.lang === "ar" ? "rtl" : "ltr"

  return (
    <div
      className="mx-auto w-full max-w-[1400px] px-6 pb-20 pt-8 md:px-10 lg:px-16"
      dir={direction}
    >
      <Link
        href="/projects"
        className="group inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground transition-colors hover:text-foreground"
      >
        <ArrowLeft size={12} className="transition-transform duration-300 group-hover:-translate-x-0.5" />
        <span className="border-b border-border transition-colors group-hover:border-accent">
          Back to projects
        </span>
      </Link>

      <article className="relative">
        {/* Decorative background orbs */}
        <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
          <div
            className="absolute -right-32 -top-32 h-[600px] w-[600px] opacity-[0.05]"
            style={{
              background: "radial-gradient(ellipse at center, hsl(var(--accent)), transparent 70%)",
            }}
          />
          <div
            className="absolute -left-32 top-0 h-[400px] w-[400px] opacity-[0.03]"
            style={{
              background: "radial-gradient(ellipse at center, hsl(326 100% 62%), transparent 70%)",
            }}
          />
        </div>

        {/* === Editorial header — meta column on the left, title on the right. === */}
        <header className="mt-12 grid grid-cols-12 gap-x-6 border-b border-border pb-10">
          <div className="col-span-12 lg:col-span-3">
            <div className="flex flex-row gap-x-6 rounded-xl border border-border/50 bg-card/60 p-5 backdrop-blur-md lg:flex-col lg:gap-y-5">
              {project.date && (
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                    Date
                  </div>
                  <time
                    dateTime={project.date}
                    className="mt-1 block font-display text-base text-foreground"
                  >
                    {formatDate(project.date)}
                  </time>
                </div>
              )}
              {project.tags && project.tags.length > 0 && (
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                    Stack
                  </div>
                  <div className="mt-2 flex flex-wrap gap-x-2 gap-y-1 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                    {project.tags.map((tag) => (
                      <span key={tag} className="border-b border-border">
                        #{tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {(project.githubLink || (project.liveDemoUrl && project.liveDemoUrl !== "#")) && (
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                    Links
                  </div>
                  <div className="mt-2 flex flex-col gap-2">
                    {project.githubLink && (
                      <a
                        href={project.githubLink}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-foreground"
                      >
                        <Github size={13} className="text-muted-foreground transition-colors group-hover:text-accent" />
                        <span className="border-b border-border transition-colors group-hover:border-accent">
                          Source code
                        </span>
                      </a>
                    )}
                    {project.liveDemoUrl && project.liveDemoUrl !== "#" && (
                      <a
                        href={project.liveDemoUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-foreground"
                      >
                        <ExternalLink
                          size={13}
                          className="text-muted-foreground transition-colors group-hover:text-accent"
                        />
                        <span className="border-b border-border transition-colors group-hover:border-accent">
                          Live demo
                        </span>
                      </a>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="col-span-12 mt-8 lg:col-span-9 lg:mt-0">
            <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-accent">
              Case Study
            </span>
            <h1 className="font-display text-display-xl mt-4 text-balance">
              {project.title}
            </h1>
            {project.excerpt && (
              <p className="font-display mt-6 max-w-2xl text-xl italic leading-snug text-muted-foreground md:text-2xl">
                {project.excerpt}
              </p>
            )}
          </div>
        </header>

        {project.coverImage && (
          <div className="relative my-10 aspect-[16/9] overflow-hidden rounded-xl border border-border/50 bg-secondary shadow-lg shadow-black/20 md:my-14">
            <Image
              src={project.coverImage}
              alt={project.title}
              fill
              className="object-cover"
              priority
              sizes="(max-width: 1400px) 100vw, 1400px"
            />
            <div className="pointer-events-none absolute inset-0 rounded-xl ring-1 ring-inset ring-border/20" />
          </div>
        )}

        {/* Body — single editorial column, full-width within container */}
        <div className="mx-auto mt-10">
          {project.content && (
            <div className="project-prose-container">
              <MarkdownRenderer content={project.content} />
            </div>
          )}

          {/* Closing rule */}
          <div className="mx-auto mt-20 border-t border-border pt-8">
            <div className="flex flex-wrap items-center justify-between gap-3 font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
              <span>End of case study</span>
              {project.githubLink && (
                <a
                  href={project.githubLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group inline-flex items-center gap-2 text-foreground"
                >
                  <span className="border-b border-border transition-colors group-hover:border-accent">
                    Read the source
                  </span>
                  <ArrowUpRight
                    size={12}
                    className="transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
                  />
                </a>
              )}
            </div>
          </div>
        </div>
      </article>

      {relatedProjects.length > 0 && (
        <section className="mt-24 border-t border-border pt-16">
          <div className="mb-10 flex items-center gap-3">
            <span aria-hidden className="h-px w-8 bg-accent" />
            <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
              More case studies
            </span>
          </div>
          <h2 className="font-display text-display-md mb-10 text-balance">
            Related projects
          </h2>
          <RelatedProjects projects={relatedProjects} />
        </section>
      )}
    </div>
  )
}
