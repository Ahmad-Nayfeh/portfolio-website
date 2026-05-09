import Image from "next/image"
import Link from "next/link"
import { ArrowUpRight } from "lucide-react"
import type { Project } from "@/types"
import { formatDate } from "@/lib/utils"

/**
 * ProjectCard — like BlogCard, but project-shaped.
 *
 * May 2026: tag chips now link to the filtered projects index. Image has a
 * slight cobalt tint on hover (via mix-blend multiply + palette accent).
 * Focus ring matches the accent so keyboard navigation is consistent.
 */

interface ProjectCardProps {
  project: Project
  index?: number
}

/**
 * ProjectCard — like BlogCard, but project-shaped. The image holds a
 * monospaced "case study" sticker; the body uses the same serif title and
 * mono meta row to keep the editorial cadence on the homepage.
 */
export default function ProjectCard({ project, index }: ProjectCardProps) {
  const tags = project.tags ?? []
  return (
    <article className="group flex flex-col">
      <Link
        href={`/projects/${project.slug}`}
        className="relative block aspect-[4/3] w-full overflow-hidden bg-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        <Image
          src={project.coverImage || "/placeholder.svg"}
          alt={project.title}
          fill
          sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw"
          className="object-cover transition-transform duration-700 ease-out group-hover:scale-[1.03]"
        />
        <span className="absolute left-3 top-3 bg-background/85 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.2em] text-foreground backdrop-blur-sm">
          {typeof index === "number"
            ? `Case · ${String(index + 1).padStart(2, "0")}`
            : "Case Study"}
        </span>
      </Link>

      <div className="flex flex-1 flex-col pt-5">
        <div className="mb-3 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          {project.date && <time dateTime={project.date}>{formatDate(project.date)}</time>}
        </div>

        <Link href={`/projects/${project.slug}`} className="flex items-start justify-between gap-3">
          <h3 className="font-display text-2xl leading-tight tracking-editorial text-balance text-foreground transition-colors group-hover:text-accent">
            {project.title}
          </h3>
          <ArrowUpRight
            size={18}
            className="mt-1 shrink-0 text-muted-foreground transition-all duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-accent"
          />
        </Link>

        {project.excerpt && (
          <p className="mt-3 line-clamp-3 text-sm leading-relaxed text-muted-foreground">
            {project.excerpt}
          </p>
        )}

        {tags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            {tags.slice(0, 3).map((tag) => (
              <Link
                key={tag}
                href={`/projects?tag=${tag}`}
                className="transition-colors hover:text-accent"
              >
                #{tag}
              </Link>
            ))}
            {tags.length > 3 && <span>+{tags.length - 3}</span>}
          </div>
        )}
      </div>
    </article>
  )
}
