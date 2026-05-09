import Image from "next/image"
import Link from "next/link"
import { ArrowUpRight } from "lucide-react"
import type { Project } from "@/types"
import { formatDate } from "@/lib/utils"

interface ProjectCardProps {
  project: Project
  index?: number
}

export default function ProjectCard({ project, index }: ProjectCardProps) {
  const tags = project.tags ?? []
  return (
    <article className="group flex flex-col">
      <Link
        href={`/projects/${project.slug}`}
        className="relative block aspect-[4/3] w-full overflow-hidden rounded-lg transition-shadow duration-300"
        style={{
          backgroundColor: "hsl(var(--card))",
        }}
      >
        <Image
          src={project.coverImage || "/placeholder.svg"}
          alt={project.title}
          fill
          sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw"
          className="object-cover transition-transform duration-700 ease-out group-hover:scale-[1.03]"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent" />
        <span className="absolute left-3 top-3 rounded px-2 py-1 font-mono text-[10px] uppercase tracking-[0.2em] backdrop-blur-sm"
          style={{
            backgroundColor: "hsl(var(--background) / 0.6)",
            color: "hsl(var(--foreground))",
            border: "1px solid hsl(var(--border) / 0.5)",
          }}
        >
          {typeof index === "number"
            ? `Case · ${String(index + 1).padStart(2, "0")}`
            : "Case Study"}
        </span>
      </Link>

      <div className="flex flex-1 flex-col pt-5">
        <div className="mb-3 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          {project.date && <time dateTime={project.date}>{formatDate(project.date)}</time>}
        </div>

        <Link href={`/projects/${project.slug}`} className="flex items-start justify-between gap-3">
          <h3 className="text-2xl font-bold leading-tight tracking-tight text-balance text-foreground transition-colors">
            {project.title}
          </h3>
          <ArrowUpRight
            size={18}
            className="mt-1 shrink-0 transition-all duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
            style={{ color: "hsl(var(--section-accent))" }}
          />
        </Link>

        {project.excerpt && (
          <p className="mt-3 line-clamp-3 text-base leading-relaxed text-muted-foreground">
            {project.excerpt}
          </p>
        )}

        {tags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[10px] uppercase tracking-[0.18em]">
            {tags.slice(0, 3).map((tag) => (
              <Link
                key={tag}
                href={`/projects?tag=${tag}`}
                className="transition-colors"
                style={{ color: "hsl(var(--section-accent))" }}
              >
                #{tag}
              </Link>
            ))}
            {tags.length > 3 && <span className="text-muted-foreground">+{tags.length - 3}</span>}
          </div>
        )}
      </div>
    </article>
  )
}
